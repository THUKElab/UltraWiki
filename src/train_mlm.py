import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from time import time
import logging
from utils import *
from model import *
from dataset_for_ent_predict import *
from dataset_for_cl import *
np.random.seed(42)

class Trainer:
    def __init__(
        self, args, model, dataset1, loss_fn1=None, dataset2=None, loss_fn2=None
    ):
        self.CL = args.CL
        self.args = args
        self.local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{self.local_rank}")
        self.is_master =  self.local_rank == 0
        self.log = False
        if self.is_master and args.log_file is not None:
            self.log = True
            logging.basicConfig(
                filename=args.log_file,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                encoding="utf-8",
            )
        
        unfreeze_layers = ['encoder.layer.11', 'ent_pred_head', 'cl_head']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters, lr=args.lr
        )
        self.model = model.cuda()
        self.pre_epoch=self.get_pre_epoch()
        if self.pre_epoch>0:
            map_device=torch.device(f"cuda:{self.local_rank}")
            model_name=self.args.model_save_name.format(self.pre_epoch)
            model_save_path = os.path.join(self.args.model_save_dir, model_name)
            self.model.load_state_dict(torch.load(model_save_path,map_location=map_device))
            optimizer_name = self.args.optimizer_save_name.format(self.pre_epoch)
            optimizer_save_path = os.path.join(self.args.optimizer_save_dir, optimizer_name)
            self.optimizer.load_state_dict(torch.load(optimizer_save_path,map_location=map_device))
            self.out_message(f"train model from {self.pre_epoch+1}th epoch")
            self.out_message()
        self.model = DDP(self.model)

        self.sampler1 = DistributedSampler(dataset1)
        self.dataloader1 = DataLoader(
            dataset1,
            batch_size=args.batch_size,
            sampler=self.sampler1,
            collate_fn=collate_for_ent_predict,
            num_workers=4,
        )
        if loss_fn1 is None:
            loss_fn1 = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
        self.loss_fn1 = loss_fn1

        if self.CL:
            self.sampler2 = DistributedSampler(dataset2, drop_last=True)
            self.dataloader2 = DataLoader(
                dataset2,
                batch_size=args.cl_batch_size,
                sampler=self.sampler2,
                collate_fn=collate_for_cl,
                num_workers=4,
                drop_last=True,
            )
            
            if loss_fn2 is None:
                loss_fn2 = InfoNCELoss(temperature=args.temperature)
            self.loss_fn2 = loss_fn2


    def get_pre_epoch(self):
        for epoch in range(50,0,-1):
            model_name=self.args.model_save_name.format(epoch)
            model_save_path = os.path.join(self.args.model_save_dir, model_name)
            if os.path.exists(model_save_path):
                return epoch
        return 0

    @staticmethod
    def only_master(func):
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs) if self.is_master else None
        return wrapper

    @only_master
    def out_message(self, message: str = ""):
        print(message)
        if self.log:
            logging.info(message)

    @only_master
    def save_checkpoint(self, model_name, optimizer_name):
        if not os.path.exists(self.args.model_save_dir):
            os.mkdir(self.args.model_save_dir)
        if not os.path.exists(self.args.optimizer_save_dir):
            os.mkdir(self.args.optimizer_save_dir)
        model_save_path = os.path.join(self.args.model_save_dir, model_name)
        self.out_message(f'saving model to \'{model_save_path}\'')
        torch.save(self.model.module.state_dict(), model_save_path)
        optimizer_save_path = os.path.join(self.args.optimizer_save_dir, optimizer_name)
        self.out_message(f'saving optimizer to \'{optimizer_save_path}\'')
        torch.save(self.optimizer.state_dict(), optimizer_save_path)

    def train_epoch_for_ent_pred(self, epoch: int):
        self.sampler1.set_epoch(epoch)
        n = len(self.dataloader1)
        sum_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(self.dataloader1, 1):
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            pred, _ = model.forward(input_ids, CL=False)
            loss = self.loss_fn1(pred, labels) / self.args.accumulation_steps
            loss.backward()
            sum_loss += loss.item()
            if (batch_idx % self.args.accumulation_steps == 0) or batch_idx == n:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.is_master and batch_idx % self.args.steps_per_print == 0:
                avg_loss = sum_loss / self.args.steps_per_print
                message = f"\tStep: {batch_idx:<5}  Loss: {avg_loss:>.4f}"
                self.out_message(message)
                sum_loss = 0.0

    def train_epoch_for_cl(self, epoch: int):
        self.sampler2.set_epoch(epoch)
        n = len(self.dataloader2)
        sum_loss = 0.0
        for batch_idx, (sents1, sents2, srcs1, srcs2) in enumerate(self.dataloader2, 1):
            sents1, sents2 = sents1.cuda(), sents2.cuda()
            z1, _ = model.forward(sents1, CL=True)
            z2, _ = model.forward(sents2, CL=True)
            loss = self.loss_fn2(z1, z2, srcs1, srcs2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.item()
            if self.is_master and batch_idx % self.args.steps_per_print == 0:
                avg_loss = sum_loss / self.args.steps_per_print
                message = f"\tCL Step: {batch_idx:<5}  Loss: {avg_loss:>.4f}"
                self.out_message(message)
                sum_loss = 0.0

    def train(self):
        self.model.train()
        for epoch in range(self.pre_epoch+1, self.args.n_epoch + 1):
            self.out_message(f"Epoch: {epoch}")
            self.train_epoch_for_ent_pred(epoch)
            if self.CL:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.args.cl_lr
                self.train_epoch_for_cl(epoch)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.args.lr

            self.save_checkpoint(
                self.args.model_save_name.format(epoch),
                self.args.optimizer_save_name.format(epoch))
            self.out_message()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/")
    parser.add_argument("--ent2ids", type=str, default="ent2ids.pkl")
    parser.add_argument("--entities", type=str, default="entities.txt")
    parser.add_argument("--model_save_dir", type=str, default="./model")
    parser.add_argument("--optimizer_save_dir", type=str, default="./optimizer")
    parser.add_argument("--model_save_name", type=str, default="epoch_{}.pt")
    parser.add_argument("--optimizer_save_name", type=str, default="epoch_{}.pt")
    parser.add_argument("--n_epoch", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--cl_lr", type=float, default=2e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--smoothing", type=float, default=0.075)
    parser.add_argument("--steps_per_print", type=int, default=100)
    parser.add_argument("--log_file", type=str, default="training.log")
    parser.add_argument("--CL", action="store_true")
    parser.add_argument("--cln2groups", type=str, default="cln2groups.json")
    parser.add_argument("--cl_batch_size", type=int, default=128)
    parser.add_argument("--max_sent_pairs_per_ent_pair", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--sample_rate", type=float, default=0.2)


    args = parser.parse_args()
    args.ent2ids = os.path.join(args.data, args.ent2ids)
    args.entities = os.path.join(args.data, args.entities)
    args.cln2groups = os.path.join(args.data, args.cln2groups)
    return args

def create_subset(dataset, ratio=0.01):
    dataset_size = len(dataset)
    subset_size = int(np.floor(ratio * dataset_size))
    indices = torch.randperm(dataset_size)[:subset_size]
    return Subset(dataset, indices)

if __name__ == "__main__":
    args = parse_args()
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu_id)
    dist.init_process_group("nccl")

    ent2ids = read_pkl(args.ent2ids)
    candidate_ents, ent2eid = load_candidate_ents(args.entities)
    model = Model(vocab_size=len(candidate_ents))

    cln2groups = None
    if args.CL:
        cln2groups = read_json(args.cln2groups)

    ds1 = EntityPredictionDataset(
        ent2ids=ent2ids, ent2eid=ent2eid, 
        tokenizer=model.tokenizer, sample_rate=args.sample_rate,
        cln2groups=cln2groups,
    )

    ds2 = None
    if args.CL:
        ds2 = CLDataset(
            ent2ids=ent2ids,
            cln2groups=cln2groups,
            tokenizer=model.tokenizer,
            max_sent_pairs_per_ent_pair=args.max_sent_pairs_per_ent_pair,
        )
        

    trainer = Trainer(args=args, model=model, dataset1=ds1, dataset2=ds2)
    trainer.train()

    dist.destroy_process_group()
