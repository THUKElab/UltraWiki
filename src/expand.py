import os
import copy
from typing import Any
import torch
import numpy as np
from torch.nn.modules import Module
from tqdm import tqdm
from time import time
from model import Model
from utils import *
from inferencer import *


class CalEmbedding(Forward):
    def __init__(self, tup_id: int):
        super().__init__()
        self.tup_id = tup_id

    def forward(self, model: Module, batch_data: list[list[int]]) -> Any:
        max_len = max(len(sent) for sent in batch_data)
        input_ids = pad_to_length(batch_data, max_len, return_tensors=True)
        input_ids = input_ids.to(self.device)
        tup = model.forward(input_ids, CL=False)
        embs = (F.softmax(tup[0], dim=-1), tup[1])[self.tup_id].cpu()
        return embs


class Expan:
    # in order to unify the computation of cosine similarity and kl divergence
    similarity_metric_method2tup_id = {"kl_divergence": 0, "cosine_similarity": 1}
    similarity_metrics = [kl_div, cosine_similarity]

    def __init__(self, args):
        self.args = args
        self.CL = args.CL
        self.cache_ent_embeddings = args.cache_ent_embeddings
        self.similarity_metric_method = args.similarity_metric
        self.tup_id = Expan.similarity_metric_method2tup_id[
            self.similarity_metric_method
        ]
        self.similarity_metric = Expan.similarity_metrics[self.tup_id]
        self.ent2ids = read_pkl(args.ent2ids)
        self.candidate_ents, self.ent2eid = load_candidate_ents(args.entities)
        self.vocab_size = len(self.candidate_ents)
        self.model = Model(vocab_size=self.vocab_size)
        if args.pretrained_model is not None:
            self.model.load_state_dict(torch.load(args.pretrained_model))
        # self.model.cuda()
        self.tokenizer = self.model.tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id
        # self.ent_embeddings = self.get_ent_embeddings()  # (N,vocab_size) or (N,768)
        # self.ent_embeddings = self.ent_embeddings.cuda()
        self.ent_embeddings = None  # (N,vocab_size) or (N,768)
        self.pre_pos_group_seeds = None
        self.pre_neg_group_seeds = None

    def out_message(self, message: str):
        print(message)

    
    @torch.inference_mode()
    def get_ent_embeddings(self):
        if self.args.cache_ent_embeddings and os.path.exists(self.args.ent_embeddings):
            return torch.load(self.args.ent_embeddings)
        self.out_message("making ent_embeddings...")
        lens_by_ent = []
        total_input_ids = []
        for i, ent in enumerate(self.candidate_ents):
            input_ids = self.ent2ids[ent]
            total_input_ids.extend(input_ids)
            lens_by_ent.append(len(input_ids))
        stt = time()
        total_embs = Inferencer(
            model=self.model,
            data=total_input_ids,
            batch_size=self.args.batch_size_in_cal_embs,
            forward_fn=CalEmbedding(tup_id=self.tup_id),
            nproc=torch.cuda.device_count(),
        ).inference()
        print(f"cost time: {time()-stt:.2f}")
        total_embs = torch.cat(total_embs, dim=0)
        embs_by_ent = [embs.mean(0) for embs in torch.split(total_embs, lens_by_ent)]
        embs_by_ent = torch.stack(embs_by_ent)

        if self.similarity_metric_method == "cosine_similarity":
            embs_by_ent = F.normalize(embs_by_ent, p=2, dim=-1)
        if self.cache_ent_embeddings:
            print(f"writing ent_embeddings to '{self.args.ent_embeddings}'")
            torch.save(embs_by_ent, self.args.ent_embeddings)
        return embs_by_ent

    # select k most similar to seeds from candidates
    # if you want the selected k entities without seeds,
    #   you should set 'remove_seeds' to True and 'seeds' to seed_eids
    def select(
        self,
        emb_cands: Tensor,
        emb_seeds: Tensor,
        k: int,
        remove_seeds=False,
        seeds: Tensor = None,
        reduction = "mean"
    ) -> Tensor:
        similarities_by_cand = self.similarity_metric(emb_cands, emb_seeds, reduction=reduction)  # (N)
        if remove_seeds:
            top_indices = torch.topk(similarities_by_cand, k + len(seeds)).indices
            top_indices = tensor_remove(top_indices, seeds)[:k]
        else:
            top_indices = torch.topk(similarities_by_cand, k).indices
        return top_indices

    def rank(self, top_indices, pos_seeds, neg_seeds):
        seg_length = self.args.seg_length
        neg_sim = self.similarity_metric(
            self.ent_embeddings[top_indices], 
            self.ent_embeddings[neg_seeds], 
            reduction="mean",
        )
        
        scores = -neg_sim
        _top_indices = seg_sort(scores, seg_length=seg_length, descending=True)
        return top_indices[_top_indices]
    
    def add_seeds_text(self, pos_seeds: list[str], neg_seeds: list[str]):
        seeds_text = ""
        for ent in pos_seeds:
            seeds_text=seeds_text+f"[POS_SEP] {ent} "
        for ent in neg_seeds:
            seeds_text=seeds_text+f"[NEG_SEP] {ent} "
        seeds_text = self.tokenizer.encode(
            seeds_text, add_special_tokens=False, truncation=True, max_length=150
        )
        new_ent2ids = {}
        for ent, ids_list in self.ent2ids.items():
            new_ent2ids[ent] = [ids+seeds_text for ids in ids_list]
        self.ent2ids = new_ent2ids

    def expand(
        self, pos_seeds: list[str], neg_seeds: list[str], target_size=200,
        pos_group_seeds=None, neg_group_seeds=None,
    ) -> list[str]:
        if self.CL:
            pre_ent2ids = self.ent2ids
            if pos_group_seeds != self.pre_pos_group_seeds or neg_group_seeds != self.pre_neg_group_seeds:
                self.add_seeds_text(pos_group_seeds, neg_group_seeds)
                self.ent_embeddings = self.get_ent_embeddings()
                self.ent_embeddings.cuda()
        elif self.ent_embeddings is None:
            self.ent_embeddings = self.get_ent_embeddings()
            self.ent_embeddings.cuda()
        emb_cands = self.ent_embeddings
        device = emb_cands.device

        pos_seeds = torch.tensor(
            [self.ent2eid[ent] for ent in pos_seeds], dtype=torch.int64
        ).to(device)
        
        top_indices = self.select(
            emb_cands=emb_cands,
            emb_seeds=emb_cands[pos_seeds],
            k=target_size,
            remove_seeds=True,
            seeds=pos_seeds,
            reduction="mean"
        )
        if self.args.expand_task == "NegESE":
            neg_seeds = torch.tensor(
                [self.ent2eid[ent] for ent in neg_seeds], dtype=torch.int64
            ).to(device)
            top_indices = self.rank(top_indices, pos_seeds, neg_seeds)

        if self.CL:
            self.ent2ids = pre_ent2ids
            self.pre_pos_group_seeds = copy.deepcopy(pos_group_seeds)
            self.pre_neg_group_seeds = copy.deepcopy(neg_group_seeds)
        return [self.candidate_ents[eid] for eid in top_indices.cpu().numpy()]
