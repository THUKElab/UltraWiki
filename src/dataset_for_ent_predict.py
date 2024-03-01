import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import *
np.random.seed(42)



class EntityPredictionDataset(Dataset):
    def __init__(
            self, ent2ids: dict[str, list[list[int]]], ent2eid: dict[str, int], 
                tokenizer, sample_rate:float=0.2, cln2groups:dict=None):
        input_ids, labels = [], []
        
        if cln2groups is not None:
            all_seeds_pair=[(group["pos_seeds"],group["neg_seeds"]) for groups in cln2groups.values() for group in groups.values()]
            all_seeds_text=[]
            for pos_seeds,neg_seeds in all_seeds_pair:
                seeds_text = ""
                for ent in pos_seeds:
                    seeds_text=seeds_text+f"[POS_SEP] {ent} "
                for ent in neg_seeds:
                    seeds_text=seeds_text+f"[NEG_SEP] {ent} "
                seeds_text = tokenizer.encode(
                    seeds_text, add_special_tokens=False, truncation=True, max_length=150
                )
                all_seeds_text.append(seeds_text)

            print(f"making EntityPredictionDataset...")
            for ent, ids in tqdm(ent2ids.items()):
                
                eid = ent2eid[ent]
                sz = int(sample_rate*len(all_seeds_text))
                seeds_ids=np.random.choice(len(all_seeds_text),size=sz,replace=False)
                for seeds_id in seeds_ids:
                    seeds_text=all_seeds_text[seeds_id]
                    for _ids in ids:
                        input_ids.append(_ids+seeds_text)
                        labels.append(eid)
        else:
            print(f"making EntityPredictionDataset...")
            for ent, ids in tqdm(ent2ids.items()):
                eid = ent2eid[ent]
                for _ids in ids:
                    input_ids.append(_ids)
                    labels.append(eid)

        self.input_ids: list[list[int]] = input_ids
        self.labels: list[int] = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i]


def collate_for_ent_predict(batch):
    input_ids, labels = zip(*batch)
    max_length = max(len(x) for x in input_ids)
    input_ids = pad_to_length(input_ids, length=max_length, return_tensors=True)
    labels = torch.tensor(labels, dtype=torch.int64)
    return input_ids, labels


if __name__ == "__main__":
    from model import Model

    ent2ids = read_pkl("./data/ent2ids.pkl")
    candidate_ents, ent2eid = load_candidate_ents("./data/entities.txt")

    model = Model(vocab_size=len(candidate_ents))
    tokenizer = model.tokenizer

    cln2groups = read_json("./data/cln2groups.json")
    # dataset = EntityPredictionDataset(ent2ids=ent2ids, ent2eid=ent2eid, tokenizer=tokenizer)
    dataset = EntityPredictionDataset(
        ent2ids=ent2ids, ent2eid=ent2eid, 
        tokenizer=tokenizer, sample_rate=0.2, cln2groups=cln2groups)
    print()
    print(f"There are {len(dataset)} rows.")
    print()
    print(dataset[0])
