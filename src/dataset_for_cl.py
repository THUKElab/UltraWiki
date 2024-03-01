import os
import random
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *
random.seed(42)
np.random.seed(42)


def convert_cln2groups(
    cln2groups: dict[str, dict[str, dict[str, list[str]]]]
) -> dict[str, dict[int, dict[str, list[str]]]]:
    new_cln2groups = {}
    for cln, groups in cln2groups.items():
        new_cln2groups[cln] = {
            int(group_id): group for group_id, group in groups.items()
        }
    return new_cln2groups


class SentencePairsMaker:
    def __init__(
        self,
        cln2groups: dict[str, dict[str, dict[str, list[str]]]],
        ent2ids: dict[str, dict[str, list[list[str]]]],
        tokenizer,
        max_sent_pairs_per_ent_pair: int,
    ):
        self.cln2groups = convert_cln2groups(cln2groups)
        self.ent2ids = ent2ids
        self.tokenizer = tokenizer
        self.max_sent_pairs_per_ent_pair = max_sent_pairs_per_ent_pair
        # src indicates which group the entity is from
        #  it is a triple like (cln,group_id,idx) such as ('countries',1,0)
        #  sid is the hash of src and uniquely identifies a src
        #  src_pair means that two srcs are from the same query
        self.src2sid, self.sid2ents = self.make_src2sid()
        self.sid2src = {sid:src for src,sid in self.src2sid.items()}

    def make_src2sid(
        self,
    ) -> tuple[dict[tuple[str, int, int], int], dict[int, list[str]],]:
        src2sid, sid2ents = {}, {}
        sid_cnt = 1
        for cln, groups in self.cln2groups.items():
            for group_id, group in groups.items():
                pos_ents = group["pos_ents"]
                neg_ents = group["neg_ents"]
                pos_src, neg_src = (cln, group_id, 1), (cln, group_id, -1)
                pos_sid, neg_sid = sid_cnt, -sid_cnt
                src2sid[pos_src], src2sid[neg_src] = pos_sid, neg_sid
                sid2ents[pos_sid], sid2ents[neg_sid] = pos_ents, neg_ents
                sid_cnt += 1
        return src2sid, sid2ents

    def make_sentence_pairs(self, ent1, ent2) -> list[tuple[list[int], list[int]]]:
        sents1, sents2 = self.ent2ids[ent1], self.ent2ids[ent2]
        n1, n2 = len(sents1), len(sents2)
        n_pairs = min(self.max_sent_pairs_per_ent_pair, max(n1, n2))
        indices1 = np.random.choice(n1, n_pairs, replace=n1 < n_pairs)
        indices2 = np.random.choice(n2, n_pairs, replace=n2 < n_pairs)
        return [(sents1[i1], sents2[i2]) for i1, i2 in zip(indices1, indices2)]

    # randomly select k entities from the semantic class except the cln
    def select_k_ents_except(
        self, cln: str, k: int,
    ) -> tuple[list[str], list[int]]:
        ents, sids = [], []
        idxs = [1, -1]
        clns = [_cln for _cln in self.cln2groups.keys() if _cln != cln]
        for cln in np.random.choice(clns, k):
            group_ids = list(self.cln2groups[cln].keys())
            group_id = group_ids[np.random.choice(len(group_ids))]
            idx = idxs[np.random.choice(2)]
            sid = self.src2sid[(cln, group_id, idx)]
            ent = random.choice(self.sid2ents[sid])
            ents.append(ent)
            sids.append(sid)
        return ents, sids

    def make_ent_pairs_from_ents(
        self, ents: list[str], pairs_per_ent: int = 2
    ) -> list[tuple[str, str]]:
        ent_pairs = []
        n = len(ents)
        for i, ent1 in enumerate(ents):
            n2 = n - i - 1
            k = min(n2, pairs_per_ent)
            for j in np.random.choice(n2, k, replace=False) + i + 1:
                ent_pairs.append((ent1, ents[j]))
        return ent_pairs

    def make_neg_ent_pairs_from_two_ents(
        self,
        ents1: list[str], ents2: list[str],
        sid1: int, sid2: int,
        pairs_per_ent: int = 1,
    ) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
        if len(ents1) < len(ents2):
            return self.make_neg_ent_pairs_from_two_ents(
                ents2, ents1, sid2, sid1, pairs_per_ent
            )
        ent_pairs = []
        for ent1 in ents1:
            for ent2 in random.choices(ents2, k=pairs_per_ent):
                ent_pairs.append((ent1, ent2))
        src_pairs = [(sid1, sid2)] * len(ent_pairs)
        return ent_pairs, src_pairs

    # select 4 types of entity pairs
    #   1. (x,x)
    #   2. (x,x'): x and y are all in 'pos_ents' or 'neg_ents'
    #   3. (x,y): x is in 'pos_ents' but y is in 'neg_ents' or x is in 'neg_ents' but y is in 'pos_ents'
    #   4. (x,z): x is in 'ents', but the class of z is not 'cln'
    def make_sent_pairs_from_group(
        self, cln: str, group_id: int,
    ) -> tuple[list[tuple[list[int], list[int]]], list[tuple[int, int]]]:
        group = self.cln2groups[cln][group_id]
        pos_ents = group["pos_ents"]
        neg_ents = group["neg_ents"]
        pos_seeds = group["pos_seeds"]
        neg_seeds = group["neg_seeds"]
        seeds_text = ""
        for ent in pos_seeds:
            seeds_text=seeds_text+f"[POS_SEP] {ent} "
        for ent in neg_seeds:
            seeds_text=seeds_text+f"[NEG_SEP] {ent} "
        seeds_text = self.tokenizer.encode(
            seeds_text, add_special_tokens=False, truncation=True, max_length=150
        )
        
        ents = pos_ents + neg_ents
        pos_src, neg_src = (cln, group_id, 1), (cln, group_id, -1)
        pos_sid, neg_sid = self.src2sid[pos_src], self.src2sid[neg_src]
        ent_pairs, src_pairs = [], []
        ent_pair_types = []

        # 1 (x,x)
        pos_ent_pairs1 = [(ent, ent) for ent in pos_ents]
        neg_ent_pairs1 = [(ent, ent) for ent in neg_ents]
        ent_pairs1 = pos_ent_pairs1 + neg_ent_pairs1
        src_pairs1 = [(pos_sid, pos_sid)] * len(pos_ent_pairs1) + \
                     [(neg_sid, neg_sid)] * len(neg_ent_pairs1)
        ent_pairs.extend(ent_pairs1)
        src_pairs.extend(src_pairs1)
        ent_pair_types.extend([1]*len(ent_pairs1))

        # 2 (x,x')
        pos_ent_pairs2 = self.make_ent_pairs_from_ents(
            pos_ents, pairs_per_ent=2)
        neg_ent_pairs2 = self.make_ent_pairs_from_ents(
            neg_ents, pairs_per_ent=2)
        ent_pairs2 = pos_ent_pairs2 + neg_ent_pairs2
        src_pairs2 = [(pos_sid, pos_sid)] * len(pos_ent_pairs2) + \
                    [(neg_sid, neg_sid)] * len(neg_ent_pairs2)
        ent_pairs.extend(ent_pairs2)
        src_pairs.extend(src_pairs2)
        ent_pair_types.extend([2]*len(ent_pairs2))

        # 3 (x,y)
        ent_pairs3, src_pairs3 = self.make_neg_ent_pairs_from_two_ents(
            pos_ents, neg_ents, pos_sid, neg_sid, pairs_per_ent=2
        )
        ent_pairs.extend(ent_pairs3)
        src_pairs.extend(src_pairs3)
        ent_pair_types.extend([3]*len(ent_pairs3))

        # 4 (x,z)
        ent_pairs4, src_pairs4 = [], []
        sids1 = [pos_sid]*len(pos_ents)+[neg_sid]*len(neg_ents)
        for ent1, sid1 in zip(ents, sids1):
            for ent2, sid2 in zip(*self.select_k_ents_except(cln, k=2)):
                ent_pairs4.append((ent1, ent2))
                src_pairs4.append((sid1, sid2))
        ent_pairs.extend(ent_pairs4)
        src_pairs.extend(src_pairs4)
        ent_pair_types.extend([4]*len(ent_pairs4))

        total_sent_pairs = []
        total_src_pairs = []
        for (ent1, ent2), src_pair, ent_pair_type in zip(ent_pairs, src_pairs, ent_pair_types):
            sent_pairs = self.make_sentence_pairs(ent1, ent2)
            _src_pairs = [src_pair] * len(sent_pairs)
            if ent_pair_type <= 3:
                for i, (sent1, sent2) in enumerate(sent_pairs):
                    sent1 = sent1 + seeds_text
                    sent2 = sent2 + seeds_text
                    sent_pairs[i] = (sent1, sent2)
            elif ent_pair_type == 4:
                for i, (sent1, sent2) in enumerate(sent_pairs):
                    sent1 = sent1 + seeds_text
                    sid1, sid2 = src_pair
                    src2 = self.sid2src[sid2]
                    cln2, group_id2, _ = src2
                    group2 = self.cln2groups[cln2][group_id2]
                    pos_seeds2 = group2["pos_seeds"]
                    neg_seeds2 = group2["neg_seeds"]
                    seeds_text2 = ""
                    for ent in pos_seeds2:
                        seeds_text2=seeds_text2+f"[POS_SEP] {ent} "
                    for ent in neg_seeds2:
                        seeds_text2=seeds_text2+f"[NEG_SEP] {ent} "
                    seeds_text2 = self.tokenizer.encode(
                        seeds_text2, add_special_tokens=False, truncation=True, max_length=150
                    )
                    sent2 = sent2 + seeds_text2
                    sent_pairs[i] = (sent1, sent2)

            total_sent_pairs.extend(sent_pairs)
            total_src_pairs.extend(_src_pairs)

        return total_sent_pairs, total_src_pairs

    def make(
        self,
    ) -> tuple[list[tuple[list[int], list[int]]], list[int]]:
        total_sent_pairs = []
        total_src_pairs = []
        for cln, groups in self.cln2groups.items():
            for group_id, group in groups.items():

                sent_pairs, src_pairs = self.make_sent_pairs_from_group(
                    cln=cln, group_id=group_id
                )
                total_sent_pairs.extend(sent_pairs)
                total_src_pairs.extend(src_pairs)
        return total_sent_pairs, total_src_pairs

#  src_pair means that two srcs are from the same query
def is_src_pair(sid1: int | Tensor, sid2: int | Tensor) -> bool | Tensor:
    return sid1 + sid2 == 0


class CLDataset(Dataset):
    def __init__(self, ent2ids, cln2groups, tokenizer, max_sent_pairs_per_ent_pair=50):
        sentence_pair_maker = SentencePairsMaker(
            cln2groups=cln2groups,
            ent2ids=ent2ids,
            tokenizer=tokenizer,
            max_sent_pairs_per_ent_pair=max_sent_pairs_per_ent_pair,
        )
        sent_pairs, src_pairs = sentence_pair_maker.make()

        self.sents1, self.sents2 = zip(*sent_pairs)
        self.srcs1, self.srcs2 = zip(*src_pairs)

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, i):
        return self.sents1[i], self.sents2[i], self.srcs1[i], self.srcs2[i]


def collate_for_cl(batch):
    sents1, sents2, srcs1, srcs2 = zip(*batch)
    max_length1 = max(len(sent) for sent in sents1)
    max_length2 = max(len(sent) for sent in sents2)
    sents1 = pad_to_length(sents1, length=max_length1, return_tensors=True)
    sents2 = pad_to_length(sents2, length=max_length2, return_tensors=True)
    srcs1 = torch.tensor(srcs1, dtype=torch.int64)
    srcs2 = torch.tensor(srcs2, dtype=torch.int64)
    return sents1, sents2, srcs1, srcs2


def get_masks(srcs1, srcs2) -> tuple[Tensor, Tensor]:
    N = srcs1.shape[0]
    # get pos_mask
    pos_mask = torch.zeros((N, 2 * N), dtype=torch.bool)
    pos_mask[:, N:] = torch.eye(N, dtype=torch.bool)

    # get neg_mask
    mask = pos_mask.clone()
    mask[:, :N] = torch.eye(N, dtype=torch.bool)
    mask |= torch.hstack([srcs1, srcs2]).unsqueeze(0).repeat(
        [N, 1]
    ) == srcs1.unsqueeze(1)
    neg_mask = ~mask

    return pos_mask, neg_mask


def get_weight(srcs1, srcs2, weight_value=1.0):
    N = len(srcs1)
    srcs = torch.hstack([srcs1, srcs2])
    weight_matrix = torch.ones(size=(N, 2 * N), dtype=torch.float32)
    indices = torch.where(is_src_pair(srcs1.unsqueeze(1), srcs.unsqueeze(0)))
    weight_matrix[indices] = weight_value
    return weight_matrix

class InfoNCELoss(nn.Module):
    def __init__(self, temperature, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    @staticmethod
    def get_neg_masked_value(sim_matrix, neg_mask) -> Tensor:
        total_neg_sims = sim_matrix.masked_select(neg_mask).exp()
        neg_sims_by_row = torch.split(
            total_neg_sims, neg_mask.sum(-1).tolist())
        neg_sim_sum = torch.tensor(
            [neg_sims.sum() for neg_sims in neg_sims_by_row], device=sim_matrix.device
        )
        return neg_sim_sum

    def forward(self, embs1, embs2, srcs1, srcs2):
        is_positive = srcs1 == srcs2
        # _is_src_pair = is_src_pair(srcs1, srcs2)
        n_pos = is_positive.sum().item()
        n_pos_srcs = len(srcs1[is_positive].unique())
        if (n_pos == 0) or (n_pos_srcs == 1):  # no positive sent_pairs
            return torch.tensor(0.0, device=embs1.device, requires_grad=True)

        N, E = embs1.shape
        device = embs1.device
        embs = torch.cat([embs1, embs2], dim=0)
        sim_matrix = embs1 @ embs.T / self.temperature  # (N,2N)
        weight_matrix = get_weight(srcs1, srcs2).to(device)
        sim_matrix = sim_matrix * weight_matrix

        pos_mask, neg_mask = get_masks(srcs1, srcs2)
        pos_mask, neg_mask = pos_mask.to(device), neg_mask.to(device)

        pos = sim_matrix[pos_mask].exp()[is_positive]  # (n_pos)
        neg = self.get_neg_masked_value(sim_matrix, neg_mask)[
            is_positive]  # (n_pos)

        loss = -torch.log(pos / (pos + neg + self.eps))  # (N)
        return loss.mean()


if __name__ == "__main__":
    ent2ids = read_pkl("./data/ent2ids.pkl")
    cln2groups = read_json("./data/cln2groups.json")
    max_sent_pairs_per_ent_pair = 50

    dataset = CLDataset(
        ent2ids=ent2ids,
        cln2groups=cln2groups,
        max_sent_pairs_per_ent_pair=max_sent_pairs_per_ent_pair,
    )
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_for_cl)
    for batch_idx, (sents1, sents2, srcs1, srcs2) in enumerate(dataloader, 1):
        if batch_idx == 1:
            print()
    print()
    print(f"There are {len(dataset)} rows.")
    print()
    print(dataset[0])
