import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Sequence

def tensor_intersection(x: Tensor, y: Tensor)->Tensor:
    return x[torch.isin(x,y)]

def tensor_merge(x: Tensor, y: Tensor)->Tensor:
    return torch.cat([x,y[~torch.isin(y,x)]])

def tensor_remove(x: Tensor, y: Tensor)->Tensor:
    return x.masked_select(~torch.isin(x,y))

def read_json(file_path: str) -> object:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: object, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def read_pkl(file_path: str) -> object:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pkl(obj: object, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_lines(file_path: str) -> list[str]:
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_lines(lines: Sequence, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def mean(lst: Sequence):
    return sum(lst) / len(lst)


def sample(lst: Sequence, k: int) -> list:
    if len(lst) > k:
        lst = np.random.choice(lst, size=k, replace=False).tolist()
    return lst


def split_k(lst: Sequence, k: int):
    # split lst into k pieces
    c, r = divmod(len(lst), k)
    beg = 0
    for i in range(k):
        end = beg + c + (i < r)
        yield lst[beg:end]
        beg = end


def load_candidate_ents(entities_file: str) -> tuple[list[str], dict[str, int]]:
    candidate_ents = read_lines(entities_file)
    ent2eid = {ent: eid for eid, ent in enumerate(candidate_ents)}
    return candidate_ents, ent2eid


def pad_to_length(
    ids_list: list[list[int]], length: int, pad_token_id: int = 0, return_tensors=False
) -> Tensor | list[list[int]]:
    # ids_list: list of shape(B,L)
    ids_list = [ids + [pad_token_id] * (length - len(ids)) for ids in ids_list]
    if return_tensors:
        return torch.tensor(ids_list, dtype=torch.int64)
    else:
        return ids_list


def torch_max(x: Tensor, dim=-1):
    return torch.max(x, dim=dim)[0]

def torch_min(x: Tensor, dim=-1):
    return torch.min(x, dim=dim)[0]

reduction_map = {
    "mean": torch.mean,
    "max": torch_max,
    "min": torch_min,
}

def cosine_similarity(emb_cands: Tensor, emb_seeds: Tensor, reduction="mean") -> Tensor:
    reduce = reduction_map[reduction]
    similarity_matrix = emb_cands @ emb_seeds.T  # (N,M)
    similarity_by_cand = reduce(similarity_matrix, dim=-1)  # (N)
    return similarity_by_cand


def kl_div(dist_cands: Tensor, dist_seeds: Tensor, reduction="mean") -> Tensor:
    reduce = reduction_map[reduction]
    N, M = dist_cands.shape[0], dist_seeds.shape[0]
    batch_size = 10240
    device = dist_cands.device
    kl_divergence_matrix = torch.zeros(
        N, M, dtype=torch.float32, device=device
    )  # (N,M)
    for i, dist_seed in enumerate(dist_seeds):
        for beg in range(0, N, batch_size):
            end = min(N, beg + batch_size)
            kl_divergence_matrix[beg:end:, i] = F.kl_div(
                dist_seed.log(), dist_cands[beg:end], reduction="none"
            ).sum(dim=1)
    kl_divergence_by_cand = reduce(kl_divergence_matrix, dim=-1)  # (N)
    # in order to unify with the cosine similarity,
    #   we apply exp(-kl_div) to to project kl_div to [0,1]
    return (-kl_divergence_by_cand).exp()

def seg_sort(scores: torch.Tensor, seg_length: int, descending=True) -> torch.Tensor:
    num_segments = (len(scores) + seg_length - 1) // seg_length
    segment_indices_list = []
    device = scores.device

    for i in range(num_segments):
        beg = i * seg_length
        end = min((i + 1) * seg_length, len(scores))
        segment_indices = torch.arange(beg, end, device=device)
        segment_scores = scores[segment_indices]
        sorted_indices_segment = segment_scores.argsort(descending=descending)
        segment_indices_list.append(segment_indices[sorted_indices_segment])

    return torch.cat(segment_indices_list)


def cal_apk(actual, predicted, k) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]
    num_hits = 0.0
    score = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1)
    return score / min(len(actual), k)


def cal_pk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    num_hits = 0.0

    for p in predicted:
        if p in actual:
            num_hits += 1.0

    return num_hits / min(len(actual), k)