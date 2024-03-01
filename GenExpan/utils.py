import json
import pickle
import os
from typing import Sequence
import torch
import torch.nn.functional as F
import numpy as np
from EntityTrie import EntityTrie
import fcntl
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def write_json(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(obj, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pkl(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def split_k(lst: Sequence, k: int):
    # split lst into k pieces
    c, r = divmod(len(lst), k)
    beg = 0
    for i in range(k):
        end = beg + c + (i < r)
        yield lst[beg:end]
        beg = end

def load_entities(entities_path) -> tuple[list[str], dict[str, int]]:
    ents = []
    ent2eid = {}
    with open(entities_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            ent = line.strip()
            ents.append(ent)
            ent2eid[ent] = i
    return ents, ent2eid


def convert_to_tokens_ids(ents, tokenizer, eos_token=None):
    eos_token = eos_token if eos_token is not None else ""
    ents = [f"{ent}{eos_token}" for ent in ents]
    encoded_ents = tokenizer.batch_encode_plus(
        ents,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return encoded_ents


def get_entity_trie(
    ents: list[str],
    tokenizer: AutoTokenizer,
    eos_token: str = None,
) -> EntityTrie:
    print("Constructing Entity Trie...")
    encoded_ents = convert_to_tokens_ids(ents, tokenizer, eos_token)
    trie = EntityTrie(encoded_ents)
    print("Entity Trie has been constructed.")
    return trie


def clear_eos_and_leading_space(sent: str, eos_token=".\n") -> str | None:
    pos = sent.find(eos_token)
    return sent[:pos].strip() if ~pos else None


# padding and make tensor
# return the tensor and bool padding masks
def ids_lists_to_tensor(
    batch_ids: list[list[int]], pad_token_id: int = 0
) -> torch.Tensor:
    max_len = max(len(ids) for ids in batch_ids)
    padded_list = [(ids + [pad_token_id] * (max_len - len(ids))) for ids in batch_ids]
    return torch.tensor(padded_list, dtype=torch.long)


# calculate the masksed probablities
# equivalent to 1/PPL in the masksed position
# batch_ids:(B,1+L) masks:(B,1+L)  both begin with 'bos'
# masks is like '[[0,0,0,1,1],[0,1,1,1,0]]', only the positions that are 1 can be caldulated
@torch.inference_mode()
def cal_probs(
    batch_ids: torch.Tensor,
    masks: torch.Tensor,
    model: AutoModelForCausalLM,
    batch_size=128,
) -> torch.Tensor:
    device = model.device
    batch_ids = batch_ids.to(device)
    masks = masks.to(device)
    input_ids = batch_ids[:, :-1]  # (B,L)
    target_ids = batch_ids[:, 1:]  # (B,L)
    masks = masks[:, 1:]  # (B,L)
    log_probs = torch.tensor([], device=device, dtype=torch.float32)
    n = input_ids.shape[0]
    for beg in range(0, n, batch_size):
        end = min(n, beg + batch_size)
        batch_input_ids = input_ids[beg:end]
        batch_target_ids = target_ids[beg:end]
        batch_masks = masks[beg:end]
        batch_logits = (
            model(batch_input_ids).logits.to(torch.float32).transpose(1, 2)
        )  # (B,V,L)
        batch_loss = F.cross_entropy(
            batch_logits, batch_target_ids, reduction="none"
        )  # (B,L)
        batch_loss *= batch_masks
        batch_log_probs = -batch_loss.sum(dim=-1) / batch_masks.sum(dim=-1)  # (B)
        log_probs = torch.cat([log_probs, batch_log_probs], dim=0)

    return torch.exp(log_probs)

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