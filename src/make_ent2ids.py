import os
import re
import random
from time import time
from transformers import BertTokenizerFast
import argparse
from tqdm import tqdm
from utils import *
random.seed(42)
np.random.seed(42)

def focus_ent(masked_ids: list[int], mask_token_id: int, max_length: int):
    # make sure the entity is always in the sentence
    if len(masked_ids) <= max_length:
        return masked_ids
    mask_index = masked_ids.index(mask_token_id)

    # keep the entity position centered
    start_focus = max(0, mask_index - (max_length // 2))
    end_focus = start_focus + max_length

    # if the end exceeds the list length, then truncate it in reverse
    if end_focus > len(masked_ids):
        end_focus = len(masked_ids)
        start_focus = max(0, end_focus - max_length)

    return masked_ids[start_focus:end_focus]


def batch_tokenize(
    masked_sents: list[str], tokenizer, max_length: int
) -> list[list[int]]:
    n, batch_size = len(masked_sents), 256
    mask_token_id = tokenizer.mask_token_id
    total_ids = []
    for i in tqdm(range(0, n, batch_size)):
        j = min(n, i + batch_size)
        ids = tokenizer(masked_sents[i:j])["input_ids"]
        for j, _ids in enumerate(ids):
            if len(_ids) > max_length:
                ids[j] = focus_ent(_ids, mask_token_id, max_length)
        total_ids.extend(ids)

    return total_ids


def replace_ent_with_mask(sent, ent, beg_pos, mask_token="[MASK]", count=1):
    ent = re.escape(ent)
    pattern = re.compile(rf"\b{ent}\b", flags=re.IGNORECASE)
    if not pattern.search(sent, pos=beg_pos):
        # some entities may not begin or end with a character
        pattern = re.compile(f"{ent}", flags=re.IGNORECASE)
    return sent[:beg_pos] + \
        pattern.sub(mask_token, sent[beg_pos:], count=count)


def get_ent2ids(
    ent2sents: dict[str, list[str]], ent2etext: dict[str, str],
    tokenizer, max_ids_per_ent: int, max_length: int, unique_etext=False
) -> dict[str, list[list[int]]]:
    total_masked_sents = []
    end_pos = []
    cur_end_pos = 0
    for ent, sents in ent2sents.items():
        etext = ent2etext.get(ent, None) if ent2etext is not None else None
            
        _masked_sents = []
        sampled_sents = sample(sents, max_ids_per_ent)
        if unique_etext:
            chosen_sent_id = np.random.choice(len(sampled_sents))
        for sent_id, sent in enumerate(sampled_sents):
            sent_beg_pos = 0
            if etext is not None and  \
                (not unique_etext or unique_etext and sent_id == chosen_sent_id):
                sent = etext + " " + sent
                sent_beg_pos = len(etext) + 1
            masked_sent = replace_ent_with_mask(sent, ent, beg_pos=sent_beg_pos)
            _masked_sents.append(masked_sent)
        
        total_masked_sents.extend(_masked_sents)
        cur_end_pos += len(_masked_sents)
        end_pos.append(cur_end_pos)
    total_ids = batch_tokenize(total_masked_sents, tokenizer, max_length)
    ent2ids = {}
    pre_end_pos = 0
    for i, ent in enumerate(ent2sents):
        ent2ids[ent] = total_ids[pre_end_pos : end_pos[i]]
        pre_end_pos = end_pos[i]
    return ent2ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/")
    parser.add_argument("--ent2sents", type=str, default="ent2sents.json")
    parser.add_argument("--ent2ids", type=str, default="ent2ids.pkl")
    parser.add_argument("--ent2etext", type=str, default=None, 
                        help="specify it when using retrieval augmentation")
    # parser.add_argument("--ent2etext", type=str, default="ent2etext.json", 
    #                     help="specify it when using retrieval augmentation")
    parser.add_argument("--unique_etext", action="store_true",
                        help="specify it when you want to make sure each entity only have one etext sentence")
    parser.add_argument("--max_ids_per_ent", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=300)
    args = parser.parse_args()
    args.ent2sents = os.path.join(args.data, args.ent2sents)
    if args.ent2etext is not None:
        args.ent2etext = os.path.join(args.data, args.ent2etext)
    args.ent2ids = os.path.join(args.data, args.ent2ids)
    return args


if __name__ == "__main__":
    args = parse_args()
    stt = time()
    print("making ent2ids...")
    ent2sents = read_json(args.ent2sents)
    ent2etext = read_json(args.ent2etext) if args.ent2etext is not None else None

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    ent2ids = get_ent2ids(
        ent2sents=ent2sents,
        ent2etext=ent2etext,
        tokenizer=tokenizer,
        max_ids_per_ent=args.max_ids_per_ent,
        max_length=args.max_length,
        unique_etext=args.unique_etext,
    )

    print(f"writing ent2ids to '{args.ent2ids}'...")
    write_pkl(ent2ids, args.ent2ids)
    print("done")
    print(f"totally cost {time()-stt:.2f} seconds")
