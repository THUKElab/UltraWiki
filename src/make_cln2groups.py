import os
import re
from time import time
import argparse
from tqdm import tqdm
from utils import *
import openai
from openai.error import APIError

# OpenAI API - Version: 0.27.8

raw_classfication_prompt = """I have a task that involves classifying candidate entities based on their alignment with a seed entity set. \
The seed entities are grouped together because they share certain attributes, referred to as seed attributes. \
I will provide a list of seed entities along with their seed attributes. \
Additionally, I have a list of candidate entities that are similar to the seed entities \
but may not necessarily share the same seed attributes. \
I need you to identify the seed attributes and use them to \
classify each candidate entity into one of two categories: \
1) consistent with the seed entity set in terms of seed attributes, or \
0) inconsistent with the seed entity set in terms of seed attributes. \
For the given N candidate entities, please output N values, each being 1 or 0, \
indicating whether each candidate is consistent (1) or \
inconsistent (0) with the seed entity set based on the seed attributes.

Input:
Seed entities: ['Mark Twain', 'Ernest Hemingway', 'F. Scott Fitzgerald']
Candidate entities: ['J.K. Rowling', 'Stephen King', 'Agatha Christie', 'John Steinbeck', 'Harper Lee', 'Charles Dickens', 'Virginia Woolf'], totally 7 entities
Output:
{{\"result\": [0,1,0,1,1,0,0]}}

Input:
Seed entities: ['Golden Retriever', 'German Shepherd', 'Labrador Retriever']
Candidate entities: ['Bengal Tiger', 'Beagle', 'Siberian Husky', 'African Elephant', 'Pug'], totally 5 entities
Output:
{{\"result\": [0,1,1,0,1]}}

Input:
Seed entities: {}
Candidate entities: {}, totally {} entities
Output:"""


def get_classification(
    seeds: list[str],
    ents_to_be_classified: list[str],
    model="gpt-4-1106-preview",
    max_tokens=2000,
    *args,
    **kwargs,
) -> tuple[np.ndarray, int]:
    global raw_classfication_prompt
    classfication_prompt = raw_classfication_prompt.format(
        seeds, ents_to_be_classified, len(ents_to_be_classified)
    )

    messages = [
        {"role": "system", "content": "The output must be a json object."},
        {"role": "user", "content": classfication_prompt},
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            top_p=0.01,
            seed=42,
            response_format={"type": "json_object"},
        )
    except APIError as e:
        print(e)

    finish_reason = response["choices"][0]["finish_reason"]
    if finish_reason != "stop":
        raise APIError(f"finish_reason is not 'stop' but '{finish_reason}'")
    content = response["choices"][0]["message"]["content"].strip()

    try:
        classfication = np.array(json.loads(content)["result"])
    except ValueError as ve:
        print(ve)

    usage = response["usage"]
    cost_tokens = usage["prompt_tokens"] + usage["completion_tokens"] * 2
    return classfication, cost_tokens


def cal_classification_acc(scores: np.array, n_right: int):
    n = len(scores)
    if n == 0:
        return 0.0
    n_valid = (scores[:n_right] == 1).sum()
    n_valid += (scores[n_right:] == 0).sum()
    return n_valid / n


def test_classification():
    pos_seeds = ["Tengzhou", "Huili", "Rongcheng", "Kuytun"]
    neg_seeds = ["Jiangshan", "Tongxiang", "Longgang"]
    ents_to_be_classified1 = [
        "Aksu", "Alashankou", "Altay", "Anda", "Anguo",
        "Cixi", "Dongyang", "Haining", "Hangzhou", "Huzhou",
        "Ankang", "Anqing", "Anshan", "Anshun", "Anyang",
    ]
    ents_to_be_classified2 = [
        "Cixi", "Dongyang", "Haining", "Hangzhou", "Huzhou",
        "Aksu", "Alashankou", "Altay", "Anda", "Anguo",
        "Ankang", "Anqing", "Anshan", "Anshun", "Anyang",
    ]
    classification1, cost_tokens1 = get_classification(pos_seeds, ents_to_be_classified1)
    classification2, cost_tokens2 = get_classification(neg_seeds, ents_to_be_classified2)
    cost_tokens = cost_tokens1+cost_tokens2
    print(f'classification1: {classification1}')
    print(f'classification2: {classification2}')
    acc1 = cal_classification_acc(classification1, 5)
    acc2 = cal_classification_acc(classification2, 5)
    print(f"acc1: {acc1*100:.2f}%")
    print(f"acc2: {acc2*100:.2f}%")
    print(f"cost_tokens: {cost_tokens}")


def read_seeds_from_queries(
    queries: list[dict],
) -> dict[int, tuple[list[str], list[str]]]:
    group_id2seeds: dict[int, tuple[set[str], set[str]]] = {}
    for query in queries:
        group_id = query["group_id"]
        pos_seeds = query["pos_seeds"]
        neg_seeds = query["neg_seeds"]
        if group_id not in group_id2seeds:
            group_id2seeds[group_id] = (set(), set())
        group_id2seeds[group_id][0].update(pos_seeds)
        group_id2seeds[group_id][1].update(neg_seeds)
    return {
        group_id: (list(pos_seeds), list(neg_seeds))
        for group_id, (pos_seeds, neg_seeds) in group_id2seeds.items()
    }


def read_entities_from_expand_results(
    cln: str,
    n_ents_per_expansion: int,
    expand_results_dir: str,
    group_id2seeds: dict[int, tuple[list[str], list[str]]],
) -> dict[int, list[str]]:
    pattern = re.compile(f"(\d+)_(\d+)_{cln}.txt")
    group_id2ents: dict[int, set[str]] = {}
    for filename in sorted(os.listdir(expand_results_dir)):
        match = pattern.match(filename)
        if not match:
            continue
        group_id = int(match.group(1))
        lines = read_lines(os.path.join(expand_results_dir, filename))
        for i, line in enumerate(lines):
            if line == "expanded entities:":
                i += 1
                break
        # n_ents is the number of entities in this expansion
        n_ents = len(lines) - i
        n_ents = min(n_ents // 2, n_ents_per_expansion)
        if group_id not in group_id2ents:
            group_id2ents[group_id] = set()
        group_id2ents[group_id].update(lines[i : i + n_ents])

    # remove seeds from ents
    for group_id, ents in group_id2ents.items():
        seeds = group_id2seeds[group_id]
        if seeds is not None:
            pos_seeds, neg_seeds = seeds
            ents.difference_update(pos_seeds + neg_seeds)

    return {group_id: list(ents) for group_id, ents in group_id2ents.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/")
    parser.add_argument("--query", type=str, default="query")
    parser.add_argument("--expand_results_pos", type=str, default="expand_results_cl1_pos")
    parser.add_argument("--expand_results_neg", type=str, default="expand_results_cl1_neg")
    parser.add_argument("--cln2groups", type=str, default="cln2groups.json")
    parser.add_argument("--n_ents_per_expansion_pos", type=int, default=10)
    parser.add_argument("--n_ents_per_expansion_neg", type=int, default=10)

    args = parser.parse_args()
    args.query = os.path.join(args.data, args.query)
    args.expand_results_pos = os.path.join(args.data, args.expand_results_pos)
    args.expand_results_neg = os.path.join(args.data, args.expand_results_neg)
    args.cln2groups = os.path.join(args.data, args.cln2groups)
    return args


if __name__ == "__main__":
    # test_classification()
    args = parse_args()
    print("making cln2groups...")
    total_tokens = 0
    cln2groups: dict[str, dict[int, dict[str, list[str]]]] = {}

    if os.path.exists(args.cln2groups):
        cln2groups = read_json(args.cln2groups)

    filenames = sorted(os.listdir(args.query))
    for fid, filename in enumerate(filenames, 1):
        cln = filename.split(".")[0]
        if cln in cln2groups:
            continue
        print(f"\tprocessing '{cln}' - {fid}/{len(filenames)}")
        queries = read_json(os.path.join(args.query, f"{cln}.json"))
        seeds_by_group = read_seeds_from_queries(queries)
        expanded_pos_ents_by_group = read_entities_from_expand_results(
            cln, args.n_ents_per_expansion_pos, args.expand_results_pos, seeds_by_group
        )
        expanded_neg_ents_by_group = read_entities_from_expand_results(
            cln, args.n_ents_per_expansion_neg, args.expand_results_neg, seeds_by_group
        )

        groups: dict[int, dict[str, list[str]]] = {}
        n_groups = len(seeds_by_group)
        for idx, (group_id, (pos_seeds, neg_seeds)) in tqdm(enumerate(
            seeds_by_group.items(), 1
        )):
            if group_id not in expanded_pos_ents_by_group:
                pos_indices, neg_indices = [], []
            else:
                expanded_pos_ents = expanded_pos_ents_by_group[group_id]
                expanded_neg_ents = expanded_neg_ents_by_group[group_id]

                try:
                    classification_pos, cost_tokens_pos = get_classification(
                        seeds=pos_seeds,
                        ents_to_be_classified=expanded_pos_ents,
                    )
                    classification_neg, cost_tokens_neg = get_classification(
                        seeds=neg_seeds,
                        ents_to_be_classified=expanded_neg_ents,
                    )
                except Exception as e:
                    write_json(cln2groups, args.cln2groups)
                    print(e)
                    exit(0)
                total_tokens += cost_tokens_pos+cost_tokens_neg
                pos_indices = np.where(classification_pos == 1)[0]
                neg_indices = np.where(classification_neg == 1)[0]

            pos_ents = pos_seeds + [expanded_pos_ents[i] for i in pos_indices]
            neg_ents = neg_seeds + [expanded_neg_ents[i] for i in neg_indices]
            groups[group_id] = (pos_ents, neg_ents)
            groups[group_id] = {
                "pos_seeds": pos_seeds,
                "neg_seeds": neg_seeds,
                "pos_ents": pos_ents,
                "neg_ents": neg_ents,
            }
        cln2groups[cln] = groups

    print(f"writing cln2groups to '{args.cln2groups}'...")
    write_json(cln2groups, args.cln2groups)
    print("done")
    print(f"totally cost {total_tokens/1000:.1f}K tokens.")
