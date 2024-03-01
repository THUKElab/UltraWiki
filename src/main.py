import os
import argparse
import copy
import torch.multiprocessing as mp
from expand import Expan
from tqdm import tqdm
from time import time
from utils import *


ks = [10, 20, 50, 100]
columns_of_k = [f"k={k}" for k in ks]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/")
    parser.add_argument("--ent2ids", type=str, default="ent2ids.pkl")
    parser.add_argument("--cln2groups", type=str, default="cln2groups.json")
    parser.add_argument("--entities", type=str, default="entities.txt")
    parser.add_argument("--query", type=str, default="query")
    parser.add_argument("--expand_results", type=str, default="expand_results")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=200)
    parser.add_argument("--cache_ent_embeddings", action="store_true")
    parser.add_argument("--ent_embeddings", type=str, default="ent_embeddings.pt")
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine_similarity",
        choices=["cosine_similarity", "kl_divergence"],
    )
    parser.add_argument(
        "--expand_task",
        type=str,
        default="NegESE",
        choices=["NegESE", "ESE", "InverseESE"],
        help="if 'NegESE' is chosen, both pos_seeds and neg_seeds are used; if 'ESE' is chosen, only pos_seeds are used.",
    )
    parser.add_argument("--batch_size_in_cal_embs", type=int, default=640)
    parser.add_argument("--seg_length", type=int, default=4)
    parser.add_argument("--CL", action="store_true")

    args = parser.parse_args()
    args.ent2ids = os.path.join(args.data, args.ent2ids)
    args.cln2groups = os.path.join(args.data, args.cln2groups)
    args.entities = os.path.join(args.data, args.entities)
    args.query = os.path.join(args.data, args.query)
    args.expand_results = os.path.join(args.data, args.expand_results)
    args.ent_embeddings = os.path.join(args.data, args.ent_embeddings)
    
    return args


def print_rows(title2values, columns=columns_of_k, indention=0, f=None):
    ends = ["", "\n"]
    indention = "\t" * indention
    sp1, sp2 = 10, 8
    print(f'{indention}{" "*sp1}', end="", file=f)
    for i, col in enumerate(columns, 1):
        print(f"{col:>{sp2+1}}", end=ends[i == len(columns)], file=f)
    for row_title, values in title2values.items():
        print(f"{indention}{row_title:{sp1}}", end="", file=f)
        for i, value in enumerate(values, 1):
            print(f"{value*100:>{sp2}.2f}%", end=ends[i == len(values)], file=f)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_args()
    expan = Expan(args)

    if args.CL:
        cln2groups = read_json(args.cln2groups)

    n_total_query = 0
    for filename in sorted(os.listdir(args.query)):
        queries = read_json(os.path.join(args.query, filename))
        n_total_query += len(queries)

    if not os.path.exists(args.expand_results):
        os.mkdir(args.expand_results)

    n_solved_query = 0
    for filename in sorted(os.listdir(args.query)):
        cln = filename.split(".")[0]
        _filename = f"__{cln}.txt"
        _filepath = os.path.join(args.expand_results, _filename)
        if os.path.exists(_filepath):
            continue
        group_cnt = {}
        queries = read_json(os.path.join(args.query, filename))
        queries = sorted(queries, key=lambda query:query["group_id"])
        n_query = len(queries)

        sum_pos_APks = [0 for _ in range(len(ks))]
        sum_neg_APks = [0 for _ in range(len(ks))]
        sum_pos_Pks = [0 for _ in range(len(ks))]
        sum_neg_Pks = [0 for _ in range(len(ks))]

        print(f"expand {cln}: ")
        
        for query_id, query in enumerate(queries, 1):
            group_id = query["group_id"]
            if args.CL:
                pos_group_seeds = cln2groups[cln][str(group_id)]["pos_seeds"]
                neg_group_seeds = cln2groups[cln][str(group_id)]["neg_seeds"]
            else:
                pos_group_seeds = None
                neg_group_seeds = None

            if group_id not in group_cnt:
                group_cnt[group_id] = 0
            pos_gt = query["pos_gt"]
            neg_gt = query["neg_gt"]
            pos_seeds = query["pos_seeds"]
            neg_seeds = query["neg_seeds"]
            if args.expand_task == "ESE":
                # neg_seeds = None 
                pass
            elif args.expand_task == "InverseESE":
                pos_seeds = copy.deepcopy(neg_seeds)
                # neg_seeds = None
                pos_gt, neg_gt = neg_gt, pos_gt
            expanded = expan.expand(pos_seeds, neg_seeds, target_size=args.target_size, 
                                    pos_group_seeds=pos_group_seeds,
                                    neg_group_seeds=neg_group_seeds)
            n_solved_query += 1

            pos_APks = [cal_apk(pos_gt, expanded, k) for k in ks]
            neg_APks = [cal_apk(neg_gt, expanded, k) for k in ks]
            pos_Pks = [cal_pk(pos_gt, expanded, k) for k in ks]
            neg_Pks = [cal_pk(neg_gt, expanded, k) for k in ks]
            for i in range(len(ks)):
                sum_pos_APks[i] += pos_APks[i]
                sum_neg_APks[i] += neg_APks[i]
                sum_pos_Pks[i] += pos_Pks[i]
                sum_neg_Pks[i] += neg_Pks[i]

            print(
                f"\t{query_id}th query of {cln} - {query_id/n_query*100:.2f}% - {n_solved_query/n_total_query*100:.2f}%"
            )
            row_title2APks = {
                "pos_APk": pos_APks,
                "neg_APk": neg_APks,
                "pos_Pk": pos_Pks,
                "neg_Pk": neg_Pks,
            }
            print_rows(row_title2APks, columns=columns_of_k, indention=2)

            _filename = f"{group_id}_{group_cnt[group_id]}_{cln}.txt"
            group_cnt[group_id] += 1
            filepath = os.path.join(args.expand_results, _filename)
            with open(filepath, "w", encoding="utf-8") as f:
                print(f"group_id-{group_id}, query-{group_cnt[group_id]}:", file=f)
                print(file=f)
                print(f"pos_seeds: {pos_seeds}", file=f)
                print(f"neg_seeds: {neg_seeds}", file=f)
                print(file=f)
                print_rows(row_title2APks, indention=0, f=f)
                print(file=f)
                print("expanded entities:", file=f)
                for ent in expanded:
                    print(ent, file=f)

        pos_MAPs = [sum_pos_APk / n_query for sum_pos_APk in sum_pos_APks]
        neg_MAPs = [sum_neg_APk / n_query for sum_neg_APk in sum_neg_APks]
        pos_MPs = [sum_pos_Pk / n_query for sum_pos_Pk in sum_pos_Pks]
        neg_MPs = [sum_neg_Pk / n_query for sum_neg_Pk in sum_neg_Pks]

        _filename = f"__{cln}.txt"
        _filepath = os.path.join(args.expand_results, _filename)
        with open(_filepath, "w", encoding="utf-8") as f:
            print("--- total ---", file=f)
            row_title2MAPks = {
                "pos_MAPk": pos_MAPs,
                "neg_MAPk": neg_MAPs,
                "pos_MPk": pos_MPs,
                "neg_MPk": neg_MPs,
            }
            print_rows(row_title2MAPks, columns=columns_of_k, indention=0, f=f)
            print(file=f)
