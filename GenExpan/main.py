import torch.multiprocessing as mp
import os
import time
from Expan import *
from utils import *
from summarize import MetricsSummarizer
import argparse
from queue import Empty as QueueEmpty
from tqdm import tqdm



ks = [10, 20, 50, 100]
columns_of_k = [f"MAP@{k}" for k in ks]+[f"MP@{k}" for k in ks]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--entities", type=str, default="entities.txt")
    parser.add_argument("--ent2etext", type=str, default=None)
    # parser.add_argument("--ent2etext", type=str, default="ent2etext.json")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--generated_clns", type=str, default="generated_clns.json")
    parser.add_argument("--query", type=str, default="query")
    parser.add_argument("--expand_results", type=str, default="expand_results")
    parser.add_argument("--model_path", type=str, default=None)
    # parser.add_argument("--model_path", type=str,default="./train_output/checkpoint-132")
    parser.add_argument("--num_ents_per_epoch", type=int, default=40)
    parser.add_argument("--topp_prob", type=float, default=0.7)
    parser.add_argument("--early_stopping_cnt", type=int, default=5)
    parser.add_argument("--no_neg_rank", action="store_true")
    parser.add_argument("--target_size", type=int, default=200)
    parser.add_argument("--seg_length", type=int, default=4)
    args = parser.parse_args()

    args.entities = os.path.join(args.data, args.entities)
    if args.ent2etext is not None:
        args.ent2etext = os.path.join(args.data, args.ent2etext)
    args.generated_clns = os.path.join(args.data, args.generated_clns)
    args.query = os.path.join(args.data, args.query)
    args.expand_results = os.path.join(args.data, args.expand_results)
    return args

def pool_init(_n_solved_query, _query_info_queue):
    global n_solved_query, query_info_queue
    n_solved_query = _n_solved_query
    query_info_queue = _query_info_queue

def tqdm_worker(n_total_query):
    pbar = tqdm(total=n_total_query)
    while True:
        with n_solved_query.get_lock():
            completed = n_solved_query.value
        pbar.update(completed - pbar.n)
        if completed == n_total_query:
            break
        time.sleep(1)
    pbar.close()


def expand_worker(rank, args):
    torch.cuda.set_device(rank)
    expan = Expan(args)

    while True:
        try:
            query_info = query_info_queue.get_nowait()
            # print(4)
            cln, query_id, query = query_info
            pos_gt = query["pos_gt"]
            neg_gt = query["neg_gt"]
            pos_seeds = query["pos_seeds"]
            neg_seeds = query["neg_seeds"]
            expanded = expan.expand(pos_seeds, neg_seeds, target_size=args.target_size)

            with n_solved_query.get_lock():
                n_solved_query.value += 1

            pos_APks = [cal_apk(pos_gt, expanded, k) for k in ks]
            neg_APks = [cal_apk(neg_gt, expanded, k) for k in ks]
            pos_Pks = [cal_pk(pos_gt, expanded, k) for k in ks]
            neg_Pks = [cal_pk(neg_gt, expanded, k) for k in ks]

            filename = f"{query_id}_{cln}.txt"
            filepath = os.path.join(args.expand_results, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                print(f"{query_id+1}th query:", file=f)
                print(file=f)
                print(f"pos_seeds: {pos_seeds}", file=f)
                print(f"neg_seeds: {neg_seeds}", file=f)
                print(file=f)
                titles = ["pos_APk", "neg_APk", "none_APk", "pos_Pk", "neg_Pk", "none_Pk"]
                metrics = [pos_APks, neg_APks, pos_Pks, neg_Pks]
                for title, metric in zip(titles, metrics):
                    print(f"{title}: ", end="", file=f)
                    for i in range(len(ks))[:-1]:
                        print(f"{metric[i]}, ", end="", file=f)
                    print(f"{metric[-1]}", file=f)
                print(file=f)
                print("expanded entities:", file=f)
                for ent in expanded:
                    print(ent, file=f)

        except QueueEmpty:
            if args.CoT:
                expan.cot_prompt_generator.save_cln_attrs_list()
            break



if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_args()

    nproc = torch.cuda.device_count()
    if not os.path.exists(args.expand_results):
        os.mkdir(args.expand_results)

    all_queries_info = []
    for filename in sorted(os.listdir(args.query)):
        cln = filename.split(".")[0]
        queries = read_json(os.path.join(args.query, filename))
        queries = sorted(queries, key=lambda query:query["group_id"])
        for query_id, query in enumerate(queries):
            all_queries_info.append((cln, query_id, query))

    n_solved_query = mp.Value("i", 0)
    n_total_query = len(all_queries_info)
    query_info_queue = mp.Queue(maxsize=n_total_query)
    for query_info in all_queries_info:
        cln, query_id, query = query_info
        if os.path.exists(os.path.join(args.expand_results, f"{query_id}_{cln}.txt")):
            continue
        query_info_queue.put(query_info)
    n_query_need_to_expand = query_info_queue.qsize()
    print(f"\nnum queries that need to expand: {n_query_need_to_expand}\n")

    results = []
    pool = mp.Pool(processes=nproc + 1, initializer=pool_init, initargs=(n_solved_query, query_info_queue))
    for rank in range(nproc):
        result = pool.apply_async(
            expand_worker,
            args=(rank, args),
        )
        results.append(result)
    pool.apply_async(tqdm_worker, args=(n_query_need_to_expand,))

    pool.close()
    pool.join()

    summrizer = MetricsSummarizer()
    summrizer.summarize(args.expand_results)
