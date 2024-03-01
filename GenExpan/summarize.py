import os
import re
import numpy as np


ks = [10, 20, 50, 100]
columns_of_k = [f"MAP@{k}" for k in ks]+[f"MP@{k}" for k in ks]


def print_rows(title2values, columns, indention=0, f=None):
    ends = ["", "\n"]
    indention = "\t" * indention
    sp1, sp2 = 10, 8
    print(f'{indention}{" "*sp1}', end="", file=f)
    for i, col in enumerate(columns, 1):
        print(f"{col:>{sp2+1}}", end=ends[i == len(columns)], file=f)
    for row_title, values in title2values.items():
        print(f"{indention}{row_title:{sp1}}", end="", file=f)
        for i, value in enumerate(values, 1):
            print(f"{value*100:>{sp2}.2f}%",
                  end=ends[i == len(values)], file=f)


class MetricsSummarizer:
    def __init__(self):
        self.query_id_pattern = re.compile(r"(\d+)th query")
        self.metrics_pattern = re.compile(r"([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)")
        self.cln_pattern = re.compile(r"\d+_([a-zA-Z_]+).txt")

        self.keys = ["Pos", "Neg"]

    def parse_query_id_line(self, line) -> int:
        return int(self.query_id_pattern.search(line).group(1))

    def parse_metric_line(self, line) -> list[float]:
        metrics = self.metrics_pattern.search(line).groups()
        return list(map(float, metrics))
    
    def parse_cln(self, filename) -> str:
        cln = self.cln_pattern.search(filename).group(1)
        return cln

    # 返回的metric为3*8
    def parse_file(self, filepath) -> tuple[int, np.ndarray]:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        query_id = self.parse_query_id_line(lines[0])

        metrics = []
        n_keys = len(self.keys)
        metric_begin_line = 5

        for i in range(metric_begin_line, metric_begin_line + n_keys):
            j = i + n_keys  # AP对应的P所在的行
            value = self.parse_metric_line(
                lines[i]) + self.parse_metric_line(lines[j])
            metrics.append(value)

        metrics = np.array(metrics)
        return query_id, metrics

    def summarize(self, expand_results_dir):

        clns = list(set(self.parse_cln(filename) for filename in os.listdir(expand_results_dir)\
                     if not filename.startswith("__")))

        for cln in sorted(clns):
            n_rows = len(self.keys)
            n_cols = 8
            n_files = 0
            metrics = np.zeros(shape=(n_rows, n_cols), dtype=float)
            for filename in sorted(os.listdir(expand_results_dir)):
                if filename.startswith("__") or self.parse_cln(filename) != cln:
                    continue
                n_files += 1
                filepath = os.path.join(expand_results_dir, filename)
                query_id, cur_metrics = self.parse_file(filepath)
                metrics += cur_metrics
            metrics /= n_files

            filepath = os.path.join(expand_results_dir, f"__{cln}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                print("--- total ---", file=f)
                row_title2metrics = {
                    "Pos": metrics[0],
                    "Neg": metrics[1],
                }
                print_rows(row_title2metrics,
                           columns=columns_of_k, indention=0, f=f)
                print(file=f)


if __name__ == "__main__":

    summrizer = MetricsSummarizer()
    summrizer.summarize("../data/expand_results_GenExpan_base")
