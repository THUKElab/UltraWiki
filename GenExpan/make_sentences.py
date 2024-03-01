
import json
import argparse

"""
convert ent2sents to sentences
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ent2sents", type=str, default="../data/ent2sents.json")
    parser.add_argument("--sentences", type=str, default="sentences.txt")
    args = parser.parse_args()

    with open(args.ent2sents, encoding="utf-8") as f:
        ent2sents = json.load(f)

    sents = []
    for ent, _sents in ent2sents.items():
        sents.extend(_sents)
    with open(args.sentences, "w", encoding="utf-8") as f:
        for sent in sents:
            f.write(sent+"\n")
