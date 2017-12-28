#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A *tool* that has three functionalities:

1. takes evaluation results by score_tripletws and mixes triplets from two runs and
prepares evaluation csv that can be used in like gdocs
2. takes evaluated csv and redistributes it
3. allows for peeking into triplets produced by given model

Use as:

    python scripts/evaluate/human_evaluate_triplets.py prepare save_path run_A_scores run_B_scores K

, where K is top K triplets to get. Or:

    python scripts/evaluate/human_evaluate_triplets.py process save_path run_A_scores run_B_scores K

, it expects .csv in save_path to have human evaluations in last column. Then saves it back to run_A_scores.human and
run_B_scores.human


TODO(kudkudak): Allow for mixing of any number of runs. For now always assumes 2 runs
TODO(kudkudak): Unit test this
TODO(kudkudak): Do we really need stupid naming conventions? Change to just passing names to script
"""
import os
import sys
import numpy as np
import pandas as pd

import random

def get_top(f, K):
    lines = open(f).read().splitlines()
    scores = [float(l.split("\t")[-1]) for l in lines]
    topK = np.argsort(scores)[-K:]
    return [lines[id] for id in topK]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Use as python scripts/evaluate/human_evaluate_triplets.py prepare save_path (...) ")
        exit(1)

    # Save command used for reference
    save_path = sys.argv[2]
    os.system("mkdir -p " + save_path)
    open(os.path.join(save_path, "cmd.txt"), "w").write(" ".join(sys.argv))

    # Eval name
    eval_name = os.path.basename(save_path)

    if sys.argv[1] == "prepare":
        # Parse args
        if len(sys.argv) != 6:
            print("Use as python scripts/evaluate/human_evaluate_triplets.py prepare save_path run_A_scores run_B_scores K")
            exit(1)
        A, B, K = sys.argv[3:]
        K = int(K)

        # Weird version control
        assert A.endswith("_scored.txt") and B.endswith("_scored.txt")

        # Read
        topA = get_top(A, K=K)
        topB = get_top(B, K=K)
        assert len(topA) == len(topB) == K

        # Mix and save
        mixed = topA + topB
        ids = range(len(mixed))
        random.shuffle(ids)
        mixed = [mixed[id] for id in ids]
        eval = []
        mapping = []
        for true_id, line in zip(ids, mixed):
            entity = 'A' if true_id < K else 'B'
            path = A if true_id < K else B
            mapping.append({"entity": entity, "path": path})
            rel, head, tail, score = line.split("\t")
            eval.append({"rel": rel, "head": head, "tail": tail, "score": "?"})

        eval = pd.DataFrame(eval)
        eval = eval[['rel', 'head', 'tail', 'score']]
        eval.to_csv(os.path.join(save_path, "eval_{}.csv".format(eval_name)))
        pd.DataFrame(mapping).to_csv(os.path.join(save_path, "mapping_{}.csv".format(eval_name)))

    elif sys.argv[1] == "process":
        # Parse args
        if len(sys.argv) != 6:
            print("Use as python scripts/evaluate/human_evaluate_triplets.py prepare save_path run_A_scores run_B_scores K")
            exit(1)
        A, B, K = sys.argv[3:]
        K = int(K)

        A_output = A.replace("_scored.txt", "_top{}_human_scored.txt".format(K))
        B_output = B.replace("_scored.txt", "_top{}_human_scored.txt".format(K))

        evaluated = pd.read_csv(os.path.join(save_path, "eval_{}_done.csv".format(eval_name)))
        mapping = pd.read_csv(os.path.join(save_path, "mapping_{}.csv".format(eval_name))) # Some mixup with that..

        assert len(evaluated) == len(mapping)

        scores = [[], []]

        # Save scored tuples
        with open(A_output, "w") as f_A:
            with open(B_output, "w") as f_B:
                for id in range(len(evaluated)):
                    head = evaluated.iloc[id]['head']
                    tail = evaluated.iloc[id]['tail']
                    rel = evaluated.iloc[id]['rel']
                    score = evaluated.iloc[id]['score']
                    entry = "{}\t{}\t{}\t{}\n".format(rel, head, tail, score)
                    if mapping.iloc[id]['entity'] == 'A':
                        scores[0].append(score)
                        f_A.write(entry)
                    elif mapping.iloc[id]['entity'] == 'B':
                        scores[1].append(score)
                        f_B.write(entry)
                    else:
                        raise NotImplementedError()

        print((A, np.mean(scores[0])))
        print((B, np.mean(scores[1])))
    elif sys.argv[1] == "peek":
        if len(sys.argv) != 6:
            print("Use as python scripts/evaluate/human_evaluate_triplets.py prepare save_path run_A_scores run_B_scores K")
            exit(1)
        A, B, K = sys.argv[3:]
        K = int(K)
        # Weird version control
        assert A.endswith("_scored.txt") and B.endswith("_scored.txt")

        # Read
        topA = get_top(A, K=K)
        topB = get_top(B, K=K)
        assert len(topA) == len(topB) == K

        for id in range(10):
            print(topA[id])
        for id in range(10):
            print(topB[id])
    else:
        raise NotImplementedError()