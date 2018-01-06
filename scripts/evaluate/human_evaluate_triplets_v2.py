#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A *tool* that has 4 functionalities

1. prepares regular evaluation.
2. allows for interactive scoring
3. list examples
4. processes scored triplets

To prepare normal evaluation:

    python scripts/evaluate/human_evaluate_triplets.py prepare save_path run_A_scores,run_B_scores,.. K L

, where K is # triplets to sample, L is out of what top

To list examples:

    python scripts/evaluate/human_evaluate_triplets.py list run_scores K L

, where K is # triplets to sample, L is out of what top

To evaluate run:

    python scripts/evaluate/human_evaluate_triplets.py evaluate mapping_csv scored_csv json_output

, mapping_csv and scored_csv are produced by prepare. Expects scores in score column of scored_file.
Prints results and saves to output_csv

To score run:

    python scripts/evaluate/human_evaluate_triplets.py score scored_csv

, continues scoring from last not-resolved triplet

Notes
-----
TODO: Disentangle from get top. This is stupid to entangle them
TODO: List of ignored
"""
import os
import json
import sys
import numpy as np
import pandas as pd
import random

from collections import defaultdict

def get_top(f, K, L, seed=777):
    lines = open(f).read().splitlines()
    scores = [float(l.split("\t")[-1]) for l in lines]
    topL = np.argsort(scores)[-L:]
    rng = np.random.RandomState(seed)
    assert len(topL) == L
    ids = rng.choice(L, K, replace=False)
    selK = topL[ids]
    return [lines[id] for id in selK]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Use as python scripts/evaluate/human_evaluate_triplets_v2.py prepare save_path (...) ")
        exit(1)

    # Save command used for reference
    save_path = sys.argv[2]

    if sys.argv[1] == "prepare":
        os.system("mkdir -p " + save_path)
        open(os.path.join(save_path, "cmd.txt"), "w").write(" ".join(sys.argv))

        # Eval name
        eval_name = os.path.basename(save_path)

        # Parse args
        if len(sys.argv) != 6:
            print("Use as python scripts/evaluate/human_evaluate_triplets_v2.py prepare save_path run_A_scores,run_B_scores,.. K L")
            exit(1)
        runs, K, L = sys.argv[3:]
        K, L = int(K), int(L)
        if not L >= K:
            raise Exception("L should be larger or equal to K")
        runs = runs.split(",")
        key_runs = range(len(runs)) # Key of run is its id

        # Read
        tops = [get_top(r, K=K, L=L, seed=777) for r in runs]
        assert len(set(len(top) for top in tops)) == 1

        # Save what runs were used
        with open(os.path.join(save_path, "runs.json"), "w") as f:
            json.dump(dict(zip(key_runs, runs)), f) # Key of run is its id

        # Mix and save
        all_triplets = sum(tops, [])
        paths = sum([[r]*len(t) for r, t in zip(runs, tops)], [])
        keys_triplets = sum([[key]*len(t) for key, t in zip(key_runs, tops)], [])

        ids = range(len(all_triplets))
        random.shuffle(ids)

        eval = []
        mapping = []
        for true_id in ids:
            line, path, key = all_triplets[true_id], paths[true_id], keys_triplets[true_id]
            mapping.append({"key": key, "path": path})
            rel, head, tail, score = line.split("\t")
            eval.append({"rel": rel, "head": head, "tail": tail, "score": np.nan})

        eval = pd.DataFrame(eval)
        eval = eval[['rel', 'head', 'tail', 'score']]
        eval.to_csv(os.path.join(save_path, "eval_{}.csv".format(eval_name)), index=False)
        pd.DataFrame(mapping).to_csv(os.path.join(save_path, "mapping_{}.csv".format(eval_name)), index=False)
    elif sys.argv[1] == "list":
        # Parse args
        if len(sys.argv) != 5:
            print(
            "Use as python scripts/evaluate/human_evaluate_triplets_v2.py list scores K L")
            exit(1)
        run, K, L = sys.argv[2:]
        K, L = int(K), int(L)
        if not L >= K:
            raise Exception("L should be larger or equal to K")

        # Read
        examples = get_top(run, K=K, L=L, seed=777)

        for ex in examples:
            print(ex)
    elif sys.argv[1] == "score":
        if len(sys.argv) != 3:
            print("Use as python scripts/evaluate/human_evaluate_triplets.py score scored_csv")
            exit(1)
        scored_csv = sys.argv[2]
        evaluated = pd.read_csv(scored_csv, index_col=False)

        for row_id, row in evaluated.iterrows():
            if np.isnan(row['score']):
                head = row['head']
                tail = row['tail']
                rel = row['rel']
                entry = "{},{},{}\n".format(rel, head, tail)
                score = raw_input("{}/{} {}".format(row_id, len(evaluated), entry))
                evaluated.loc[row_id, 'score'] = score
                evaluated.to_csv(scored_csv, index=False)
    elif sys.argv[1] == "evaluate":
        # Parse args
        if len(sys.argv) != 5:
            print("Use as python scripts/evaluate/human_evaluate_triplets.py evaluate mapping_csv scored_csv json_output")
            exit(1)
        mapping_csv, scored_csv, json_output = sys.argv[2:]

        if scored_csv.endswith("tsv"):
            evaluated = pd.read_csv(scored_csv, sep="\t")
        else:
            evaluated = pd.read_csv(scored_csv)

        if mapping_csv.endswith("tsv"):
            mapping = pd.read_csv(mapping_csv, sep="\t")
        else:
            mapping = pd.read_csv(mapping_csv)

        assert len(evaluated) == len(mapping)

        eval_result = {}
        scores = defaultdict(list)

        # Unmix
        key_to_name = {}
        for id in range(len(evaluated)):
            score = evaluated.iloc[id]['score']
            if "key" in mapping:
                run_key = mapping.iloc[id]['key']
            else:
                run_key = mapping.iloc[id]['entity']
            key_to_name[run_key] = mapping.iloc[id][mapping.columns[-1]]
            scores[run_key].append(score)

        eval_result = {key: np.mean(scores[key]) for key in scores}
        eval_result['key_to_name'] = key_to_name

        print(eval_result)

        with open(json_output, "w") as f:
            json.dump(eval_result, f)
    else:
        raise NotImplementedError()