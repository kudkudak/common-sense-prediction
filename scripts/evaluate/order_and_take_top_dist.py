#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to order triplets and take top K from given range of distance

Usage:
order_and_take_top_dist.py source_csv K target_csv embeddings dist_min dist_max batch_size
"""
import os
import json
import sys
import numpy as np
import pandas as pd
import random
import numpy as np

from src import PROJECT_ROOT
COMPUTE_DISTANCE_PATH = os.path.join(PROJECT_ROOT, "scripts/evaluate/compute_distances_fast.py")

from collections import defaultdict

def _ec(cmd):
    print(cmd)
    assert os.system(cmd) == 0

def get_A_B(f, A, B):
    lines = open(f).read().splitlines()
    scores = [float(l.split("\t")[-1]) for l in lines]
    sel_lines = np.argsort(scores)[-B:-A]
    assert len(sel_lines) == (B - A)
    return sel_lines

if __name__ == "__main__":
    source_csv, K, target_csv, embeddings, dist_min, dist_max, batch_size = sys.argv[1:]
    K = int(K)
    batch_size = int(batch_size)
    dist_min, dist_max = float(dist_min), float(dist_max)
    lines = []

    # Idea: iteratively recompute distance using sciprt
    k = 0
    while len(lines) < K:
        print("Iteration " + str(k))

        lines_batch = get_A_B(open(target_csv), k*K, (k+1)*K)
        with open(target_csv, "w") as f:
            for l in lines_batch:
                lines_batch.write(l + "\n")

        _ec("python {} {} {} {} {} {}".format(
            COMPUTE_DISTANCE_PATH,
            source_csv,
            target_csv,
            embeddings,
            target_csv + ".dists",
            batch_size
        ))

        batch_dists = np.array([float(v) for v in open(target_csv + ".dists").read().splitlines()])

        assert len(lines_batch) == len(batch_dists) == K

        ids = np.where((batch_dists >= dist_min) & (batch_dists <= dist_max))

        print("Selected {} with min and max {} {}".format(len(ids), batch_dists[ids].min(), batch_dists[ids].max()))

        # NOTE(kudkudak): Assumes computes_distances_fast doesn't shuffle
        lines += lines_batch[ids]

        k += 1

    lines = lines[0:K]
    with open(target_csv, "w") as f:
        for l in lines:
            f.write(l + "\n")