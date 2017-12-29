#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to order triplets and take top K

Usage:
order_and_take_top.py source_csv K L target_csv
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
    source_csv, K, L, target_csv = sys.argv[1:]
    K, L = int(K), int(L)
    if not L >= K:
        raise Exception("L should be larger or equal to K")
    top_lines = get_top(source_csv, K=K, L=L, seed=777)
    with open(target_csv, "w") as f:
        for l in top_lines:
            f.write(l + "\n")