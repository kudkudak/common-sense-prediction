#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Creates LiACL/conceptnet_my and LiACL/conceptnet_my_random, based on LiACL/conceptnet

Execute in data folder, like other scripts.

Notes
-----
TODO(kudkudak): Not sure if there is any reason to keep K=1?
"""

import os
import numpy as np
from src.data import LiACL_OMCS_EMBEDDINGS

SOURCE = "LiACL/conceptnet"
TRAIN_FILE = "train100k.txt"
DEV1_FILE = "dev1.txt"
DEV2_FILE = "dev2.txt"
TEST_FILE = "test.txt"
ALL_FILES = [TRAIN_FILE, DEV1_FILE, DEV2_FILE, TEST_FILE]

AUGMENT_SCRIPT = "$PROJECT_ROOT/scripts/data/augment_with_negative_samples.py"
DISTANCE_SCRIPT = "$PROJECT_ROOT/scripts/evaluate/compute_distances.py"
K = 1

DESTINATION1 = "LiACL/conceptnet_my"
DESTINATION2 = "LiACL/conceptnet_my_random"

SEED = 777

def _ec(cmd):
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    _ec("mkdir " + DESTINATION1)
    _ec("mkdir " + DESTINATION2)

    # Add negative samples to all files
    for f in ALL_FILES:
        N = len(open(os.path.join(SOURCE, f)).read().splitlines())
        assert N%100 == 0
        _ec("python {} {} {} {}".format(AUGMENT_SCRIPT, K, os.path.join(DESTINATION1, f)))

    # Add OMCS distances
    for f in ['test.txt', 'dev1.txt', 'dev2.txt']:
        N = len(open(os.path.join(SOURCE, f)).read().splitlines())
        assert N%100 == 0
        _ec("python {} {} {} {}".format(AUGMENT_SCRIPT, K, os.path.join(DESTINATION1, f)))

    # TODO(kudkudak)
    # # Create allrel.txt file
    # _ec("cat {} > {}/allrel.txt".format(DESTINATION2, " ".join([os.path.join(SOURCE, r) for r in ALL_FILES])))

    # Perform split and save
    lines = open(os.path.join(DESTINATION2, "allrel.txt"))
    rng = np.random.RandomState(SEED)
    rng.shuffle(lines)

    d = {"test.txt.tmp": lines[0:1200],
        "dev1.txt.tmp": lines[1200:(1200+600)],
        "dev2.txt.tmp": lines[(1200+600):2400],
        "train.txt.tmp": lines[2400:]}
    for f in d:
        with open(os.path.join(DESTINATION2, f), "w") as dev_f:
            for l in d[f]:
                dev_f.write(l + "\n")

    # Add negative samples
    for f in ALL_FILES:
        N = len(open(os.path.join(DESTINATION2, f + ".tmp")).read().splitlines())
        assert N % 100 == 0
        _ec("python {} {} {} {}".format(AUGMENT_SCRIPT, K, os.path.join(DESTINATION2, f)))



