#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Creates LiACL/conceptnet_my and LiACL/conceptnet_my_random, based on LiACL/conceptnet

Execute in data folder, like other scripts.

Takes a while to run (like 20 minutes)

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
DESTINATION2 = "LiACL/conceptnet_my_random_100k"

SEED = 777

def _ec(cmd):
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    _ec("mkdir " + DESTINATION1)
    _ec("mkdir " + DESTINATION2)

    # Cp train and rel files
    # TODO(kudkudak): After refactoring vocab change to copying vocab
    _ec("cp {} {}".format(os.path.join(SOURCE, "*train*txt"), DESTINATION1))
    _ec("cp {} {}".format(os.path.join(SOURCE, "*rel*txt"), DESTINATION1))
    _ec("cp {} {}".format(os.path.join(SOURCE, "*train*txt"), DESTINATION2))
    _ec("cp {} {}".format(os.path.join(SOURCE, "*rel*txt"), DESTINATION2))

    # Link batch embeddings
    # Shouldn't be required
    # _ec{"ln -s {} {}".format(os.path.join(SOURCE, "*embeddings*"), DESTINATION1)}
    # _ec{"ln -s {} {}".format(os.path.join(SOURCE, "*embeddings*"), DESTINATION2)}

    # Add negative samples
    # TODO: Uncomment
    for f in ['test.txt', 'dev1.txt', 'dev2.txt']:
        N = len(open(os.path.join(SOURCE, f)).read().splitlines())
        assert N%100 == 0
        _ec("cp {} {}".format(os.path.join(SOURCE, f), os.path.join(DESTINATION1, f + ".tmp")))
        # NOTE: Assumes here 50% of dataset are positive examples
        _ec("head -n {} {} > {}".format(N/2, os.path.join(DESTINATION1, f + ".tmp"), os.path.join(DESTINATION1, f + ".tmp.2")))
        _ec("python {} {} {} {}".format(AUGMENT_SCRIPT, os.path.join(DESTINATION1, f + ".tmp.2"), K, os.path.join(DESTINATION1, f)))

    # TODO(kudkudak)
    # Add OMCS distances
    # for f in ['test.txt', 'dev1.txt', 'dev2.txt']:
    #     N = len(open(os.path.join(SOURCE, f)).read().splitlines())
    #     assert N%100 == 0
    #     _ec("python {} {} {} {}".format(AUGMENT_SCRIPT, K, os.path.join(DESTINATION1, f)))

    # Create allrel.txt file
    _ec("cat {} > {}/allrel.txt".format(" ".join([os.path.join(SOURCE, r) for r in ALL_FILES]), DESTINATION2))

    # Perform split and save
    # TODO(kudkudak): Removes scores!!
    lines = open(os.path.join(DESTINATION2, "allrel.txt")).read().splitlines()
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
    for f in ['test.txt', 'dev1.txt', 'dev2.txt']:
        N = len(open(os.path.join(DESTINATION2, f + ".tmp")).read().splitlines())
        assert N%100 == 0
        _ec("python {} {} {} {}".format(AUGMENT_SCRIPT, os.path.join(DESTINATION2, f + ".tmp"), K, os.path.join(DESTINATION2, f)))



