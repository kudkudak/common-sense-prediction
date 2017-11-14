#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script to fetch, unpack and split wiki tuples from ACL paper

Execute in data folder, like other scipts

Format of training data:
ReceivesAction  hockey  play on ice     3.4594316186372978

TODO(kudkudak): Create top10k version
"""

import os
import numpy as np

WIKI_HREF="http://ttic.uchicago.edu/~kgimpel/comsense_resources/tuples.wiki.tar.gz"
REL_FILES=["atlocation.txt", "capableof.txt", "causes.txt", "hasa.txt",
    "hasprerequisite.txt", "hasproperty.txt", "hassubevent.txt", "isa.txt",  "receivesaction.txt",  "usedfor.txt"]

def _exec_cmd(cmd):
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    _exec_cmd("wget {}".format(WIKI_HREF))
    _exec_cmd("gunzip tuples.wiki.tar.gz")
    _exec_cmd("tar -xvf tuples.wiki.tar")
    _exec_cmd("cat {} > tuples.wiki/allrel.txt".format(" ".join([os.path.join("tuples.wiki", r) for r in REL_FILES])))
    _exec_cmd("wc -l tuples.wiki/allrel.txt")

    f = "allrel.txt"
    rng = np.random.RandomState(777)
    lines = open("tuples.wiki/" + f).read().splitlines()
    print("Read {} lines from {}".format(len(lines), f))
    rng.shuffle(lines)
    N_dev = int(0.85 * len(lines))
    with open("tuples.wiki/" + f + ".dev.tmp", "w") as dev_f:
        for l in lines[0:N_dev]:
            dev_f.write(l + "\n")
    with open("tuples.wiki/" + f + ".test.tmp", "w") as test_f:
        for l in lines[N_dev:]:
            test_f.write(l + "\n")
    # Remove middle weird column
    _exec_cmd("cat tuples.wiki/allrel.txt.test.tmp | cut -f 1-3,5- >  tuples.wiki/allrel.txt.test")
    _exec_cmd("cat tuples.wiki/allrel.txt.dev.tmp | cut -f 1-3,5- >  tuples.wiki/allrel.txt.dev")
    _exec_cmd("mv tuples.wiki LiACL")

