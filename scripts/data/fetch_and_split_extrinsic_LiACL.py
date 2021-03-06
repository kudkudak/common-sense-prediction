#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch, unpack, create all conceptnet and wikipedia tuples

Tuples are those used by Commonsense Knowledge Base Completion (Li et al, 2016)
Creates copies of conceptnet (cn) and wikipedia (wiki) tuples to manually score (*_scored.txt)

Format of training data:
ReceivesAction  hockey  play on ice  3.4594316186372978

Execute this in data folder ($DATA_DIR) creates
LiACL/
    tuples.cn/
        tuples.cn.txt
        tuples.cn.txt.dev
        tuples.cn.txt.dev_scored.txt
        tuples.cn.txt.test
        tuples.cn.txt.test_scored.txt
        ...

    tuples.wiki/
        allrel.txt.shuffled <- evaluate/wiki
        allrel.txt.dev
        allrel.txt.dev_scored.txt
        allrel.txt.test
        allrel.txt.test_scored.txt
        top10k.txt.dev
        top10k.txt.test
        ...
"""

import os
import numpy as np

WIKI_HREF="http://ttic.uchicago.edu/~kgimpel/comsense_resources/tuples.wiki.tar.gz"
CN_HREF="http://ttic.uchicago.edu/~kgimpel/comsense_resources/tuples.cn.txt.gz"
REL_FILES=["atlocation.txt", "capableof.txt", "causes.txt", "hasa.txt",
    "hasprerequisite.txt", "hasproperty.txt", "hassubevent.txt", "isa.txt",
    "receivesaction.txt",  "usedfor.txt"]

def _ec(cmd):
    print(cmd)
    assert os.system(cmd)==0

if __name__ == "__main__":
    _ec("wget {}".format(WIKI_HREF))
    _ec("gunzip tuples.wiki.tar.gz")
    _ec("wget {}".format(CN_HREF))
    _ec("gunzip tuples.cn.txt.gz")
    _ec("tar -xvf tuples.wiki.tar")
    _ec("cat {} > tuples.wiki/allrel.txt".format(" ".join([os.path.join("tuples.wiki", r) for r in REL_FILES])))
    _ec("wc -l tuples.wiki/allrel.txt")

    f_name = "allrel.txt"
    lines = open("tuples.wiki/" + f_name).read().splitlines()
    print("Read {} lines from {}".format(len(lines), f_name))

    # Create top10k version
    scores = [float(l.split("\t")[-1]) for l in lines]
    top10k = np.argsort(scores)[-10000:]
    with open("tuples.wiki/top10k.txt", "w") as f:
        for id in top10k:
            f.write(lines[id] + "\n")

    # Create top100 version
    scores = [float(l.split("\t")[-1]) for l in lines]
    top100 = np.argsort(scores)[-100:]
    with open("tuples.wiki/top100.txt", "w") as f:
        for id in top100:
            f.write(lines[id] + "\n")

    # Split 10k and full into dev/test
    for f_name in ["tuples.wiki/allrel.txt", "tuples.wiki/top10k.txt", "tuples.wiki/top100.txt", "tuples.cn.txt"]:
        lines = open(f_name).read().splitlines()
        rng = np.random.RandomState(777)
        rng.shuffle(lines)

        N_dev = int(0.85 * len(lines))
        with open(f_name + ".dev.tmp", "w") as dev_f:
            for l in lines[0:N_dev]:
                dev_f.write(l + "\n")
        with open(f_name + ".test.tmp", "w") as test_f:
            for l in lines[N_dev:]:
                test_f.write(l + "\n")
        with open(f_name + ".shuffled.tmp", "w") as shuffled_f:
            for l in lines:
                shuffled_f.write(l + "\n")

        # Remove middle weird column
        if "tuples.wiki" in f_name:
            _ec("cat {0}.test.tmp | cut -f 1-3,5- > {0}.test".format(f_name))
            _ec("cat {0}.shuffled.tmp | cut -f 1-3,5- > {0}.shuffled".format(f_name))
            _ec("cat {0}.dev.tmp | cut -f 1-3,5- >  {0}.dev".format(f_name))
        else:
            _ec("cat {0}.test.tmp  > {0}.test".format(f_name))
            _ec("cat {0}.shuffled.tmp  > {0}.shuffled".format(f_name))
            _ec("cat {0}.dev.tmp  >  {0}.dev".format(f_name))

    _ec("mv tuples.wiki LiACL")
    _ec("mkdir -p LiACL/tuples.cn")
    _ec("mv tuples.cn.txt* LiACL/tuples.cn/")

    _ec("cp LiACL/tuples.wiki/allrel.txt.dev LiACL/tuples.wiki/allrel.txt.dev_scored.txt")
    _ec("cp LiACL/tuples.wiki/allrel.txt.test LiACL/tuples.wiki/allrel.txt.test_scored.txt")
    _ec("cp LiACL/tuples.cn/tuples.cn.txt.dev LiACL/tuples.cn/tuples.cn.txt.dev_scored.txt")
    _ec("cp LiACL/tuples.cn/tuples.cn.txt.test LiACL/tuples.cn/tuples.cn.txt.test_scored.txt")

    # Cleanup
    # TODO(mnoukhov): other files we don't use?
    _ec("rm tuples.wiki.tar")
    _ec("rm LiACL/tuples.cn/*.tmp")
    _ec("rm -r LiACL/tuples.wiki/*.tmp")

