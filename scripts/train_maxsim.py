#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains MaxSim

Run like: python scripts/train_maxsim.py results/MaxSim/glove --embeddings="tmp.txt"
"""

from src.utils.vegab import wrap_no_config_registry, MetaSaver


def train(save_path, embeddings=""):
    pass

if __name__ == "__main__":
    wrap_no_config_registry(train, plugins=[MetaSaver()])
