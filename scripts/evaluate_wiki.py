#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate script for models using wiki examples mined by original ACL paper.

python scripts/evaluate_wiki.py type save_path

Adds eval_wiki.json to save_path folder.

Uses train, dev, and all splits
"""

from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data
import numpy as np
from keras.optimizers import (Adagrad,
                              Adam,
                              SGD,
                              RMSprop)

from src import DATA_DIR
from src.callbacks import (EvaluateOnDataStream, _evaluate_with_threshold_fitting,
                           EvaluateWithThresholdFitting,
                           SaveBestScore)
from src.configs import configs_factorized
from src.data import Dataset
from src.model import factorized
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.tools import argsim_threshold
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver

def evaluate(type, which):
    pass

if __name__ == "__main__":
    pass