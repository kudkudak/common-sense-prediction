# -*- coding: utf-8 -*-
"""
Implementatons of configs used in ACL model
"""

from src.utils.vegab import ConfigRegistry

config = ConfigRegistry()

# Root should get ~ 91.9/91.96
config.set_root_config({
    'batch_size': 200,
    'epochs': 30,
    'activation': 'relu',
    'hidden_units': 1000,
    'data_dir': 'LiACL/conceptnet/',
    'merge': 'add',
    'merge_weight': False,
    'rel_init': 0.05,
    'l2': 1e-6,
    'optimizer': 'adagrad',
    'learning_rate': 0.01,
    'share_mode': 1, # 0, 1 or 2.
    'use_headtail': True,
    'use_tailrel': True,
    'use_headrel': True,
    'embedding_file': 'embeddings/LiACL/embeddings_OMCS.txt',
    'use_embedding': True,
    'batch_norm': True,
    'bias_trick': False,
    'copy_init': False, # Initializes transformation matrices to copy repr
    'momentum': True,
    'random_seed': 0,
})