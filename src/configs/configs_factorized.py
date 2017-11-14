# -*- coding: utf-8 -*-
"""
Configs for Factorized
"""

from src.utils.vegab import ConfigRegistry

config = ConfigRegistry()

# Root should get ~ 91.9/91.96 (dev2/test)
# share_mode=0 boosts slightly results (~0.1%),
# similarly tuning regularization (not checked extensively, but around 0.5%)
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
    'share_mode': 1,
    "emb_drop": 0.0,
    "trainable_word_embeddings": True,
    'use_headtail': True,
    'use_tailrel': True,
    'use_headrel': True,
    'embedding_file': 'embeddings/LiACL/embeddings_OMCS.txt',
    'use_embedding': True,
    'batch_norm': True,
    'bias_trick': False,
    'copy_init': False,
    'momentum': True,
    'random_seed': 0,
})

# Previous version of model, gets around 89%
c = config['root']
c['share_mode'] = 4
config['root_previous_share'] = c
