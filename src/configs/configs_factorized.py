# -*- coding: utf-8 -*-
"""
Configs for Factorized
"""

from src.utils.vegab import ConfigRegistry

config = ConfigRegistry()

# Warning: test scores are measured at final epoch, not max dev2. Look more at dev2 :)

# Root should get ~ 91.92/91.96 (max dev2/test @ final epoch)
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
    'emb_drop': 0.0, # Warning: not used in DNN_CE
    'l2': 1e-6,
    'optimizer': 'adagrad',
    'learning_rate': 0.01,
    'share_mode': 1,
    'use_headtail': True,
    'use_tailrel': True,
    'use_headrel': True,
    'embedding_file': 'embeddings/LiACL/embeddings_OMCS.txt',
	'external_embedding_file': 'embeddings/glove/glove.42B.300d.txt',
    'ext_sub_embedding_file': 'embeddings/glove/glove.42B.300d.sub.txt',
    'use_embedding': True,
    'batch_norm': True,
    "trainable_word_embeddings": True,
    'bias_trick': False,
    'copy_init': False,
    'momentum': True,
    'random_seed': 0,
})

# Previous version of model, gets around 89.6% max dev2
c = config['root']
c['share_mode'] = 4
config['root_previous_share'] = c

# Argsim, ~78% max dev2
c = config['root']
c['use_headrel'] = False
c['use_tailrel'] = False
config['argsim++'] = c

# Prototypical
c = config['root']
c['use_headtail'] = False
config['prototypical'] = c
