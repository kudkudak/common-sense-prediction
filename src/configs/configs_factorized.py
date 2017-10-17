# -*- coding: utf-8 -*-
"""
Implementatons of configs used in ACL model
"""

from src.utils.vegab import ConfigRegistry

config = ConfigRegistry()

config.set_root_config({
    'batch_size': 200,
    'epochs': 30,
    'activation': 'relu',
    'hidden_units': 150,
    'merge': 'add',
    'merge_weight': False,
    'rel_init': 0.02,
    'l2': 1e-6, # "cost_new = (1000*loss) +(self.LC * l2_penalty1)" from original code ;)
    # 'lambda_2': 0.0, # Matrix for relation matrix # No identity matrix in DNN CE
    'optimizer': 'adagrad',
    'learning_rate': 0.01,
    'embedding_file': 'embeddings/LiACL/embeddings_OMCS.txt',
    'batch_norm': False,
    'bias_trick': True,
})
