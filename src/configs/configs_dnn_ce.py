# -*- coding: utf-8 -*-
"""
Implementatons of configs used in ACL model
"""

from src.utils.vegab import ConfigRegistry

config = ConfigRegistry()

config.set_root_config({
    'batch_size': 600,
    'epochs': 30,
    'rel_init': 0.02,
    'rel_vec_size': 200,
    'activation': 'relu',
    'hidden_units': 1000,
    'l2': 0.00001, # "cost_new = (1000*loss) +(self.LC * l2_penalty1)" from original code ;)
    # 'lambda_2': 0.0, # Matrix for relation matrix # No identity matrix in DNN CE
    'learning_rate': 0.01,
})

