#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate script for models. Run as:

python scripts/evaluate_wiki.py type save_path

Adds eval_wiki.json to save_path folder
"""

from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data