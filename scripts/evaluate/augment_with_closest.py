#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script taking in file (rel/head/tail/score)
and producing file (rel/head/tail/score/closest), where

closest is a list of ;-separated tuples describing closest neighbours

Notes
-----
For now can use only embeddings based simple dot-product. It would be nice to add measures based on adversarial per-
turbation
"""
