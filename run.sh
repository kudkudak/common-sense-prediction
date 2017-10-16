#!/bin/bash

source ~/.bashrc

cd ~/common-sense-prediction
source env.sh
python scripts/train_factorized.py root "$@"

