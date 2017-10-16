#!/bin/bash
source activate csp
source env.sh
python scripts/train_factorized.py root "$@"

