#!/usr/bin/env bash

set -x
set -e

exp_path=$RESULTS_DIR/experiments/glove_vs_OMCS/confident
data_dir=$DATA_DIR/LiACL/conceptnet_my
epochs=100 # Needed for glove sometimes

for l2_a in 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2; do
    for lr in 1e-3 1e-2 1e-1; do
        name=l2a=${l2_a}_lr=${lr}
        save_path=${exp_path}/factorized/${name}
        mkdir -p $save_path
        if [ ! -f $save_path/eval_results.json ]; then
            if [ -f $save_path/model.h5 ]; then
                rm $save_path/*
            fi
            python scripts/train_factorized.py root $save_path --data_dir=$data_dir --learning_rate=$lr --epochs=$epochs --l2_a=${l2_a}
        fi
    done
done

# --embedding_file=embeddings/LiACL/embeddings_glove200_norm.txt