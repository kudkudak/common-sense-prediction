#!/usr/bin/env bash
# Runs score_triplets.py on models

set -x
set -e

wiki_target=allrel.txt.shuffled_scored.txt
wiki_dataset=/data/milatmp1/jastrzes/l2lwe/data/LiACL/tuples.wiki/allrel.txt.shuffled

# Factorized models
for model in factorized/3_01_root_conceptnet_my_glove_2 factorized/3_01_prototypical_conceptnet_my factorized/3_01_root_conceptnet_my_2 factorized/3_01_prototypical_conceptnet_my_glove_3; do
    target_file=$RESULTS_DIR/$model/wiki/$wiki_target
    target_dir=$RESULTS_DIR/$model/wiki/

    if [ ! -f $target_file ]; then
        python scripts/evaluate/score_triplets.py $wiki_dataset factorized $RESULTS_DIR/$model $target_dir
    fi
done

# DNN models
for model in dnn_ce/3_01_root_conceptnet_my; do
    target_file=$RESULTS_DIR/$model/wiki/$wiki_target
    target_dir=$RESULTS_DIR/$model/wiki/
    if [ ! -f $target_file ]; then
        python scripts/evaluate/score_triplets.py $wiki_dataset dnn $RESULTS_DIR/$model $target_dir
    fi
done
