#!/usr/bin/env bash
# For each bucket prepares separate evaluation using human_evaluate_triplets_v2.sh

set -x
set -e

basename=7
wiki_scores_template=wiki/allrel.txt.shuffled_scored.txt
bucket_size=100

for bucket in _b0.txt _b1.txt _b2.txt; do
    # First model
    models=${RESULTS_DIR}/factorized/3_01_prototypical_conceptnet_my/${wiki_scores_template}$bucket

    # Collect all models
    for modelpath in factorized/3_01_root_conceptnet_my_2 dnn_ce/3_01_root_conceptnet_my; do
        models=$models,${RESULTS_DIR}/${modelpath}/${wiki_scores_template}$bucket
    done

    # Prepare eval
    save_path=${SCRATCH}/l2lwe/eval/${basename}_$bucket
    python scripts/evaluate/human_evaluate_triplets_v2.py prepare $save_path $models $bucket_size $bucket_size
done

# Extra evaluation
basename=8
for bucket in _b0.txt _b1.txt _b2.txt; do
    # First model
    models=${RESULTS_DIR}/dnn_ce/3_01_root_conceptnet_my/${wiki_scores_template}$bucket

    # Collect all models
    for modelpath in factorized/3_01_prototypical_conceptnet_my_glove_3; do
        models=$models,${RESULTS_DIR}/${modelpath}/${wiki_scores_template}$bucket
    done

    # Prepare eval
    save_path=${SCRATCH}/l2lwe/eval/${basename}_$bucket
    python scripts/evaluate/human_evaluate_triplets_v2.py prepare $save_path $models $bucket_size $bucket_size
done

