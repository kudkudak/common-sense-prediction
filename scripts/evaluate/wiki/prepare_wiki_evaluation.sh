#!/usr/bin/env bash
# For each bucket prepares separate evaluation using human_evaluate_triplets_v2.sh

set -x
set -e

basename=7
wiki_scores_template=wiki/allrel.txt.shuffled_scored.txt
bucket_size=100

# Bucket based evaluation
#for bucket in _b0.txt _b1.txt _b2.txt; do
#    # First model
#    models=${RESULTS_DIR}/factorized/3_01_prototypical_conceptnet_my/${wiki_scores_template}$bucket
#
#    # Collect all models
#    for modelpath in factorized/3_01_root_conceptnet_my_2 dnn_ce/3_01_root_conceptnet_my; do
#        models=$models,${RESULTS_DIR}/${modelpath}/${wiki_scores_template}$bucket
#    done
#
#    # Prepare eval
#    save_path=${SCRATCH}/l2lwe/eval/${basename}_$bucket
#    python scripts/evaluate/human_evaluate_triplets_v2.py prepare $save_path $models $bucket_size $bucket_size
#done

# Re-evaluation of Li et al. models (Bilinear and DNN)
basename=9

models=${RESULTS_DIR}/factorized/3_01_prototypical_conceptnet_my/${wiki_scores_template}$bucket
models=${DATA_DIR}/LiACL/tuples.wiki/allrel.txt.shuffled,$models

# Collect rest of the models
for modelpath in factorized/3_01_root_conceptnet_my_2 dnn_ce/3_01_root_conceptnet_my; do
    models=$models,${RESULTS_DIR}/${modelpath}/${wiki_scores_template}$bucket
done

# Prepare eval
save_path=${SCRATCH}/l2lwe/eval/${basename}
python scripts/evaluate/human_evaluate_triplets_v2.py prepare $save_path $models $bucket_size $bucket_size