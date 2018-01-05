#!/usr/bin/env bash
# For each bucket prepares separate evaluation using human_evaluate_triplets_v2.sh

basename=7
wiki_scores_template=wiki/allrel.txt.shuffled_scored.txt
bucket_size=100

for bucket in _b0.txt _b1.txt _b2.txt; do
    models=

    # Collect all models
    for modelpath in factorized/3_01_prototypical_conceptnet_my factorized/3_01_root_conceptnet_my_2 factorized/3_01_prototypical_conceptnet_my_glove_3 dnn_ce/3_01_root_conceptnet_my dnn_ce/3_01_root_conceptnet_my; do
    models=$models,``echo $modelpath/$wiki_scores_template``$bucket
    done

    # Prepare eval
    save_path=``echo $SCRATCH/l2lwe/eval/$basename``_$bucket
    python scripts/evaluate/human_evaluate_triplets_v2.py prepare $save_path $models $bucket_size $bucket_size
done
