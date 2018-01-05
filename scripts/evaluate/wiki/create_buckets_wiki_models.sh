#!/usr/bin/env bash
# For each model creates 3 buckets (order_and_take_top_dist.py)

set -x
set -e

# Buckets for wiki, approximated using compute_distances_fast.py (it prints them out when running)
# Adapt them if training dataset, embeddings, or compute_distances_fast code changes
t1=3.22
t2=4.31

source_csv_template=wiki/allrel.txt.shuffled_scored.txt
distance_csv=$DATA_DIR/ACL/conceptnet/train100k.txt
batch_size=25
bucket_size=100
embeddings_file=$DATA_DIR/embeddings/LiACL/embeddings_OMCS.txt

for modelpath in factorized/3_01_prototypical_conceptnet_my factorized/3_01_root_conceptnet_my_2 factorized/3_01_prototypical_conceptnet_my_glove_3 dnn_ce/3_01_root_conceptnet_my dnn_ce/3_01_root_conceptnet_my; do
    source_csv=$RESULTS_DIR/${modelpath}/${source_csv_template}
#
#    # Middle bucket
#    target_file=${source_csv}_b1.txt
#    if [ ! -f $target_file ]; then
#        python scripts/evaluate/order_and_take_top_dist.py ${source_csv} $bucket_size $target_file $distance_csv $embeddings_file $t1 $t2 $batch_size
#    fi
#
#    # Top bucket
#    target_file=${source_csv}_b2.txt
#    if [ ! -f $target_file ]; then
#        python scripts/evaluate/order_and_take_top_dist.py $source_csv $bucket_size $target_file $distance_csv $embeddings_file $t2 100000000 $batch_size
#    fi

    # Bottom bucket
    target_file=${source_csv}_b0.txt
    if [ ! -f $target_file ]; then
        python scripts/evaluate/order_and_take_top_dist.py $source_csv $bucket_size $target_file $distance_csv  $embeddings_file -10000000 $t1 $batch_size
    fi
done