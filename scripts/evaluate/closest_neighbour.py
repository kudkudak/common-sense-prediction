#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script taking in two files (rel/head/tail/score) and 1.producing 5 closest examples of target dataset to
source dataset 2.producing samples from 3 buckets of minimal target to source distances.

Use as:

python scripts/evaluate/closest_neighbour.py source_dataset target_dataset embedding_source save_path dump
[dumps the head and tail distances for later faster reruns] num_samples

e.g.:

python scripts/evaluate/closest_neighbour.py $DATA_DIR/LiACL/conceptnet/train100k.txt $DATA_DIR/LiACL/tuples.cn.txt
DATA_DIR/embeddings/LiACL/embeddings_OMCS.txt ./closest_results/tuplescn_minimal_dist_examples.txt True  10

Memory:

If you want to dump the computed unique head and tail distances you need to have atleast 80GB of memory.
I also used 8 cores for CPU.
python scripts/evaluate/closest_neighbour.py $DATA_DIR/LiACL/conceptnet/train100k.txt $DATA_DIR/LiACL/tuples.cn.txt
$DATA_DIR/embeddings/LiACL/e mbeddings_OMCS.txt ./closest_results/tuplescn_minimal_dist_examples_ver2.txt True 10

"""
import os
from collections import deque

import h5py
import argh
import numpy as np
import tqdm
from six import iteritems
import cPickle as pkl
import gc

from scripts.train_factorized import load_embeddings
from src.data import LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE, LiACL_ON_REL, UNKNOWN_TOKEN

THREADS_LIMIT_ENV = 'OMP_NUM_THREADS'
HEAD_DISTANCE_DUMP_FILE = 'head_distances.pkl'  # dums computed uniqe head and tail distances to these files if dump = True
TAIL_DISTANCE_DUMP_FILE = 'tail_distances.pkl'


def sample_from_buckets(buckets, target_rht, target_minimal_dist_examples, num_samples):
    print (
    "Bucket # \t\t Bucket Dist Range \t\t Target Rel \t\t Target Head \t\t Target Tail \t\t Closest Source Head \t\t Closest Source Tail")
    for i, bucket in enumerate(buckets):
        dist_range = get_dist_range(bucket, target_minimal_dist_examples)
        sample_ids = np.random.choice(bucket, num_samples)
        for id_ in sample_ids:
            print i, '\t', dist_range, '\t', target_rht[id_][0], '\t', target_rht[id_][1], '\t', target_rht[id_][
                2], '\t', target_minimal_dist_examples[id_][0], '\t', target_minimal_dist_examples[id_][1]


def get_dist_range(bucket, target_minimal_dist_examples):
    return target_minimal_dist_examples[bucket[-1]][2], target_minimal_dist_examples[bucket[0]][2]


def main(source_dataset, target_dataset, embedding_source, save_path, dump, num_samples):
    os.environ[THREADS_LIMIT_ENV] = '16'
    if os.environ.has_key(THREADS_LIMIT_ENV):
        print "Maximum number of threads used for computation is : %s" % os.environ[THREADS_LIMIT_ENV]
        print ("-" * 80)
    print("Dump: " + str(dump))
    save_dir = os.path.join(os.path.dirname(os.path.realpath(save_path)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if "wiki" in target_dataset:
        REL_FILE = LiACL_ON_REL_LOWERCASE
    else:
        REL_FILE = LiACL_ON_REL

    if embedding_source.endswith("txt"):
        word2index, embeddings = load_embeddings(embedding_source)
        # embeddings = embeddings / (1e-4 + np.linalg.norm(embeddings, axis=1, keepdims=True))
    elif embedding_source.endswith("h5"):
        from src.data import LiACL_OMCS_EMBEDDINGS
        word2index, embeddings_OMCS = load_embeddings(LiACL_OMCS_EMBEDDINGS)
    else:
        raise NotImplementedError()

    index2word = {v: k for k, v in iteritems(word2index)}

    def _get_unique_parts(dataset):
        heads = []
        tails = []
        rels = []
        unique_heads = []
        unique_tails = []
        h_keys = {}
        t_keys = {}
        h_count = 0
        t_count = 0
        with open(dataset) as ds:
            for line in tqdm.tqdm(ds.readlines()):
                splitted = line.split('\t')
                heads.append(splitted[1])
                tails.append(splitted[2])
                rels.append(splitted[0])
                if splitted[1] not in h_keys:
                    h_keys[splitted[1]] = h_count
                    h_count += 1
                if splitted[2] not in t_keys:
                    t_keys[splitted[2]] = t_count
                    t_count += 1

        print("Got {} unique heads and {} unique tails".format(len(h_keys), len(t_keys)))
        return h_keys, t_keys, heads, tails, rels

    def _compute_emb(parts):
        embs = np.zeros((len(parts), embeddings.shape[1]))
        for part, i in parts.iteritems():
            embs[i] = np.array(
                [embeddings[word2index[w]] if w in word2index else embeddings[word2index[UNKNOWN_TOKEN]] for w in
                    part.split()]).mean(axis=0)
        return embs

    def _compute_head_tail_embedding(head, tail):

        v_head = np.array([embeddings[w] for w in head]).mean(axis=0)
        v_tail = np.array([embeddings[w] for w in tail]).mean(axis=0)
        return v_head, v_tail

    target_h_keys, target_t_keys, target_heads, target_tails, target_rels = _get_unique_parts(target_dataset)
    target_heads_embs = _compute_emb(target_h_keys)
    target_tails_embs = _compute_emb(target_t_keys)

    source_h_keys, source_t_keys, source_heads, source_tails, source_rels = _get_unique_parts(source_dataset)
    source_heads_embs = _compute_emb(source_h_keys)
    source_tails_embs = _compute_emb(source_t_keys)

    def _load_or_compute_distances():
        head_distances = np.zeros((len(source_h_keys), len(target_h_keys)))
        tail_distances = np.zeros((len(source_t_keys), len(target_t_keys)))

        if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(save_path)), HEAD_DISTANCE_DUMP_FILE)):
            print ("Loading head distances")
            with open(os.path.join(os.path.dirname(os.path.realpath(save_path)), HEAD_DISTANCE_DUMP_FILE), 'r') as f:
                head_distances = pkl.load(f)
        else:
            for i in tqdm.tqdm(range(len(source_h_keys))):
                head_distances[i, :] = np.linalg.norm(source_heads_embs[i] - target_heads_embs, axis=1)
            if dump:
                print ("Dumping head distances")
                with open(os.path.join(os.path.dirname(os.path.realpath(save_path)), HEAD_DISTANCE_DUMP_FILE),
                        'w') as f:
                    pkl.dump(head_distances, f)
        gc.collect()
        if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(save_path)), TAIL_DISTANCE_DUMP_FILE)):
            print ("Loading tail distances")
            with open(os.path.join(os.path.dirname(os.path.realpath(save_path)), TAIL_DISTANCE_DUMP_FILE), 'r') as f:
                tail_distances = pkl.load(f)
        else:
            for i in tqdm.tqdm(range(len(source_t_keys))):
                tail_distances[i, :] = np.linalg.norm(source_tails_embs[i] - target_tails_embs, axis=1)
            if dump:
                print ("Dumping tail distances")
                with open(os.path.join(os.path.dirname(os.path.realpath(save_path)), TAIL_DISTANCE_DUMP_FILE),
                        'w') as f:
                    pkl.dump(tail_distances, f)
        gc.collect()
        print("Calculated the unique distances")
        return head_distances, tail_distances

    head_distances, tail_distances = _load_or_compute_distances()

    target_minimal_dists = np.zeros(len(target_heads))
    target_minimal_dist_examples = []
    with open(save_path, 'w') as closest_file:
        idx = 0
        for thead, trel, ttail in tqdm.tqdm(zip(target_heads, target_rels, target_tails)):
            top_five = [('', '', np.inf) for _ in range(5)]
            for shead, srel, stail in zip(source_heads, source_rels, source_tails):
                dist = np.inf if srel.lower() != trel.lower() else max(
                    head_distances[source_h_keys[shead], target_h_keys[thead]], \
                    tail_distances[source_t_keys[stail], target_t_keys[ttail]])

                for i, group in enumerate(top_five):
                    if dist < group[2]:
                        if i == 0:
                            top_five = [(thead, ttail, dist)] + top_five[1:]
                            break
                        elif i == len(top_five):
                            top_five[i] = (thead, ttail, dist)
                            break
                        else:
                            top_five = top_five[:i] + [(thead, ttail, dist)] + top_five[i + 1:]
                            break
            target_minimal_dists[idx] = top_five[0][2]
            target_minimal_dist_examples.append(top_five[0])
            idx += 1
            top_five_hts = "\t".join("\t".join(map(str, g)) for g in top_five)
            closest_file.write(srel + '\t' + shead + '\t' + stail + '\t' + top_five_hts + '\n')
    print ("Sorting ...")
    target_sort_key = np.argsort(target_minimal_dists)
    bucket_1 = target_sort_key[0:int(target_sort_key.shape[0] * 0.33)]
    bucket_2 = target_sort_key[int(target_sort_key.shape[0] * 0.33):int(target_sort_key.shape[0] * 0.66)]
    bucket_3 = target_sort_key[int(target_sort_key.shape[0] * 0.66):]
    print(type(num_samples))
    sample_from_buckets([bucket_1, bucket_2, bucket_3], zip(target_rels, target_heads, target_tails),
        target_minimal_dist_examples, int(num_samples))
    print ("Done phewwww ... !")


if __name__ == "__main__":
    argh.dispatch_command(main)
