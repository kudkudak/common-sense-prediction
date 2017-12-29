#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script taking in file (rel/head/tail/score) and producing line-aligned file (closest),
where closest is a list of ;-separated tuples describing closest neighbours

Use as:

python scripts/evaluate/augment_with_closest.py source_dataset target_dataset K embedding_source save_path

e.g.:

python scripts/evaluate/augment_with_closest.py $SCRATCH/l2lwe/data/LiACL/conceptnet/train100k.txt $SCRATCH/l2lwe/data/LiACL/tuples.wiki/top100.txt.dev 3 /data/lisa/exp/jastrzes/l2lwe/data/embeddings/LiACL/embeddings_OMCS.txt $SCRATCH/l2lwe/results/augment_with_closest/tuples_wiki_top100_dev_OMCS_closest.txt

Notes
-----
TODO: Normalize or not?
30s/100 =>  ~0.2s for 1 => 320k s => 80h for whole wiki?
"""
import h5py

import os
import pandas as pd
import argh
import numpy as np
import tqdm
from functools import partial
import logging

logger = logging.getLogger(__name__)

from six import iteritems
from scripts.train_factorized import load_embeddings
# TODO(kudkudak): Fix madness with lowercase or uppercase rel.txt :P
from src.data import LiACL_ON_REL_LOWERCASE, LiACL_ON_REL
from src import DATA_DIR


def _calculate_distance(ex, S, same_rel=False):
    # Assumes that featurization is [head, rel, tail]
    D = S.shape[1] / 3
    dist1 = np.linalg.norm(ex.reshape(1, -1)[:, 0:D] - S[:, 0:D], axis=1)
    dist2 = np.linalg.norm(ex.reshape(1, -1)[:, -D:] - S[:, -D:], axis=1)
    if same_rel:
        dist3 = np.linalg.norm(ex.reshape(1, -1)[:, D:2 * D] - S[:, D:2 * D], axis=1)
        same_rel_id = (dist3 == 0).astype("int")
        return same_rel_id * (dist1 + dist2) + (1 - same_rel_id) * 1000000000
    else:
        return (dist1 + dist2)


def _calculate_distances(df, df_feat, train_feat, same_rel=False):
    if same_rel:
        raise NotImplementedError("Not implemented relation-aware distance")

    scores = []
    for id in tqdm.tqdm(range(len(df)), total=len(df)):
        scores.append(_calculate_distance(df_feat[id], train_feat, same_rel=same_rel))
    scores_min = np.array([a.min() for a in scores])
    return scores, scores_min


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def main(source_dataset, target_dataset, K, embedding_source, save_path):
    K = int(K)

    os.system("mkdir -p " + os.path.dirname(save_path))

    # TODO(kudkudak): Get rid of this madness
    if "wiki" in target_dataset:
        logger.info("Warning: relations detected as lowercase")
        REL_FILE = LiACL_ON_REL_LOWERCASE
    else:
        REL_FILE = LiACL_ON_REL

    V_rel = open(REL_FILE).read().splitlines()
    V_rel += ['random']

    print("Loading embeddings")
    if embedding_source.endswith("txt"):
        word2index, embeddings = load_embeddings(embedding_source)
        # TODO(kudkudak): Not sure if this should always be used?
        # if normalize:
        #     embeddings = embeddings / (1e-4 + np.linalg.norm(embeddings, axis=1, keepdims=True))
        D = embeddings.shape[1]
        # Rel embedding is random
        # NOTE(kudkudak): Better norm determination would be nice
        We_rel = {v: np.random.uniform(-0.1, 0.1, size=D) for v in V_rel}
        for v in We_rel:
            We_rel[v] = We_rel[v] / np.linalg.norm(We_rel[v]) * np.linalg.norm(embeddings[1])
    elif embedding_source.endswith("h5"):
        from src.data import LiACL_OMCS_EMBEDDINGS
        word2index, embeddings_OMCS = load_embeddings(LiACL_OMCS_EMBEDDINGS)

        # TODO(kudkudak) Assumes word2index is OMCS embedding, extremely ugly
        # but not work around before issue #33 is fixed
        file = h5py.File(embedding_source)

        # file['model_weights']['embedding_1']['embedding_1']['embeddings:0'].shape
        # Assumes this is whole model,
        if "model_weights" in file:
            weights = file['model_weights']
        else:
            weights = file

        embeddings = weights['embedding_1'].values()[0].values()[0][:]
        embeddings2 = weights['embedding_2'].values()[0].values()[0][:]
        # embeddings = embeddings / (1e-4 + np.linalg.norm(embeddings, axis=1, keepdims=True))
        # embeddings2 = embeddings2 / (1e-4 + np.linalg.norm(embeddings2, axis=1, keepdims=True))
        #
        assert len(embeddings[0]) == len(embeddings2[0])
        assert len(embeddings_OMCS) == len(embeddings)
        assert len(embeddings2) == len(V_rel)

        We_rel = {v: embeddings2[id_v] for id_v, v in enumerate(V_rel)}
    else:
        raise NotImplementedError()

    index2word = {v: k for k, v in iteritems(word2index)}

    def _featurize_triplet(v, We, We_rel, words):
        return np.concatenate([np.array([We[words.get(w, 0)] for w in v[1].split()]).mean(axis=0),
            We_rel.get(v[0], We_rel['random']).reshape(1, -1),
            np.array([We[words.get(w, 0)] for w in v[2].split()]).mean(axis=0)]).reshape(-1, )

    def _featurize_df(df, dim, featurizer=_featurize_triplet):
        feat = np.zeros(shape=(len(df), dim))
        for row_id, row in tqdm.tqdm(enumerate(df.values), total=len(df)):
            feat[row_id] = featurizer(row)
        return feat

    source = pd.read_csv(source_dataset, sep="\t", header=None)
    target = pd.read_csv(target_dataset, sep="\t", header=None)
    source.columns = target.columns = ['rel', 'head', 'tail', 'score']

    logger.info("Featurizing source and target")
    featurizer = partial(_featurize_triplet, We=embeddings, We_rel=We_rel, words=word2index)
    source_feat = _featurize_df(source, dim=3 * D, featurizer=featurizer)
    target_feat = _featurize_df(target, dim=3 * D, featurizer=featurizer)

    invididual_dists, dists = _calculate_distances(target, target_feat, source_feat, same_rel=False)

    # Compute
    closest_ids = []
    for id in range(len(invididual_dists)):
        example_closest_ids = np.argsort(invididual_dists[id])
        closest_ids.append(example_closest_ids[0:K])
    closest_ids = np.array(closest_ids)

    # Save
    with open(save_path, "w") as f_target:
        for id_pos, closest_id in enumerate(closest_ids):
            head_target, tail_target, rel_target = \
                target.iloc[id_pos]['head'], target.iloc[id_pos]['tail'], target.iloc[id_pos]['rel']
            line = [rel_target, head_target, tail_target]
            for id in closest_id:
                head, tail, rel = source.iloc[id]['head'], source.iloc[id]['tail'], source.iloc[id]['rel']
                line.append("{},{},{}".format(rel, head, tail))
            line = "\t".join(line)
            f_target.write(line + "\n")
    with open(save_path + ".dists", "w") as f_target:
        f_target.write("\n".join([str(v) for v in dists]))

if __name__ == "__main__":
    argh.dispatch_command(main)
