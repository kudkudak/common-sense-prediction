#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script taking in file (rel/head/tail/score) and producing line-aligned file (closest),
where closest is a list of ;-separated tuples describing closest neighbours

Use as:

python scripts/evaluate/augment_with_closest.py source_dataset target_dataset K embedding_source save_path

e.g.:

python scripts/evaluate/augment_with_closest.py $SCRATCH/l2lwe/data/LiACL/conceptnet/train100k.txt $SCRATCH/l2lwe/data/LiACL/tuples.wiki/top100.txt.dev 3 /data/lisa/exp/jastrzes/l2lwe/data/embeddings/LiACL/embeddings_OMCS.txt $SCRATCH/l2lwe/results/augment_with_closest/tuples_wiki_top100_dev_OMCS_closest.txt

"""

import h5py

import os

import argh
import numpy as np
import tqdm
from six import iteritems

from scripts.train_factorized import load_embeddings
# TODO(kudkudak): Fix madness with lowercase or uppercase rel.txt :P
from src.data import LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE, LiACL_ON_REL


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main(source_dataset, target_dataset, K, embedding_source, save_path):
    K = int(K)

    os.system("mkdir -p " + os.path.dirname(save_path))

    # TODO(kudkudak): Get rid of this madness
    if "wiki" in target_dataset:
        REL_FILE = LiACL_ON_REL_LOWERCASE
    else:
        REL_FILE = LiACL_ON_REL

    V_rel = open(REL_FILE).read().splitlines()

    print("Loading embeddings")
    if embedding_source.endswith("txt"):
        word2index, embeddings = load_embeddings(embedding_source)
        # TODO(kudkudak): Add as option. But normalization usually helps. Focuses more on relation
        embeddings = embeddings / (1e-4 + np.linalg.norm(embeddings, axis=1, keepdims=True))
        D = embeddings.shape[1]
        # Rel embedding is random, essentionally
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
        embeddings = embeddings / (1e-4 + np.linalg.norm(embeddings, axis=1, keepdims=True))
        embeddings2 = embeddings2 / (1e-4 + np.linalg.norm(embeddings2, axis=1, keepdims=True))

        assert len(embeddings[0]) == len(embeddings2[0])
        assert len(embeddings_OMCS) == len(embeddings)
        assert len(embeddings2) == len(V_rel)

        We_rel = {v: embeddings2[id_v] for id_v, v in enumerate(V_rel)}
    else:
        raise NotImplementedError()

    index2word = {v: k for k, v in iteritems(word2index)}

    Dt = LiACLDatasetFromFile(target_dataset, rel_file_path=REL_FILE)
    st, batches_per_epocht = Dt.data_stream(1, word2index=word2index, target='score', shuffle=False)
    Ds = LiACLDatasetFromFile(source_dataset, rel_file_path=LiACL_ON_REL)
    ss, batches_per_epochs = Ds.data_stream(1, word2index=word2index, target='score', shuffle=False)

    def _featurize_triplet(head, rel, tail):
        v_head = np.array([embeddings[w] for w in head]).mean(axis=0)
        v_rel = We_rel[V_rel[rel]].reshape(-1,)
        v_tail = np.array([embeddings[w] for w in tail]).mean(axis=0)
        return np.concatenate([v_head, v_rel, v_tail]).reshape(-1, )

    def _featurize(stream):
        X = []
        D_feat = []  # 1 vector per example in train
        # We need to collect and featurize source dataset
        print("Featurizing source dataset")
        for x, y in tqdm.tqdm(stream.get_epoch_iterator(), total=batches_per_epochs):
            assert len(x['rel']) == 1
            rel = x['rel'][0][0]
            v = _featurize_triplet(x['head'].reshape(-1, ), rel, x['tail'].reshape(-1, )).reshape(1, -1)
            X.append({k: x[k][0] for k in x}) # Remove BS dimension for simplciity
            D_feat.append(v)
        D_feat = np.concatenate(D_feat, axis=0)
        print("Featurized {} examples. Shape={}".format(len(D_feat), D_feat.shape))
        return D_feat, X

    # Assuming both are small enough
    Dt_featurize, Xt = _featurize(st)
    Ds_featurize, Xs = _featurize(ss)

    # Compute
    closest_ids = []
    for batch in tqdm.tqdm(_batch(Dt_featurize, 100), total=len(Dt_featurize) / 100):
        sim = batch.dot(Ds_featurize.T)
        assert sim.ndim == 2
        sim_ids = np.argsort(sim, axis=1)
        closest_ids.append(sim_ids[:, -K:])
    closest_ids = np.concatenate(closest_ids, axis=0)

    # Save
    # TODO(kudkudak): Code below could be refactored a bit
    with open(save_path, "w") as f_target:
        for id_pos, closest_id in enumerate(closest_ids):
            head_target, tail_target, rel_target = Xt[id_pos]['head'],  Xt[id_pos]['tail'],  Xt[id_pos]['rel']
            head_target, tail_target, rel_target = " ".join([index2word[w_id] for w_id in head_target]), \
                " ".join([index2word[w_id] for w_id in tail_target]), \
                " ".join([V_rel[w_id] for w_id in rel_target])
            line = [rel_target, head_target, tail_target]
            for id in closest_id:
                head = " ".join([index2word[w_id] for w_id in Xs[id]['head']])
                tail = " ".join([index2word[w_id] for w_id in Xs[id]['tail']])
                rel = V_rel[Xs[id]['rel'][0]]
                line.append("{}\t{}\t{}".format(rel, head, tail))
            line = ",".join(line)
            f_target.write(line + "\n")


if __name__ == "__main__":
    argh.dispatch_command(main)
