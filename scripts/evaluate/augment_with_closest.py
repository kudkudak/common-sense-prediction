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

    print("Loading embeddings")
    if embedding_source.endswith("txt"):
        word2index, embeddings = load_embeddings(embedding_source)
        index2word = {v: k for k, v in iteritems(word2index)}
        D = embeddings.shape[1]
        V_rel = open(LiACL_ON_REL_LOWERCASE).read().splitlines()
        # Rel embedding is random, essentionally
        # NOTE(kudkudak): Better norm determination would be nice
        We_rel = {v: np.random.uniform(-0.1, 0.1, size=D) for v in V_rel}
        for v in We_rel:
            We_rel[v] = We_rel[v] / np.linalg.norm(We_rel[v]) * np.linalg.norm(embeddings[1])
    elif embedding_source.endswith("h5"):
        # Assumes comes from saved model with embeddings layer.
        # TODO(kudkudak): Implement
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    Dt = LiACLDatasetFromFile(target_dataset, rel_file_path=LiACL_ON_REL_LOWERCASE)
    st, batches_per_epocht = Dt.data_stream(1, word2index=word2index, target='score', shuffle=False)
    Ds = LiACLDatasetFromFile(source_dataset, rel_file_path=LiACL_ON_REL)
    ss, batches_per_epochs = Ds.data_stream(1, word2index=word2index, target='score', shuffle=False)

    def _featurize_triplet(head, rel, tail):
        v_head = np.array([embeddings[w] for w in head]).mean(axis=0)
        v_rel = We_rel[V_rel[rel]].reshape(1, -1)
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
        sim = Ds_featurize.dot(batch.T)
        assert sim.ndim == 2
        sim_ids = np.argsort(sim, axis=1)
        closest_ids.append(sim_ids[:, -K:])
    closest_ids = np.concatenate(closest_ids, axis=0)

    # Save
    with open(save_path, "w") as f_target:
        for closest_id in closest_ids:
            line = []
            for id in closest_id:
                head = " ".join([index2word[w_id] for w_id in Xt[id]['head']])
                tail = " ".join([index2word[w_id] for w_id in Xt[id]['tail']])
                rel = V_rel[Xt[id]['rel'][0]]
                line.append("{}\t{}\t{}".format(rel, head, tail))
            line = ";".join(line)
            f_target.write(line + "\n")


if __name__ == "__main__":
    argh.dispatch_command(main)
