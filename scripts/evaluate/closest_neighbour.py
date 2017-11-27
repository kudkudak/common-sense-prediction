"""
Small script taking in two files (rel/head/tail/score) and producing 5 closest examples of target dataset to source dataset.

Make sure the ** save directory ** is created!!
Use as:

python scripts/evaluate/closest_neighbour.py source_dataset target_dataset embedding_source save_path [dump distances=True/False dumps the head and tail distances for later faster reruns]

e.g.:

python scripts/evaluate/closest_neighbour.py $DATA_DIR/LiACL/conceptnet/train100k.txt $DATA_DIR/LiACL/tuples.wiki $DATA_DIR/embeddings/LiACL/embeddings_OMCS.txt ./closest_results/train100k_wiki.txt

"""



import os
from collections import deque

import h5py
import argh
import numpy as np
import tqdm
from six import iteritems
import pickle as pkl
from scripts.train_factorized import load_embeddings
from src.data import LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE, LiACL_ON_REL, UNKNOWN_TOKEN

def main(source_dataset, target_dataset, embedding_source, save_path, dump=False):

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
        for part,i in parts.iteritems():
            embs[i] = np.array([embeddings[word2index[w]] if w in word2index else embeddings[word2index[UNKNOWN_TOKEN]] for w in part.split()]).mean(axis=0)
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

    head_distances = np.zeros((len(source_h_keys),len(target_h_keys)))
    tail_distances = np.zeros((len(source_t_keys),len(target_t_keys)))

    for i in tqdm.tqdm(range(len(source_h_keys))):
        head_distances[i,:] = np.linalg.norm(source_heads_embs[i] - target_heads_embs, axis=1)
    for i in tqdm.tqdm(range(len(source_t_keys))):
        tail_distances[i,:] = np.linalg.norm(source_tails_embs[i] - target_tails_embs, axis=1)
    print("Calculated the differences")
    if dump:
        print ("Dumping distances")
        pkl.dump(head_distances, open(os.path.join(os.path.dirname(os.path.realpath(save_path)), 'head_distances.pkl'),'w'))
        pkl.dump(tail_distances, open(os.path.join(os.path.dirname(os.path.realpath(save_path)), 'tail_distances.pkl'),'w'))
        print("Dumped")

    with open(save_path,'w') as closest_file:
        for shead,srel,stail in tqdm.tqdm(zip(source_heads, source_rels, source_tails)):
            top_five = [('','',np.inf) for _ in range(5)]
            for thead,trel,ttail in zip(target_heads, target_rels, target_tails):
                dist = np.inf if srel.lower() != trel.lower() else max(head_distances[source_h_keys[shead],target_h_keys[thead]],\
                                                                       tail_distances[source_t_keys[stail],target_t_keys[ttail]])
                for i, group in enumerate(top_five):
                    if dist < group[2]:
                        if i == 0:
                            top_five = [(thead,ttail,dist)] + top_five[1:]
                            break
                        elif i == len(top_five):
                            top_five[i] = (thead,ttail,dist)
                            break
                        else:
                            top_five = top_five[:i] + [(thead,ttail,dist)] + top_five[i+1:]
                            break
            top_five_hts = "\t".join("\t".join(map(str,g)) for g in top_five)
            closest_file.write(srel + '\t' + shead + '\t' + stail + '\t' + top_five_hts +'\n')
    print ("Done phewwww!")

if __name__ == "__main__":
    argh.dispatch_command(main)
