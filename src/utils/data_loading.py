# -*- coding: utf-8 -*-

import os
import numpy as np
from src.data import DATA_DIR

def endless_data_stream(data_stream):
    for iterator in data_stream.iterate_epochs():
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break


def load_embeddings(embedding_file):
    if not os.path.isabs(embedding_file):
        embedding_file = os.path.join(DATA_DIR, embedding_file)
    word2index = {'PADDING-WORD': 0}
    embeddings = [0]
    with open(embedding_file,'r') as f:
        for index, line in enumerate(f):
            values = line.split()
            word = values[0]
            emb = [float(val) for val in values[1:]]
            word2index[word] = index + 1
            embeddings.append(emb)

    embeddings[0] = [0]*len(embeddings[1])

    return word2index, np.matrix(embeddings)

def load_external_embeddings(data_dir, embedding_file, ext_sub_embedding_file, main_word2index):
    embedding_file = os.path.join(data_dir, embedding_file)
    ext_sub_embedding_file = os.path.join(data_dir, ext_sub_embedding_file)
    if os.path.isfile(ext_sub_embedding_file):
        embeddings = []
        with open(ext_sub_embedding_file,'r') as f:
            index = 0
            for line in f.readlines():
                values = line.split()
                word = values[0]
                emb = [float(val) for val in values[1:]]
                embeddings.append(emb)
                index += 1
        corresponding_embeddings = np.asarray(embeddings)
    else:
        word2index = {}
        embeddings = []
        with open(embedding_file,'r') as f:
            index = 0
            for line in f.readlines():
                values = line.split()
                word = values[0]
                if word in main_word2index:
                    emb = [float(val) for val in values[1:]]
                    word2index[word] = index
                    embeddings.append(emb)
                    index += 1
                elif word.capitalize() in main_word2index: #elif so no co-occurrance #TODO(arian) improve this
                    emb = [float(val) for val in values[1:]]
                    word2index[word.capitalize()] = index
                    embeddings.append(emb)
                    index += 1

        corresponding_embeddings = np.zeros((len(main_word2index),len(embeddings[0])))
        hit = 0
        miss = 0
        for word in main_word2index.keys():
            if word in word2index:
                corresponding_embeddings[main_word2index[word]] = embeddings[word2index[word]]
                hit += 1
            else:
                miss += 1
        print ("loaded external embeddings with hit="+str(hit)+" and miss="+str(miss))
        with open(ext_sub_embedding_file,'w') as f:
            for word in main_word2index.keys():
                f.write(word+" "+" ".join(corresponding_embeddings[main_word2index[word]].astype(str))+"\n")
    return corresponding_embeddings
