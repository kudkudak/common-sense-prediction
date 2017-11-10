import os

import numpy as np


def endless_data_stream(data_stream):
    for iterator in data_stream.iterate_epochs():
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break


def load_embeddings(data_dir, embedding_file):
    embedding_file = os.path.join(data_dir, embedding_file)
    word2index = {'PADDING-WORD':0}
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