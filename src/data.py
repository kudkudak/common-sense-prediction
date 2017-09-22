# input data
# input embeddings
#
# embed data
# create mask for sequences???
#
# create negative samples with random sampling
# feed to


import itertools
import os

import fuel
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import (SequentialExampleScheme,
                          ShuffledExampleScheme,
                          ConstantScheme)
from fuel.transformers import (Mapping,
                               Batch,
                               Transformer)
import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_embeddings()
        self.load_train_data()

    def load_train_data(self):
        data = pd.read_csv(os.path.join(self.data_dir, "LiACL/conceptnet/train100k.txt"), sep="\t", header=None)
        data.columns = ['rel', 'head', 'tail', 'score']
        self.max_sequence_length = max(
            data['head'].map(str.split).map(len).max(),
            data['tail'].map(str.split).map(len).max()
        )
        self.dataset = IndexableDataset(data.to_dict('list'))

    def load_embeddings(self):
        embedding_file = os.path.join(self.data_dir, 'LiACL/embeddings/embeddings.txt')
        word2index = {}
        embeddings = []
        with open(embedding_file,'r') as f:
            for index, line in enumerate(f):
                values = line.split()
                word = values[0]
                emb = [float(val) for val in values[1:]]
                word2index[word] = index
                embeddings.append(emb)

        word2index['EXXXXAR'] = index+1
        emb_len = len(embeddings[0])
        embeddings.append([0]*emb_len)

        self.word2index = word2index
        self.embeddings = np.matrix(embeddings)

    def data_stream(self):
        data_stream = DataStream(self.dataset,
                                 iteration_scheme=ShuffledExampleScheme(10)))
        data_stream = Mapping(data_stream, partial(triple2index, dictionary=dataset.word2index))
        data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(batch_size=BATCH_SIZE))
        return data_stream

