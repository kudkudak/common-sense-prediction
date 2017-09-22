import itertools
import os
from functools import partial

import fuel
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import (ShuffledExampleScheme,
                          ConstantScheme)
from fuel.transformers import (Mapping,
                               Batch,
                              Transformer,
                              Padding)
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
        data = data.drop(['score'], axis=1)
        self.max_sequence_length = max(
            data['head'].map(str.split).map(len).max(),
            data['tail'].map(str.split).map(len).max()
        )
        self.dataset = IndexableDataset(data.to_dict('list'))

    def load_embeddings(self):
        embedding_file = os.path.join(self.data_dir, 'LiACL/embeddings/embeddings.txt')
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

        self.word2index = word2index
        self.embeddings = np.matrix(embeddings)

    def data_stream(self, batch_size):
        data_stream = DataStream(self.dataset,
                                iteration_scheme=ShuffledExampleScheme(10))
        data_stream = Mapping(data_stream, partial(triple2index, dictionary=self.word2index))
        data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(batch_size=batch_size))
        data_stream = NegativeSampling(data_stream, random_seed=0)
        data_stream = MaxedPadding(data_stream, max_sequence_length=self.max_sequence_length,
                                   mask_sources=('head, tail'), mask_dtype=np.int)
        return data_stream


def triple2index(triple, dictionary):
    return [[dictionary.get(word) for word in string.split()]
            for string in triple]


class NegativeSampling(Transformer):
    def __init__(self, data_stream, random_seed, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches, '
                             'not examples.')

        super(NegativeSampling, self).__init__(data_stream,
                                               produces_examples=False,
                                               **kwargs)
        np.random.seed(random_seed)

    def transform_batch(self, batch):
        rel, head, tail = batch
        batch_size = rel.size

        neg_rels_idx = np.random.randint(batch_size, size=batch_size)
        neg_head_idx = np.random.randint(batch_size, size=batch_size)
        neg_tail_idx = np.random.randint(batch_size, size=batch_size)

        neg_rel = rel[neg_rels_idx]
        neg_head = head[neg_head_idx]
        neg_tail = tail[neg_tail_idx]

        rel = np.concatenate([rel, neg_rel, rel, rel], axis=0)
        head = np.concatenate([head, head, neg_head, head], axis=0)
        tail = np.concatenate([tail, tail, tail, neg_tail], axis=0)

        return (rel, head, tail)


class MaxedPadding(Padding):
    def __init__(self, data_stream, max_sequence_length, **kwargs):
        super(MaxedPadding, self).__init__(data_stream, **kwargs)

        self.max_sequence_length = max_sequence_length

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

            shapes = [np.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = self.max_sequence_length
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = np.asarray(source_batch[0]).dtype

            padded_batch = np.zeros(
                (len(source_batch), max_sequence_length) + rest_shape,
                dtype=dtype)
            for i, sample in enumerate(source_batch):
                padded_batch[i, :len(sample)] = sample
            batch_with_masks.append(padded_batch)

            mask = np.zeros((len(source_batch), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)
