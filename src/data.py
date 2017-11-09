# -*- coding: utf-8 -*-
"""
Implementatons of data streams
"""

import os
import logging

import fuel
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import (SequentialScheme,
                          ShuffledScheme)
from fuel.transformers import (SourcewiseTransformer,
                               Transformer,
                               AgnosticTransformer,
                               FilterSources,
                               Rename,
                               Padding)
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
UNKNOWN_TOKEN = 'UUUNKKK'
REL_FILE = 'LiACL/conceptnet/rel.txt'
TRAIN_FILE = 'LiACL/conceptnet/train100k.txt'
TEST_FILE = 'LiACL/conceptnet/test.txt'
DEV1_FILE = 'LiACL/conceptnet/dev1.txt'
DEV2_FILE = 'LiACL/conceptnet/dev2.txt'

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_rel2index()
        self.train_dataset = self.load_data(TRAIN_FILE)
        self.dev1_dataset = self.load_data(DEV1_FILE)
        self.dev2_dataset = self.load_data(DEV2_FILE)
        self.test_dataset = self.load_data(TEST_FILE)

    def load_data(self, data_path):
        data = pd.read_csv(os.path.join(self.data_dir, data_path),
                           sep="\t", header=None)
        data.columns = ['rel', 'head', 'tail', 'score']
        assert(not data.empty)
        return IndexableDataset(data.to_dict('list'))

    def load_rel2index(self):
        rel2index = {}
        rel_file = os.path.join(self.data_dir, REL_FILE)
        with open(rel_file, 'r') as f:
            for index, line in enumerate(f):
                rel2index[line.strip()] = index

        self.rel2index = rel2index
        self.rel_vocab_size = len(rel2index)

    def train_data_stream(self, batch_size, word2index, **args):
        return self.data_stream(self.train_dataset, batch_size, word2index,
                                target='negative_sampling', name='train', **args)

    def test_data_stream(self, batch_size, word2index, **args):
        return self.data_stream(self.test_dataset, batch_size, word2index,
                                target='score', name='test', **args)

    def dev1_data_stream(self, batch_size, word2index, **args):
        return self.data_stream(self.dev1_dataset, batch_size, word2index,
                                target='score', name='dev1', **args)

    def dev2_data_stream(self, batch_size, word2index, **args):
        return self.data_stream(self.dev2_dataset, batch_size, word2index,
                                target='score', name='dev2', **args)

    def data_stream(self, dataset, batch_size, word2index, target='negative_sampling', name=None, shuffle=False):
        batches_per_epoch = int(np.ceil(dataset.num_examples / float(batch_size)))
        if shuffle:
            iteration_scheme = ShuffledScheme(dataset.num_examples, batch_size)
        else:
            iteration_scheme = SequentialScheme(dataset.num_examples, batch_size)
        data_stream = DataStream(dataset, iteration_scheme=iteration_scheme)
        data_stream = NumberizeWords(data_stream, word2index, default=word2index[UNKNOWN_TOKEN],
            which_sources=('head', 'tail'))
        data_stream = NumberizeWords(data_stream, self.rel2index, which_sources=('rel'))

        if target == 'negative_sampling':
            logger.info('target for data stream ' + name + ' is negative sampling')
            data_stream = FilterSources(data_stream, sources=('head', 'tail', 'rel'))
            data_stream = NegativeSampling(data_stream)
        elif target == 'score':
            logger.info('target for data stream ' + name + ' is score')
            data_stream = Rename(data_stream, {'score': 'target'})
        else:
            raise NotImplementedError('target ', target, ' must be one of "score" or "negative_sampling"')

        data_stream = Padding(data_stream, mask_sources=('head, tail'), mask_dtype=np.float32)
        data_stream = MergeSource(data_stream, merge_sources=('head', 'tail', 'head_mask', 'tail_mask', 'rel'),
                                  merge_name='input')

        return data_stream, batches_per_epoch


class NumberizeWords(SourcewiseTransformer):
    def __init__(self, data_stream, dictionary, default=None, *args, **kwargs):
        super(NumberizeWords, self).__init__(data_stream,
                                             produces_examples=data_stream.produces_examples,
                                             *args,
                                             **kwargs)

        self.dictionary = dictionary
        self.default = default

    def transform_source_batch(self, source_batch, source_name):
        return np.array([[self.dictionary.get(word, self.default) for word in string.split()]
                         for string in source_batch])


class NegativeSampling(Transformer):
    def __init__(self, data_stream, *args, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches, '
                             'not examples.')

        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)

        super(NegativeSampling, self).__init__(data_stream,
                                               produces_examples=False,
                                               *args,
                                               **kwargs)

    @property
    def sources(self):
        return self.data_stream.sources + ('target',)

    def transform_batch(self, batch):
        rel, head, tail = batch
        batch_size = len(head)

        neg_rels_idx = np.random.randint(batch_size, size=batch_size)
        neg_head_idx = np.random.randint(batch_size, size=batch_size)
        neg_tail_idx = np.random.randint(batch_size, size=batch_size)

        neg_rel = rel[neg_rels_idx]
        neg_head = head[neg_head_idx]
        neg_tail = tail[neg_tail_idx]

        rel = np.concatenate([rel, neg_rel, rel, rel], axis=0)
        head = np.concatenate([head, head, neg_head, head], axis=0)
        tail = np.concatenate([tail, tail, tail, neg_tail], axis=0)
        target = np.array([1]*batch_size + [0]*batch_size*3)

        return (rel, head, tail, target)


class MergeSource(AgnosticTransformer):
    """ Merge selected sources into a single source

    Merged source becomes {source_name: source,...} for all former sources
    Added to start
    """
    def __init__(self, data_stream, merge_sources, merge_name, *args, **kwargs):
        super(MergeSource, self).__init__(data_stream,
                                          data_stream.produces_examples,
                                          *args,
                                          **kwargs)

        self.merge_sources = merge_sources
        self.merge_name = merge_name
        self.sources = (merge_name,) + tuple(s for s in data_stream.sources if s not in merge_sources)

    def transform_any(self, data):
        merged_data = {s: d for s,d in zip(self.data_stream.sources, data)
                       if s in self.merge_sources}
        return [merged_data] + [d for d, s in zip(data, self.data_stream.sources)
                                if s not in self.merge_sources]


