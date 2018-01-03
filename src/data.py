# -*- coding: utf-8 -*-
"""
Implementatons of data streams. We have following datasets used in project:

* LiACL/conceptnet dataset
* LiACL/tuples.wiki evaluation dataset

TODO(kudkudak): merge LiACLDatasetFromFile and LiACLSplitDataset
TODO(kudkudak): code parametrization that is indepednent of threshold, just resamples. Like in GAN KB paper
TODO(kudkudak): NegativeSampling is limited now to k=3,2,1, wierd
"""

# Implementation with swithced padding order

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

from src import DATA_DIR

logger = logging.getLogger(__name__)

# Extrinsic evaluation dataset
LiACL_CN_DATASET = os.path.join(DATA_DIR, "LiACL", "conceptnet")
LiACL_ON_REL = os.path.join(LiACL_CN_DATASET, "rel.txt")
LiACL_OMCS_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings/LiACL/embeddings_OMCS.txt")
# TODO(kudkudak): Add sth to figure out lowercase automatically. Idk.
LiACL_ON_REL_LOWERCASE = os.path.join(LiACL_CN_DATASET, "rel_lowercase.txt")
assert os.path.exists(LiACL_ON_REL), LiACL_ON_REL
assert os.path.exists(LiACL_ON_REL_LOWERCASE), LiACL_ON_REL_LOWERCASE
assert os.path.exists(LiACL_OMCS_EMBEDDINGS), LiACL_OMCS_EMBEDDINGS
TUPLES_WIKI = os.path.join(DATA_DIR, "LiACL", "tuples.wiki")

UNKNOWN_TOKEN = 'UUUNKKK'


# TODO(kudkudak): Refactor k parameter
def _liacl_data_stream(dataset, rel2index, batch_size, word2index, target='negative_sampling', name="", k=3,
        shuffle=False, neg_sample_kwargs={}):
    batches_per_epoch = int(np.ceil(dataset.num_examples / float(batch_size)))
    if shuffle:
        iteration_scheme = ShuffledScheme(dataset.num_examples, batch_size)
    else:
        iteration_scheme = SequentialScheme(dataset.num_examples, batch_size)
    data_stream = DataStream(dataset, iteration_scheme=iteration_scheme)
    data_stream = NumberizeWords(data_stream, word2index, default=word2index[UNKNOWN_TOKEN],
        which_sources=('head', 'tail'))
    data_stream = NumberizeWords(data_stream, rel2index, which_sources=('rel'))

    if target == "score":
        data_stream = Rename(data_stream, {'score': 'target'})
    else:
        data_stream = FilterSources(data_stream, sources=('head', 'tail', 'rel'))

    data_stream = Padding(data_stream, mask_sources=('head, tail'), mask_dtype=np.float32)

    if target == 'negative_sampling':
        logger.info('target for data stream ' + str(name) + ' is negative sampling')
        data_stream = NegativeSampling(data_stream, k=k)
    elif target == 'filtered_negative_sampling':
        logger.info('target for data stream ' + str(name) + ' is filtered negative sampling')
        data_stream = FilteredNegativeSampling(data_stream, k=k, **neg_sample_kwargs)
    elif target == 'score':
        logger.info('target for data stream ' + str(name) + ' is score')
    else:
        raise NotImplementedError('target ', target, ' must be one of "score" or "negative_sampling"')

    data_stream = MergeSource(data_stream, merge_sources=('head', 'tail', 'head_mask', 'tail_mask', 'rel'),
        merge_name='input')

    return data_stream, batches_per_epoch


def _liacl_add_closest_neighbour(stream):
    """
    Adds closest neighbour to stream by first collecting 1 epoch of stream, ignoring negative samples
    """
    raise NotImplementedError()


class LiACLDatasetFromFile(object):
    """
    Flexible way to serve dataset in format used in Li et al. ACL paper.

    Notes
    -----
    File is assumed to be in following line format: head tail rel score
    """

    def __init__(self, file_path, rel_file_path=LiACL_ON_REL):
        self.file_path = file_path
        self.load_rel2index(rel_file_path)
        self.dataset = self.load_data(file_path)

    def load_data(self, data_path):
        logging.info("Loading: " + data_path)

        data = pd.read_csv(data_path, sep="\t", header=None)
        data.columns = ['rel', 'head', 'tail', 'score']
        assert (not data.empty)
        self.N = len(data)
        return IndexableDataset(data.to_dict('list'))

    def load_rel2index(self, rel_file):
        rel2index = {}

        if not os.path.isabs(rel_file):
            rel_file = os.path.join(DATA_DIR, rel_file)

        logging.info("Loading: " + rel_file)

        with open(rel_file, 'r') as f:
            for index, line in enumerate(f):
                rel2index[line.strip()] = index

        self.rel2index = rel2index
        self.rel_vocab_size = len(rel2index)

    def data_stream(self, batch_size, word2index, target='negative_sampling', add_neighbours=0,
            name=None, shuffle=False, **kwargs):
        return _liacl_data_stream(self.dataset, self.rel2index, batch_size, word2index,
            target=target, name=name, shuffle=shuffle, **kwargs)


class LiACLSplitDataset(object):
    """
    Class wrapping dataset used originally in Li et al. ACL paper.

    Notes
    -----
    Bakes in original splitting and gives interface for just getting train/dev/dev2/test
    streams without thinking too much

    Files:
        * train100k.txt (TODO(kudkudak): Replace with train.txt, or pass as param N)
        * dev1.txt
        * dev2.txt
        * test.txt
        * rel.txt
    """
    REL_FILE = 'rel.txt'
    TRAIN_FILE = 'train100k.txt'
    TEST_FILE = 'test.txt'
    DEV1_FILE = 'dev1.txt'
    DEV2_FILE = 'dev2.txt'

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_rel2index()
        self.train_dataset = self.load_data(LiACLSplitDataset.TRAIN_FILE)
        self.dev1_dataset = self.load_data(LiACLSplitDataset.DEV1_FILE)
        self.dev2_dataset = self.load_data(LiACLSplitDataset.DEV2_FILE)
        self.test_dataset = self.load_data(LiACLSplitDataset.TEST_FILE)

    def load_data(self, data_path):
        data_path = os.path.join(self.data_dir, data_path)
        if not os.path.isabs(data_path):
            data_path = os.path.join(DATA_DIR, data_path)

        logging.info("Loading: " + data_path)

        data = pd.read_csv(data_path,
            sep="\t", header=None)
        data.columns = ['rel', 'head', 'tail', 'score']
        assert (not data.empty)

        dataset = IndexableDataset(data.to_dict('list'))
        return dataset

    def load_rel2index(self):
        rel2index = {}
        rel_file = os.path.join(self.data_dir, LiACLSplitDataset.REL_FILE)

        if not os.path.isabs(rel_file):
            rel_file = os.path.join(DATA_DIR, rel_file)

        logging.info("Loading: " + rel_file)

        with open(rel_file, 'r') as f:
            for index, line in enumerate(f):
                rel2index[line.strip()] = index

        self.rel2index = rel2index
        self.rel_vocab_size = len(rel2index)

    def train_data_stream(self, batch_size, word2index, **args):
        return self.data_stream(self.train_dataset, batch_size, word2index,
            name='train', **args)

    def test_data_stream(self, batch_size, word2index, target="score", **args):
        return self.data_stream(self.test_dataset, batch_size, word2index,
            target=target, name='test', **args)

    def dev1_data_stream(self, batch_size, word2index, target="score", **args):
        return self.data_stream(self.dev1_dataset, batch_size, word2index,
            target=target, name='dev1', **args)

    def dev2_data_stream(self, batch_size, word2index, target="score", **args):
        return self.data_stream(self.dev2_dataset, batch_size,  word2index,
            target=target, name='dev2', **args)

    def data_stream(self, dataset, batch_size, word2index, target='negative_sampling', add_neighbours=0, name=None,
            shuffle=False, **kwargs):
        return _liacl_data_stream(dataset, self.rel2index, batch_size, word2index,
            target=target, name=name, shuffle=shuffle, **kwargs)


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


class FilteredNegativeSampling(Transformer):
    """
    Params
    ------
    filter_fnc: function
        Function taking in head, rel, tail matrix and producing 1 and 0 vector indicating if we
        accept this sample or not
    """

    def __init__(self, data_stream, filter_fnc, k=3, target="cls", *args, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches, '
                             'not examples.')

        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)

        self.k = k
        self.filter_fnc = filter_fnc

        self.target = target

        self.batch_id = 0
        self.samples_per_batch = 0

        super(FilteredNegativeSampling, self).__init__(data_stream,
            produces_examples=False,
            *args,
            **kwargs)

    @property
    def sources(self):
        return self.data_stream.sources + ('target',)

    def transform_batch(self, batch):
        head, head_mask, tail, tail_mask, rel = batch
        batch_size = len(head)

        head_list, head_mask_list, rel_list, tail_list, tail_mask_list, target_list = \
            [head], [head_mask], [rel], [tail], [tail_mask], [np.array([1] * batch_size)]

        k = 0

        # TODO: Parametrize elegantly by resampling procedure, but later?
        # For now just have argsim. Or maybe do not have the resampling?

        # TODO: Add from whole batch.

        # The way it goes is sample in while loop until it found examples fooling ArgSim
        while sum([len(x) for x in head_list]) < (self.k + 1) * batch_size:
            neg_rels_idx_sample = np.random.randint(batch_size, size=batch_size)
            neg_head_idx_sample = np.random.randint(batch_size, size=batch_size)
            neg_tail_idx_sample = np.random.randint(batch_size, size=batch_size)

            neg_rel_sample = rel[neg_rels_idx_sample]
            neg_head_sample = head[neg_head_idx_sample]
            neg_tail_sample = tail[neg_tail_idx_sample]
            neg_head_mask_sample = head_mask[neg_head_idx_sample]
            neg_tail_mask_sample = tail_mask[neg_tail_idx_sample]

            rel_sample = np.concatenate([neg_rel_sample, rel, rel], axis=0).reshape(-1, 1)
            head_sample = np.concatenate([head, neg_head_sample, head], axis=0)
            tail_sample = np.concatenate([tail, tail, neg_tail_sample], axis=0)
            head_mask_sample = np.concatenate([head_mask, neg_head_mask_sample, head_mask], axis=0)
            tail_mask_sample = np.concatenate([tail_mask, tail_mask, neg_tail_mask_sample], axis=0)
            target_sample = np.array([0] * batch_size * 3)
            type = np.array([0] * batch_size + [1] * batch_size + [2] * batch_size)

            accept, scores = self.filter_fnc(head_sample, None, tail_sample)

            assert len(accept) == len(head_sample)

            # .reshape(-1,) to be compatible with array of lists that is sometimes produced
            # if examples have varying length (remember padding is after negsampling ATM)
            # this could be changed easily, no reason to have padding after
            head_list.append(head_sample[accept])
            head_mask_list.append(head_mask_sample[accept])
            rel_list.append(rel_sample[accept])
            tail_list.append(tail_sample[accept])
            tail_mask_list.append(tail_mask_sample[accept])

            if self.target == "cls":
                target_list.append(target_sample[accept])
            elif self.target == "score":
                target_list.append(scores[accept])
            elif self.target == "type":
                target_list.append(type[accept])
            else:
                raise NotImplementedError()

            k += 3 * batch_size

        self.samples_per_batch = 0.9 * self.samples_per_batch + 0.1 * k
        if self.batch_id % 100 == 0:
            print("Avg samples_per_batch={}".format(self.samples_per_batch))
        self.batch_id += 1

        rel = np.concatenate(rel_list, axis=0)[0:(self.k + 1) * batch_size]
        head = np.concatenate(head_list, axis=0)[0:(self.k + 1) * batch_size]
        tail = np.concatenate(tail_list, axis=0)[0:(self.k + 1) * batch_size]
        head_mask = np.concatenate(head_mask_list, axis=0)[0:(self.k + 1) * batch_size]
        tail_mask = np.concatenate(tail_mask_list, axis=0)[0:(self.k + 1) * batch_size]
        target = np.concatenate(target_list, axis=0)[0:(self.k + 1) * batch_size]

        return (head, head_mask, tail, tail_mask, rel, target)


class NegativeSampling(Transformer):
    def __init__(self, data_stream, k=3, *args, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches, '
                             'not examples.')

        self.k = k
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
        head, head_mask, tail, tail_mask, rel = batch
        batch_size = len(head)

        neg_rels_idx = np.random.randint(batch_size, size=batch_size)
        neg_head_idx = np.random.randint(batch_size, size=batch_size)
        neg_tail_idx = np.random.randint(batch_size, size=batch_size)

        neg_rel = rel[neg_rels_idx]
        neg_head = head[neg_head_idx]
        neg_tail = tail[neg_tail_idx]
        neg_head_mask = head_mask[neg_head_idx]
        neg_tail_mask = tail_mask[neg_tail_idx]

        rel = np.concatenate([rel, neg_rel, rel, rel], axis=0)
        head = np.concatenate([head, head, neg_head, head], axis=0)
        tail = np.concatenate([tail, tail, tail, neg_tail], axis=0)
        head_mask = np.concatenate([head_mask, head_mask, neg_head_mask, head_mask], axis=0)
        tail_mask = np.concatenate([tail_mask, tail_mask, tail_mask, neg_tail_mask], axis=0)

        # TODO(kudkudak): This is a terrible hack
        if self.k < 3:
            # Can 1/3 of it be false negative?
            assert len(head) == 4*batch_size
            ids = range(batch_size)
            ids_chosen = np.random.choice(batch_size*3, batch_size*self.k, replace=False)
            ids = ids + [iid + batch_size for iid in ids_chosen]
            rel = rel[ids]
            head = head[ids]
            tail = tail[ids]
            head_mask = head_mask[ids]
            tail_mask = tail_mask[ids]
        elif self.k > 3:
            raise NotImplementedError()

        target = np.array([1] * batch_size + [0] * batch_size * self.k)

        return (head, head_mask, tail, tail_mask, rel, target)


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
        merged_data = {s: d for s, d in zip(self.data_stream.sources, data)
            if s in self.merge_sources}
        return [merged_data] + [d for d, s in zip(data, self.data_stream.sources)
            if s not in self.merge_sources]
