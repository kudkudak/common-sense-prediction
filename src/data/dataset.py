"""
Implementatons of datasets. We have following datasets used in project:

* LiACL/conceptnet dataset
* LiACL/tuples.wiki evaluation dataset

TODO(kudkudak): merge LiACLDatasetFromFile and LiACLSplitDataset
TODO(kudkudak): code parametrization that is indepednent of threshold, just resamples. Like in GAN KB paper
TODO(kudkudak): NegativeSampling is limited now to k=3,2,1, wierd
"""

import os
import logging

import fuel
from fuel.datasets import IndexableDataset
import numpy as np
import pandas as pd

from src import DATA_DIR
from src.data.data_stream import liacl_data_stream
from src.data.vocab import Vocabulary

logger = logging.getLogger(__name__)

# Extrinsic evaluation dataset
LiACL_CN_DATASET = os.path.join(DATA_DIR, "LiACL", "conceptnet")
LiACL_ON_REL = os.path.join(LiACL_CN_DATASET, "rel.txt")
LiACL_OMCS_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings/LiACL/embeddings_OMCS.txt")
# TODO(kudkudak): Add sth to figure out lowercase automatically. Idk.
# LiACL_ON_REL_LOWERCASE = os.path.join(LiACL_CN_DATASET, "rel_lowercase.txt")
assert os.path.exists(LiACL_ON_REL)
# assert os.path.exists(LiACL_ON_REL_LOWERCASE)
assert os.path.exists(LiACL_OMCS_EMBEDDINGS)
TUPLES_WIKI = os.path.join(DATA_DIR, "LiACL", "tuples.wiki")


class Dataset(object):
    def __init__(self, name, filepath, data_dir=''):
        self.name = name
        self.filepath = os.path.join(data_dir, filepath)
        self.dataset = self.load_data(self.filepath)

    def load_data(self, filepath):
        logging.info("Loading: " + filepath)

        data = pd.read_csv(filepath,
            sep="\t", header=None)
        data.columns = ['rel', 'head', 'tail', 'score']
        assert (not data.empty)
        return IndexableDataset(data.to_dict('list'))

    def data_stream(self, batch_size, vocab, rel_vocab, target, shuffle=False,
                    **kwargs):
        return liacl_data_stream(dataset=self.dataset,
                                 vocab=vocab,
                                 rel_vocab=rel_vocab,
                                 batch_size=batch_size,
                                 target=target,
                                 name=self.name,
                                 shuffle=shuffle,
                                 **kwargs)


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
    TRAIN_FILE = 'train100k.txt'
    TEST_FILE = 'test.txt'
    DEV1_FILE = 'dev1.txt'
    DEV2_FILE = 'dev2.txt'
    VOCAB_FILE = 'vocab.txt'
    REL_FILE = 'rel.txt'

    def __init__(self, data_dir):
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(DATA_DIR, data_dir)

        self.data_dir = data_dir
        self.train = Dataset(name='train',
                             filepath=self.TRAIN_FILE,
                             data_dir=data_dir)
        self.dev1 = Dataset(name='dev1',
                            filepath=self.DEV1_FILE,
                            data_dir=data_dir)
        self.dev2 = Dataset(name='dev2',
                             filepath=self.DEV2_FILE,
                             data_dir=data_dir)
        self.test = Dataset(name='test',
                            filepath=self.TEST_FILE,
                            data_dir=data_dir)
        self.vocab = Vocabulary(os.path.join(data_dir, self.VOCAB_FILE))
        self.rel_vocab = Vocabulary(os.path.join(data_dir, self.REL_FILE))

    def train_data_stream(self, batch_size, target='negative_sampling', **kwargs):
        return self.train.data_stream(batch_size,
                                      self.vocab,
                                      self.rel_vocab,
                                      target,
                                      **kwargs)

    def dev1_data_stream(self, batch_size, target='score', **kwargs):
        return self.dev1.data_stream(batch_size,
                                      self.vocab,
                                      self.rel_vocab,
                                      target,
                                      **kwargs)

    def dev2_data_stream(self, batch_size, target='score', **kwargs):
        return self.dev2.data_stream(batch_size,
                                      self.vocab,
                                      self.rel_vocab,
                                      target,
                                      **kwargs)

    def test_data_stream(self, batch_size, target='score', **kwargs):
        return self.test.data_stream(batch_size,
                                      self.vocab,
                                      self.rel_vocab,
                                      target,
                                      **kwargs)


