# -*- coding: utf-8 -*-
import logging
import os

import numpy as np

from src import DATA_DIR

logger = logging.getLogger(__name__)


class Embedding(object):
    def __init__(self, filepath, vocab=None):
        self.words = []
        self.word_to_id = {}
        self.embeddings = []

        if not os.path.isabs(filepath):
            filepath = os.path.join(DATA_DIR, filepath)

        with open(filepath, 'r') as f:
            for index, line in enumerate(f):
                values = line.split()
                word = values[0]
                self.words.append(word)
                self.word_to_id[word] = index
                self.embeddings.append([float(val) for val in values[1:]])

        self.embeddings = np.matrix(self.embeddings)

        if vocab is not None:
            self.embeddings = self.fit_vocab(vocab)

    @property
    def values(self):
        return self.embeddings

    @property
    def embed_size(self):
        return self.embeddings.shape[1]

    def fit_vocab(self, vocab):
        vocab_embedding = np.zeros((vocab.size, self.embed_size))
        unk_index = self.word_to_id.get('<unk>', None)
        for i, word in enumerate(vocab.words):
            index = self.word_to_id.get(word, unk_index)
            if index is not None:
                vocab_embedding[i] = self.embeddings[index]

        return vocab_embedding


def load_embeddings(embedding_file, vocab):
    """load embeddings restricted by a vocab

    relies on vocab having an unk character"""

    if not os.path.isabs(embedding_file):
        embedding_file = os.path.join(DATA_DIR, embedding_file)

    word_to_index = vocab.word_to_index

    embeddings = [None]*vocab.size
    with open(embedding_file,'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            index = word_to_index(word)
            embeddings[index] = [float(val) for val in values[1:]]

    for index, emb in embeddings:
        if emb is None:
            embeddings[index] = embeddings[vocab.unk]

    return np.matrix(embeddings)

def load_external_embeddings(data_dir, embedding_file, ext_sub_embedding_file, main_word2index, cache=True):
    # TODO(kudkudak): Remove data_dir parameter
    if not os.path.isabs(embedding_file):
        embedding_file = os.path.join(data_dir, embedding_file)
    if not os.path.isabs(ext_sub_embedding_file):
        ext_sub_embedding_file = os.path.join(data_dir, ext_sub_embedding_file)
    if os.path.isfile(ext_sub_embedding_file):
        # TODO(kudkudak): Remove support for sub. It is confusing. On top of that: should double-check alignment
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
        # TODO(kdukudak): Remove this mess with ext_sub_embedding_file. These caches are extremeley dangerous
        if cache:
            with open(ext_sub_embedding_file,'w') as f:
                for word in main_word2index.keys():
                    f.write(word+" "+" ".join(corresponding_embeddings[main_word2index[word]].astype(str))+"\n")
    return corresponding_embeddings


