#TODO:
# connect data loading to model
# create config for model call
# create input with vegab
# validation data?
# double-check consistency with original

import keras
from keras.optimizers import Adagrad
import numpy as np

from data import Dataset
from model import dnn_ce

DATA_DIR = '/home/mnoukhov/common-sense-prediction/data'
# BATCH_SIZE = 600
BATCH_SIZE = 2
REL_INIT = 0.02
REL_VEC_SIZE = 200
ACTIVATION = 'relu'
HIDDEN_UNITS = 1000
LAMBDA = 0.001
LEARNING_RATE = 0.01
MAX_SEQUENCE_LENGTH = 20


def main(data_dir):
    dataset = Dataset(data_dir)
    embeddings = dataset.embeddings

    rel_embeddings_init = np.random.uniform(-REL_INIT, REL_INIT,
                                           (dataset.rel_vocab_size, REL_VEC_SIZE))


    data_stream = dataset.data_stream(BATCH_SIZE)
    model = dnn_ce(MAX_SEQUENCE_LENGTH,
                   dataset.embeddings,
                   dataset.vocab_size,
                   rel_embeddings_init,
                   dataset.rel_vocab_size,
                   HIDDEN_UNITS,
                   ACTIVATION)

    model.compile(Adagrad(LEARNING_RATE),
                  'binary_crossentropy',
                  metrics = ['accuracy'])

    num_batches = dataset.dataset.num_examples / BATCH_SIZE

    model.fit_generator(data_stream.get_epoch_iterator(),
                        steps_per_epoch=10,
                        epochs=1)


if __name__ == '__main__':
    main(DATA_DIR)
