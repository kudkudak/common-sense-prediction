#TODO:
# connect data loading to model
# create config for model call
# create input with vegab
# validation data?
# double-check consistency with original

import keras
from keras.optimizers import Adagrad

from data import Dataset
from model import dnn_ce

DATA_DIR = '/home/mnoukhov/common-sense-prediction/data'
BATCH_SIZE = 600
ACTIVATION = 'relu'
HIDDEN_UNITS = 1000
REL_VEC_SIZE = 200
LAMBDA = 0.001
LEARNING_RATE = 0.01


def main(data_dir):
    dataset = Dataset(data_dir)
    embeddings = dataset.embeddings
    data_stream = dataset.data_stream(BATCH_SIZE)
    model = dnn_ce()
    model.compile(Adagrad(LEARNING_RATE),
                  'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(something,
              target,
              batch_size,
              epochs=1)


if __name__ == '__main__':
    main(DATA_DIR)
