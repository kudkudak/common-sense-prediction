#TODO:
# validation data?
# double-check consistency with original

from keras.optimizers import Adagrad
import numpy as np

from configs.config_reproduce import config
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver
from src.data import Dataset
from src.model import dnn_ce
from src import DATA_DIR


def train(config, save_path):
    dataset = Dataset(DATA_DIR)
    rel_embeddings_init = np.random.uniform(-config['rel_init'], config['rel_init'],
                                            (dataset.rel_vocab_size, config['rel_vec_size']))
    model = dnn_ce(dataset.embeddings,
                   dataset.vocab_size,
                   rel_embeddings_init,
                   dataset.rel_vocab_size,
                   config['hidden_units'],
                   config['activation'])

    model.compile(Adagrad(config['learning_rate']),
                  'binary_crossentropy',
                  metrics = ['accuracy'])

    train_iterator = dataset.train_data_stream(config['batch_size']).get_epoch_iterator()
    test_iterator = dataset.test_data_stream(config['batch_size']).get_epoch_iterator()
    num_batches = dataset.train_dataset.num_examples / config['batch_size']

    training_loop(model=model,
                  train=train_iterator,
                  valid=None,
                  n_epochs=config['n_epochs'],
                  steps_per_epoch=num_batches,
                  valid_steps=config['valid_steps'],
                  save_path=save_path,
                  learning_rate_schedule=None)


if __name__ == '__main__':
    wrap(config, train, plugins=[MetaSaver()])
