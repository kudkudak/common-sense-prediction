import json
import os

import numpy as np
import keras
from keras.models import load_model

from src import DATA_DIR
from src.data import Dataset
from src.evaluate import evaluate_fit_threshold
from src.utils.vegab import wrap_no_config_registry
from src.utils.data_loading import load_embeddings


def evaluate(save_path):
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config = json.load(f)

    word2index, embeddings = load_embeddings(DATA_DIR, config['embedding_file'])
    dataset = Dataset(DATA_DIR)
    model = load_model(os.path.join(save_path, 'model.h5'))

    dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index)
    dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index)
    test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index)

    eval_results = evaluate_fit_threshold(model, dev1_stream, dev2_stream, test_stream)
    json.dump(eval_results, open(os.path.join(save_path, "eval_results.json"), "w"))


if __name__ == '__main__':
    wrap_no_config_registry(evaluate)

