#TODO(mnuke): make this real unittest?
from timeit import default_timer as timer

from keras.optimizers import Adagrad

from src.data.dataset import LiACLSplitDataset
from src.data.embedding import Embedding
from src.data.stream import endless_data_stream
from src.model import dnn_ce


def test_dnn_ce():
    from src.configs.configs_dnn_ce import config
    config = config.get_root_config()

    dataset = LiACLSplitDataset(config['data_dir'])
    embedding = Embedding(config['embedding_file'], dataset.vocab)
    data_stream, batches = dataset.train_data_stream(config['batch_size'])

    model = dnn_ce(embedding_init=embedding.values,
                   vocab_size=dataset.vocab.size,
                   embedding_size=embedding.embed_size,
                   use_embedding=config['use_embedding'],
                   l2=config['l2'],
                   rel_init=config['rel_init'],
                   rel_vocab_size=dataset.rel_vocab.size,
                   rel_embed_size=config['rel_vec_size'],
                   hidden_units=config['hidden_units'],
                   hidden_activation=config['activation'],
                   batch_norm=config['batch_norm'])
    model.compile(optimizer=Adagrad(config['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    start = timer()
    model.fit_generator(generator=endless_data_stream(data_stream),
                        steps_per_epoch=batches,
                        epochs=2,
                        verbose=1)

    end = timer()
    print(end - start)


def test_stream():
    BATCH_SIZE = 10
    data_dir = 'LiACL/conceptnet/'
    data = LiACLSplitDataset(data_dir)
    data_stream, batches = data.train_data_stream(BATCH_SIZE)
    start = timer()
    for _ in data_stream.get_epoch_iterator():
        pass
    end = timer()
    print(end - start)


if __name__ == '__main__':
    test_dnn_ce()
