#TODO(mnuke): make this real unittest?
from keras.models import Model
from keras.layers import Input, Embedding, Flatten

from src import DATA_DIR
from src.data.dataset import LiACLSplitDataset
from src.data.data_stream import endless_data_stream


def test_integration():
    BATCH_SIZE = 600
    data = LiACLSplitDataset(DATA_DIR)
    data_stream, batches = data.train_data_stream(BATCH_SIZE)
    endless = endless_data_stream(data_stream)

    rel_input = Input(shape=(1,), dtype='int32', name='rel')
    rel = Embedding(34, 1, trainable=True)(rel_input)
    rel = Flatten()(rel)
    model = Model([rel_input], [rel])
    model.compile('sgd', 'mean_squared_error')

    model.fit_generator(generator=endless,
                        steps_per_epoch=batches,
                        epochs=2,
                        verbose=1)


def test_stream():
    from timeit import default_timer as timer

    BATCH_SIZE = 1000
    data_dir = 'LiACL/conceptnet/'
    data = LiACLSplitDataset(data_dir)
    data_stream, batches = data.train_data_stream(BATCH_SIZE)
    start = timer()
    for _ in data_stream.get_epoch_iterator():
        pass
    end = timer()
    print(end - start)


if __name__ == '__main__':
    test_stream()
