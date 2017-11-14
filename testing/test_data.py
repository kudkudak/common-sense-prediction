#TODO(mnuke): make this real unittest?
from keras.models import Model
from keras.layers import Input, Embedding, Flatten

from src import DATA_DIR
from src.data import LiACLSplitDataset
from scripts.train_dnn_ce import endless_data_stream


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


if __name__ == '__main__':
    test_integration()
