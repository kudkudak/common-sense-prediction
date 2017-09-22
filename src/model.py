from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def dnn_li16(config):
    model = Sequential()

    # concat v1, v2, vr
    # dense layer D1
    # relu
    # dense layer D2
    model.add(Embedding(
    model.add(Dense(1000))
    model.add(Activation('relu'))
