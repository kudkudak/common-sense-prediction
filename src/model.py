import keras
from keras.models import Sequential
from keras.layers import (Input,
                          Dense,
                          Embedding,
                          Average,
                          Masking,
                          Concatenate)


def dnn_ce(max_sequence_length, embedding_init, vocab_size,
           rel_embedding_init, rel_vocab_size, hidden_units,
           hidden_activation):
    rel_embedding_size = rel_embedding_init.shape[1]
    rel_embedding_layer = Embedding(rel_vocab_size,
                                    rel_embedding_size,
                                    weights=[rel_embedding_init],
                                    trainable=True,
                                    input_shape=(max_sequence_length,))
    embedding_size = embedding_init.shape[1]
    embedding_layer = Embedding(vocab_size,
                                embedding_size,
                                weights=[embedding_init],
                                trainable=True,
                                mask_zero=True,
                                input_shape=(max_sequence_length,))

    rel_input = Sequential()
    # rel_input.add(Input(shape=(1,), dtype='int32'))
    rel_input.add(rel_embedding_layer)

    head_input = Sequential()
    # head_input.add(Input(shape=(max_sequence_length,), dtype='int32'))
    # head_input.add(Masking(mask_value=0))
    head_input.add(embedding_layer)
    head_input.add(Average())

    tail_input = Sequential()
    # tail_input.add(Input(shape=(max_sequence_length,), dtype='int32'))
    # tail_input.add(Masking(mask_value=0))
    tail_input.add(embedding_layer)
    tail_input.add(Average())

    model = Sequential()
    model.add(Concatenate([head_input, tail_input, rel_input], axis=1))
    model.add(Dense(hidden_units, activation=hidden_activation))
    model.add(Dense(1, activation='sigmoid'))

    # head_emb = embedding_layer(head)
    # tail_emb = embedding_layer(tail)
    # v1 = average(head_emb, input_mask=head_mask, axis=1)
    # v2 = average(tail_emb, input_mask=tail_mask, axis=1)
    # vr = rel_embedding_layer(rel)
    # vin = concatenate([v1, vr, v2], axis=1)

    # u = Dense(hidden_units, activation=hidden_activation)(vin)
    # output = Dense(1, activation='sigmoid')(u)

    # model = Model([rel, head, head_mask, tail, tail_mask],
                  # [output])

    return model




