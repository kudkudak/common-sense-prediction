import keras
from keras.layers import (Input,
                          Dense,
                          Embedding,
                          average,
                          concatenate)


def dnn_ce(batch_size, max_sequence_length, embedding_init, vocab_size,
           rel_embedding_init, hidden_units, hidden_activation):
    rel = Input(shape=(batch_size,), dtype=int32)
    head = Input(shape=(max_sequence_length), dtype=int32)
    head_mask = Input(shape=(max_sequence_length), dtype=int32)
    tail = Input(shape=(batch_size, max_sequence_length), dtype=int32)
    tail_mask = Input(shape=(batch_size, max_sequence_length), dtype=int32)


    output_dim = embedding_init.shape[1]
    embedding_layer = Embedding(vocab_size,
                                output_dim,
                                weights=[embedding_init],
                                trainable=True)
    rel_embedding_layer = Embedding(rel_vocab_size,
                                    rel_embedding_size,
                                    weights=[rel_embedding_init],
                                    trainable=True)

    head_emb = embedding_layer(head)
    tail_emb = embedding_layer(tail)
    v1 = average(head_emb, input_mask=head_mask, axis=1)
    v2 = average(tail_emb, input_mask=tail_mask, axis=1)
    vr = rel_embedding_layer(rel)
    vin = concatenate([v1, vr, v2], axis=1)

    u = Dense(hidden_units, activation=hidden_activation)(vin)
    output = Dense(1, activation='sigmoid')(u)

    model = Model([rel, head, head_mask, tail, tail_mask],
                  [output])

    return model




