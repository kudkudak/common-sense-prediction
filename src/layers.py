# -*- coding: utf-8 -*-
"""
Implementatons of keras layers
"""

import keras
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec


class Bilinear(keras.layers.Layer):
    """Bilinear layer.

    This layers take two inputs:
    (a) a vector of concatenated head&tail embeddings
    (b) the relation id

    It extracts the head embedding h, the tail embedding t, and the computes

       h.dot(M^rel).dot(t)

    """
    def __init__(self, num_relations,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.num_relations = num_relations
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_spec = [InputSpec(ndim=2), InputSpec(shape=(None, 1,))]

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][1]
        self.dim = int(input_dim / 2)
        self.kernel = self.add_weight(shape=(self.num_relations, self.dim, self.dim),
                                      initializer=self.kernel_initializer,
                                      name='matrices')
        self.input_spec = [InputSpec(ndim=2, axes={1: input_dim}), InputSpec(shape=(None, 1,))]
        self.built = True

    def call(self, inputs):
        heads_and_tails, relations = inputs
        heads = heads_and_tails[:, :self.dim]
        tails = heads_and_tails[:, -self.dim:]
        relations = relations[:, 0]

        matrices = self.kernel[relations]
        result = K.batch_dot(heads, matrices, axes=1)
        result = K.batch_dot(result, tails, axes=1)
        return result

    def compute_output_shape(self, input_shape):
        assert len(input_shape[0]) == 2
        assert input_shape[1][1] == 1
        return (input_shape[0], 1)


    def get_config(self):
        config = {
            'num_relations': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(Bilinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
