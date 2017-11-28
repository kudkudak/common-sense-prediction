# -*- coding: utf-8 -*-
"""
Implementatons of keras layers
"""

import keras
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Lambda

#
# class MergeRelSplit(keras.layers.Layer):
#     """
#
#     Merged differently in each rel
#
#     """
#     def __init__(self, num_relations, k, type="max",
#                  **kwargs):
#         super(MergeRelSplit, self).__init__(**kwargs)
#         self.num_relations = num_relations
#         self.k = k
#         self.type = type
#
#         if type != "max":
#             raise NotImplementedError()
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[0][1]
#         self.dim = int(input_dim / 2)
#         self.kernel = self.add_weight(shape=(self.num_relations, self.k),
#                                       initializer="ones",
#                                       name='matrices')
#         # s1, s2, s3
#         self.input_spec = [InputSpec(shape=(None, self.k,)), InputSpec(shape=(None, 1,))]
#         self.built = True
#
#     def call(self, inputs):
#         heads_and_tails, relations = inputs
#         heads = heads_and_tails[:, :self.dim]
#         tails = heads_and_tails[:, -self.dim:]
#         relations = relations[:, 0]
#
#         matrices = self.kernel[relations]
#         result = K.batch_dot(heads, matrices, axes=1)
#         result = K.batch_dot(result, tails, axes=1)
#         return result
#
#     def compute_output_shape(self, input_shape):
#         assert len(input_shape[0]) == 2
#         assert input_shape[1][1] == 1
#         return (input_shape[0], 1)
#
#
#     def get_config(self):
#         config = {
#             'num_relations': self.units,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#         }
#         base_config = super(Bilinear, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


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


def MaskAvg(output_shape, **args):
    def _mask_avg(inputs):
        x, mask = inputs
        assert K.ndim(x) == 3  # (n_batch, len, dim)
        assert K.ndim(mask) == 2  # (n_batch, len)
        # return K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True)
        return K.sum(x * mask, axis=1)

    return Lambda(_mask_avg, output_shape=output_shape, **args)

