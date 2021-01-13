#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:28:24 2020

@author: chkaul
"""

from typing import Dict
import numpy as np
from utils.data_preprocess import preprocess_data


import tempfile
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


def binary_cross_entropy_loss(y_true, y_pred):
    """User defined cross entropy loss.

    Args:
        - y_true: true labels
        - y_pred: predictions

    Returns:
        - loss: computed loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # Cross entropy loss excluding masked labels
    loss = -(idx * y_true * tf.math.log(y_pred + 1e-6) + idx * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6))
    return loss

# class defining the custom attention layer
class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.t_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


enl_no, seq_len, dim = enlarged_data.shape
gen_no, _, _ = generated_data.shape

task= "classification"
model_type= "gru"
h_dim= dim
n_layer= 3
batch_size= 128
epoch= 20
learning_rate= 0.001

dim = 71
max_seq_len = 100

model = tf.keras.Sequential()
model.add(layers.Masking(mask_value=-1.0, input_shape=(max_seq_len, dim)))
model.add(layers.Bidirectional(layers.GRU(100, return_sequences=True)))
model.add(HierarchicalAttentionNetwork(100))
model.add(layers.Dense(1, activation='sigmoid'))


adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss=binary_cross_entropy_loss, optimizer=adam)


import os
with tempfile.TemporaryDirectory() as tmpdir:
    save_file_name = os.path.join(tmpdir, "model.ckpt")

# Callback for the best model saving
save_best = ModelCheckpoint(
    save_file_name,
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    )

train_x = np.concatenate((generated_data.copy(), enlarged_data.copy()), axis=0)
train_y = np.concatenate((np.zeros([gen_no, 1]), np.ones([enl_no, 1])), axis=0)

idx = np.random.permutation(enl_no + gen_no)
train_x = train_x[idx, :, :]
train_y = train_y[idx, :]

# Train the model
model.fit(
    train_x,
    train_y,
    batch_size=batch_size,
    epochs=epoch,
    validation_split=0.2,
    verbose=1,
    )

test_y_hat = model.predict(enlarged_data)