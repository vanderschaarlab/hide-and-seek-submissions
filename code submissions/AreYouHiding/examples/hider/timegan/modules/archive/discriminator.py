import tensorflow as tf
from .model_utils import rnn_cell, rnn_choices

class Discriminator(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.module_name = args.module_name
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        # self.pre_rnn_cells = [rnn_cell(self.module_name, self.hidden_dim) for _ in range(self.num_layers)]
        # self.rnn_cells = tf.keras.layers.StackedRNNCells(self.pre_rnn_cells)
        # self.rnn = tf.keras.layers.RNN(self.rnn_cells,
        #                                return_sequences=True, 
        #                                stateful=False,
        #                                return_state=False)
        self.rnn = tf.keras.Sequential([
            rnn_choices(self.module_name, self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.linear = tf.keras.layers.Dense(1, activation=None)

    def call(self, H, training=False):
        d_output = self.rnn(H)
        Y_hat = self.linear(d_output)
        return Y_hat