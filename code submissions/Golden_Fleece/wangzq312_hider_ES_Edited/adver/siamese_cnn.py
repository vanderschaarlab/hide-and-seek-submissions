from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
# Ref: https://github.com/ardiya/siamesenetwork-tensorflow/
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def feature_model(input, reuse=False):
    # Construct a CNN for feature extraction
    with tf.name_scope("perturb"):
        with tf.variable_scope("add_noise", reuse=reuse) as scope:
            # noise = tf.Variable(initial_value=tf.zeros(input.shape))
            noise = tf.get_variable("noise", initializer=tf.zeros((100, 71, 1)))
            input = tf.add(input, noise)

    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 64, [5, 5], activation_fn=tf.nn.sigmoid, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope=scope, reuse=reuse)
                                           # , normalizer_fn=tf.contrib.slim.batch_norm)
            net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=scope)
            net = tf.nn.dropout(net, rate=0.25)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')


        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=tf.nn.sigmoid, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope=scope, reuse=reuse)
                                           # , normalizer_fn=tf.contrib.slim.batch_norm)
            net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=scope)
            net = tf.nn.dropout(net, rate=0.25)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.sigmoid, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope=scope, reuse=reuse)
                                           # , normalizer_fn=tf.contrib.slim.batch_norm)
            net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=scope)
            net = tf.nn.dropout(net, rate=0.25)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("fc1") as scope:
            net = tf.contrib.layers.flatten(net)
            net = tf.contrib.layers.fully_connected(net, 256, scope=scope, reuse=reuse, activation_fn=None)
                                                    # normalizer_fn=tf.contrib.slim.batch_norm)
            net = tf.contrib.layers.layer_norm(net, reuse=reuse, scope=scope)

        with tf.variable_scope("fc2") as scope:
            net = tf.contrib.layers.fully_connected(net, 128, scope=scope, reuse=reuse, activation_fn=None)
            net = tf.nn.sigmoid(net)

    return net, noise


def contrastive_loss(model1, model2, y, margin):
    ''' Calculate the contrastive loss, which minimize the distance between
        embeddings "model1" and "model2" if they come from the same user, and
        maximize the distance if they come from different users
    '''
    with tf.name_scope("contrastive-loss"):
        # all_embedding = tf.concat([model1, model2], axis=0)
        #
        # model1 = tf.div(
        #     tf.subtract(
        #         model1,
        #         tf.reduce_min(all_embedding)
        #     ),
        #     tf.add(
        #         tf.subtract(
        #             tf.reduce_max(all_embedding),
        #             tf.reduce_min(all_embedding)
        #         ),
        #         tf.constant(1e-7, dtype=tf.float32)
        #     )
        # )
        #
        # model2 = tf.div(
        #     tf.subtract(
        #         model2,
        #         tf.reduce_min(all_embedding)
        #     ),
        #     tf.add(
        #         tf.subtract(
        #             tf.reduce_max(all_embedding),
        #             tf.reduce_min(all_embedding)
        #         ),
        #     tf.constant(1e-7, dtype=tf.float32)
        #     )
        # )

        distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance),0))
         # give penalty to dissimilar label if the distance is bigger than margin
        return tf.reduce_mean(dissimilarity + similarity) / 2, distance

class Dataset(object):
    '''
        Process the dataset to generate siamese pairs. For each pair, the data
        can come from either the same user or different user, by 50% chance.
        For "similar pairs", we add Gaussian noise to the second data entry
        For "dissimilar pairs", we select data entries from different users
    '''

    def __init__(self, X):
        X = X[:, :, :, np.newaxis]
        y = np.arange(len(X))
        self.images_train, self.image_test, self.labels_train, self.labels_test \
            = train_test_split(X, y, test_size=0.1, random_state=42)
        self.images_train = np.concatenate((self.images_train, self.images_train))
        self.labels_train = np.concatenate((self.labels_train, self.labels_train))
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label)
                                        for label in self.unique_train_label}

    def _get_siamese_similar_pair(self):
        label = np.random.choice(self.unique_train_label)
        l = np.random.choice(self.map_train_label_indices[label], replace=False)
        r = np.random.choice(self.map_train_label_indices[label], replace=False)
        return l, r, 1

    def _get_siamese_dissimilar_pair(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        l = np.random.choice(self.map_train_label_indices[label_l])
        r = np.random.choice(self.map_train_label_indices[label_r])
        return l, r, 0

    def _get_siamese_pair(self):
        if np.random.random() < 0.5:
            return self._get_siamese_similar_pair()
        else:
            return self._get_siamese_dissimilar_pair()

    def get_siamese_batch(self, n):
        idxs_left, idxs_right, labels = [], [], []
        for _ in range(n):
            l, r, x = self._get_siamese_pair()
            idxs_left.append(l)
            idxs_right.append(r)
            labels.append(x)
        # return self.images_train[idxs_left, :], self.images_train[idxs_right, :], np.expand_dims(labels, axis=1)

        return self.images_train[idxs_left, :], np.add(self.images_train[idxs_right, :],
                                                       np.random.normal(loc=0, scale=0.075,
                                                                       size=self.images_train[idxs_right, :].shape)), \
               np.expand_dims(labels, axis=1)
