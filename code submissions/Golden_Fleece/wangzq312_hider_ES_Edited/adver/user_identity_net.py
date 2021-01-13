import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, dropout, softmax, batch_norm
from tensorflow.nn import elu, tanh
from .siamese_cnn import feature_model, contrastive_loss, Dataset
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_user_identity_mlp(ori_data, lr=0.001, epoches=500, batch_size=32):
     '''
     Train a feature extractor using MLP. This function is not used in the final version of our hider.
     '''

    no, seq_len, dim = ori_data.shape
    n_hidden = [512, 256, 256, 256]

    train_data = np.reshape(ori_data, (no, seq_len * dim))
    train_label = np.diag(np.ones(no))

    def user_identity_mlp(x):
        # perturb_noise = tf.Variable(initial_value=tf.zeros((1, seq_len * dim)), dtype=tf.float32, trainable=False)
        # x = Add()([x, perturb_noise])
        for hidden_size in n_hidden:
            x = fully_connected(x, hidden_size, activation_fn=elu)
            x = batch_norm(x)
            x = dropout(x, keep_prob=0.8)
        x = fully_connected(x, no, activation_fn=None)
        return x

    # build input, output, loss and training op
    x = tf.placeholder(tf.float32, [None, seq_len * dim])
    y = tf.placeholder(tf.float32, [None, no])
    y_hat = user_identity_mlp(x)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat)
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # base on loss, calculate and apply gradients
    init = tf.global_variables_initializer()
    model_saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)  # initialize variables
        for epoch in range(epoches):
            epoch_loss = 0.0
            batch_steps = int(no/batch_size)
            idx = np.arange(no)
            np.random.shuffle(idx)
            for i in range(batch_steps):
                batch_idx = idx[(batch_steps-1)*batch_size: batch_steps*batch_size]
                batch_x = train_data[batch_idx]
                # batch_x = train_data[batch_idx] + np.random.normal(loc=0, scale=0.02, size=train_data[batch_idx].shape)
                batch_y = train_label[batch_idx]
                _, temp_loss, pred = sess.run([train, loss, y_hat], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += temp_loss/batch_steps
            print("Epoch {}, Loss={}".format(epoch, epoch_loss))

        # Validation on the training set
        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        x, y, pred, acc = sess.run([x,y,y_hat, accuracy], feed_dict={x: train_data, y: train_label})
        print("Accuracy is {}".format(acc))

        model_saver.save(sess, "/opt/hide-and-seek/hiders/wangzq312/res/utils/mlp_model/")  # ES Edited
    return


def train_siamese_cnn(ori_data, lr=0.001, iterations=5000, batch_size=64):

    '''
         Train a feature extractor using CNN using contrastive loss
         Ref: https://github.com/ardiya/siamesenetwork-tensorflow/
    '''


    # Define the siamese network structure
    no, seq_len, dim = ori_data.shape
    dataset = Dataset(ori_data)
    model = feature_model
    placeholder_shape = [None, seq_len, dim, 1]
    next_batch = dataset.get_siamese_batch


    left = tf.placeholder(tf.float32, placeholder_shape, name='left')
    right = tf.placeholder(tf.float32, placeholder_shape, name='right')
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
        label_float = tf.to_float(label)
    left_output, left_noise = model(left, reuse=False)
    right_output, right_noise = model(right, reuse=True)

    margin = 1
    loss, distance = contrastive_loss(left_output, right_output, label_float, margin)

    output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    output_vars = [i for i in output_vars if i.name != "add_noise/noise:0"]

    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=output_vars)  # base on loss, calculate and apply gradients
    init = tf.global_variables_initializer()
    model_saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)  # initialize variables
        for iter in range(iterations): # train the CNN feature extractor in a siamese network way using contrastive loss
            batch_left, batch_right, batch_similarity = next_batch(batch_size) # get a batch of siamese pairs
            _, iter_loss, distances, model1, model2 = sess.run([train, loss, distance,left_output,right_output], feed_dict={left: batch_left, right: batch_right, label: batch_similarity})
            debug_temp = np.sqrt(np.sum(np.power(model1 - model2, 2), 1))
            if (iter+1) % 1000 == 0:
                print("Iteration {}, Loss={}".format(iter, iter_loss))
            # if (iter + 1) % 500 == 0:
            #     feat = sess.run(left_output, feed_dict={left: dataset.images_test})
        model_saver.save(sess, "/opt/hide-and-seek/hiders/wangzq312/res/utils/siamese_model/" + "feature_extractor")  # ES Edited

if __name__ == "__main__":
    ori_data = []
    for i in range(300):
        ori_data.append(i * np.ones((100,71)))
    ori_data = np.array(ori_data)
    train_user_identity_mlp(ori_data, epoches=500)
