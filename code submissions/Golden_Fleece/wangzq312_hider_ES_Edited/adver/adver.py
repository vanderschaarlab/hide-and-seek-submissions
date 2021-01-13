# Necessary packages
import numpy as np
import tensorflow as tf
import os
from .siamese_cnn import feature_model, contrastive_loss, Dataset
from .util_baselines import feature_pred_model_fit, feature_pred_model_predict
from .util_baselines import one_step_ahead_pred_model_fit, one_step_ahead_pred_model_predict
from .user_identity_net import train_user_identity_mlp, train_siamese_cnn
from tqdm import tqdm
from sklearn.cluster import KMeans

def copy_var_from_ckpt(session, dst_var, ckpt_path):
    '''Copy the trained network weights.
     Reference: http://kmanong.top/kmn/qxw/form/article?id=12337&cate=95

    Args:
        session: the session where the destination graph is in
        dst_var: a list of tensors whose weights need to be copied
        ckpt_path: path to the checkpoint
    '''
    #
    graph = tf.Graph()
    meta_graph_path = ckpt_path + ".meta"
    checkpoint_path = ckpt_path
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_graph_path)
            saver.restore(sess, checkpoint_path)
            v_names = []
            for v in tf.trainable_variables():
                v_names.append(v.name)
            dst_var_name = [i.name for i in dst_var]
            for i in range(len(dst_var_name)):
                if dst_var_name[i] in v_names:
                    tensor = graph.get_tensor_by_name(dst_var_name[i])
                    weight = sess.run(tensor)
                    session.run(dst_var[i].assign(weight))

def adversarial_loss(embedding_output, target_embedding, noise_size, utility_score_error, one_step_score_error):
    '''
    For the noise shaping training, we use a self-defined adversarial loss that has the following parts. Say the original data is x1, noise is delta, the target data from anther use is x2
     - the distance between embedding(x1+delta) and embedding(x2)
     - the scale of the noise added [Regularization term]
     - the error of feature prediction [Optional, not used in final version]
     - the error of one-step-ahead prediction [Optional, not used in final version]
     All the distances and scales are calculated with l2-norm
    '''

    distance = tf.sqrt(tf.reduce_sum(tf.pow(embedding_output - target_embedding, 2), 1))
    alpha = tf.constant(0.18, dtype=tf.float32)
    beta = tf.constant(0, dtype=tf.float32)
    gamma = tf.constant(0, dtype=tf.float32)
    error = tf.add_n([tf.multiply(tf.constant(1.0),distance), tf.multiply(alpha, noise_size),
                      tf.multiply(beta,utility_score_error), tf.multiply(gamma, one_step_score_error)])
    return error



def adver(ori_data):
    """Add Gaussian noise on the original data and use as the synthetic data.

    Args:
        - ori_data: original time-series data [pre-processed and padding applied]
        - noise_size: amplitude of the added noise

    Returns:
        - generated_data: generated synthetic data
    """
    import os

    # ES Edited: -----------------------
    ckpt_path = '/opt/hide-and-seek/hiders/wangzq312/res/utils/siamese_model/'
    path_2 = '/opt/hide-and-seek/hiders/wangzq312/res/utils/mlp_model/'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
        os.mkdir(os.path.join(ckpt_path,'feature_extractor'))
    if not os.path.exists(path_2):
        os.mkdir(path_2)
        os.mkdir(os.path.join(path_2,'feature_extractor'))
    # ----------------------------------

    # Parameters
    no, seq_len, dim = ori_data.shape
    generated_data = []
    ##################################################
    # Step 1: Train a CNN-based feature extractor using a siamese way.
    # The feature extractor works to recognize the user and maximize the
    # embedding distances between different users
    ##################################################
    # # train_user_identity_mlp(ori_data)
    tf.reset_default_graph()
    batch_size = 64
    # iterations = int(no * (no - 1) / batch_size)
    # iterations = 100
    iterations = 30000 # 20000 for final version
    train_siamese_cnn(ori_data, lr=0.001, iterations=iterations, batch_size=batch_size)  #
    ##################################################



    ####################################################
    # Step2: Define a CNN with exactly the same structure, except for that in
    # the input layer we add a noise layer of the same dimension. Then we copy
    # the trained CNN weights and fix them. The only learnable thing is the noise.
    ####################################################

    # tf.reset_default_graph()
    final_graph = tf.Graph()
    with final_graph.as_default() as g:
        model = feature_model
        placeholder_shape = [None, seq_len, dim, 1]

        raw_data = tf.placeholder(tf.float32, placeholder_shape, name='raw_data')
        goal_of_embedding = tf.placeholder(tf.float32, 128, name='goal_of_embedding')
        noise_size = tf.placeholder(tf.float32, 1, name='noise_size')
        utility_score_error = tf.placeholder(tf.float32, 1, name='utility_score_error')
        one_step_score_error = tf.placeholder(tf.float32, 1, name='one_step_score_error')
        embedding_output, perturb_noise = model(raw_data, reuse=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        adver_session = tf.Session(config=config, graph=final_graph)

        lr = 0.0001
        loss = adversarial_loss(embedding_output, goal_of_embedding, noise_size, utility_score_error, one_step_score_error)
        output_var = tf.trainable_variables()
        output_var = [output_var[0]]
        assert output_var[0].name == "add_noise/noise:0"  # only the noise is learnable
        grads = tf.gradients(loss, output_var)
        train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=output_var)
        init = tf.global_variables_initializer()

        adver_session.run(init)
        dst_var = tf.trainable_variables()
        ckpt_path = '/opt/hide-and-seek/hiders/wangzq312/res/utils/siamese_model/feature_extractor'  # ES Edited

    # load parameters from pre-trained model
    copy_var_from_ckpt(session=adver_session, dst_var=dst_var, ckpt_path=ckpt_path) # copy the CNN weights
    feature_embeddings = []
    batch_size = 64
    for i in range(0, (int(no/batch_size)+1)):
        temp_embeddings = adver_session.run([embedding_output], feed_dict={raw_data: np.array(ori_data[i*64:(i+1)*64,:,:,np.newaxis])})
        feature_embeddings.extend(np.squeeze(temp_embeddings))


    # feature_embeddings = adver_session.run([embedding_output], feed_dict={raw_data: np.array(ori_data[:,:,:,np.newaxis])}) #OOM risk


    # ##################################################
    # # Step 0: Train the util models (feature predictor, one step ahead) with ori_data
    # ##################################################
    # # step1 start here
    # num_features = dim
    # feature_prediction_no = 3 # 15 for release
    # debug_mode = True  # False when actually do generation!
    # feature_idx = np.random.permutation(range(1, num_features))[: feature_prediction_no]
    # print(f"\nFeature prediction model on IDs: {feature_idx}\n")
    #
    # feature_pred_models = feature_pred_model_fit(
    #     train_data=ori_data,
    #     index=feature_idx,
    #     debug=debug_mode
    # )
    #
    # # feature_ref_scores = feature_pred_model_predict(
    # #             models=feature_pred_models,
    # #             test_data=ori_data,
    # #             index=feature_idx)
    #
    # feature_individual_scores = []
    # # for i in tqdm(range(no)):
    # #     feature_individual_scores.append(feature_pred_model_predict(
    # #         models=feature_pred_models,
    # #         test_data=ori_data[i],
    # #         index=feature_idx
    # #     ))
    # feature_individual_scores = feature_pred_model_predict(
    #         models=feature_pred_models,
    #         test_data=ori_data,
    #         index=feature_idx
    #     )
    #
    # feature_ref_scores = np.mean(np.array(feature_individual_scores), axis=0)
    #
    # one_step_ahead_model = one_step_ahead_pred_model_fit(
    #     train_data=ori_data,
    #     debug=debug_mode
    # )
    #
    # # one_step_ref_score = one_step_ahead_pred_model_predict(
    # #         models=one_step_ahead_model,
    # #         test_data=ori_data
    # #     )
    # one_step_individual_score = []
    # for i in tqdm(range(no)):
    #     one_step_individual_score.append(one_step_ahead_pred_model_predict(
    #         models=one_step_ahead_model,
    #         test_data=ori_data[i]
    #     ))
    # one_step_ref_score = np.mean(one_step_individual_score)
    #
    # print("Finished Utility Score Model Training")
    ##################################################
    # step0 ends here
    ##################################################

    # # with open('test.npy', 'wb') as f:
    # #     np.save(f, np.array([feature_individual_scores, feature_ref_scores,
    # #                          one_step_individual_score, one_step_ref_score]))






    ##################################################
    # Step 3: Noise Training and Adding

    # For each user (data entry x1), select another user (x2) whose
    # embedding is very different. Then we train the noise to move the current
    # embedding(x1+noise) towards embedding(x2). Then retrun (x1+noise_opt) as
    # the generated data. Repeat for all the users.
    #
    # Some details:
    # The training loss for the learnable noise is the adversarial loss
    # To reduce the calculations, we group the users based on the embedding and
    # use the user closest to the centroid of each cluster as the representative.
    # We use the representative data to train the noise and apply the noise
    # on other data entries in the same cluster

    ##################################################
    feature_embeddings = np.squeeze(np.array(feature_embeddings)) # Use the CNN to calculate the embeddings of each data entry.

    # Cluster all users (data entries) based on their data using KMeans to
    # create c clusters
    n_clusters = int(no/25.0)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_model.fit(feature_embeddings)
    labels = cluster_model.labels_

    grouped_data = []
    # grouped_idx = []
    for i in range(n_clusters):
        # register which data entry belongs to which cluster
        grouped_data.append([ori_data[j] for j in range(len(labels)) if labels[j] == i])
        # grouped_idx.append([j for j in range(len(labels)) if labels[j] == i])

    for i in tqdm(range(n_clusters)):
        with final_graph.as_default() as g:
            # calculate the embeddings of each data entry in the cluster.
            adver_session.run(output_var[0].assign(tf.zeros((100, 71, 1))))

        # find the sample closest to its cluster centroid
        cluster_data = grouped_data[i]
        # cluster_idx = grouped_idx[i]
        mean_data = np.mean(cluster_data, axis=0)
        elected = np.argmin(np.array([np.linalg.norm(j) for j in (cluster_data-mean_data)]))
        try:
            elected = elected[0]
        except:
            pass

        feed_in_data = cluster_data[elected]  # find the user whose embeddings is the closest to the embeddings of the cluster centroid
        # centroid_idx = cluster_idx[elected]
        # feature_ref_scores = feature_individual_scores[elected]  # subject to remove
        # one_step_ref_score = one_step_individual_score[elected]

        # for debug
        self_embedding, noise_added = adver_session.run([embedding_output, perturb_noise],
                                                         feed_dict={raw_data: feed_in_data[np.newaxis, :, :, np.newaxis]})
        # batchnorm will cause problem when making prediction on a single data
        # When I use predict or evaluate with different batch_sizes I get different results.
        # Sol: use layer_norm instead



        # Select m=30 users from original dataset D randomly. Calculate their embeddings
        temp = [j for j in range(len(labels)) if labels[j] != i]
        idx = np.random.permutation(np.array(temp))
        idx = idx[0:30]
        target_embedding = []
        for each in idx:
            target_data = ori_data[each]
            temp_embedding = adver_session.run([embedding_output],
                                                  feed_dict={raw_data: target_data[np.newaxis, :, :, np.newaxis]})
            target_embedding.append(temp_embedding)

        # Find the user whose embedding is most different from that of feed_in_data
        elected = np.argmax(np.array([np.linalg.norm(j) for j in (target_embedding - self_embedding)]))
        try:
            elected = elected[0]
        except:
            pass
        other_embedding = np.squeeze(np.array(target_embedding[elected])) # The embedding of the selected use is named “other_embedding”

        with final_graph.as_default() as g: # Initialize the added noise with a small value
            temp = ori_data[idx[elected]]
            noise_added = temp
            adver_session.run(output_var[0].assign(0.0075*temp[:,:,np.newaxis]))

        # feed_in_data = np.add(feed_in_data, 0.01 * ori_data[idx[elected]]) # ???

        for repeat in range(200):  # 120 for release
          # Loop 200 iterations:
              # (i)Calculate embeddings(feed_in_data+learnable_noise)
              # (ii)Calculate adversarial loss, which is mainly based on
              #     distance(embeddings(feed_in_data+learnable_noise), other_embedding)
              # (iii)Update the noise


            noise_added = np.squeeze(noise_added)

            # utility_error = feature_pred_model_predict(
            #     models=feature_pred_models,
            #     test_data=np.add(feed_in_data, noise_added),
            #     index=feature_idx
            #     )
            #
            # utility_error = np.sum(np.array([s for s in (utility_error-feature_ref_scores) if s>0]))
            #
            # one_step_error = one_step_ahead_pred_model_predict(
            #     models=one_step_ahead_model,
            #     test_data=np.add(feed_in_data, noise_added)
            # )
            #
            # one_step_error = max(0, one_step_error - one_step_ref_score)

            utility_error = 0
            one_step_error = 0
            noise_level = np.linalg.norm(np.reshape(noise_added, (-1, 1)))

            with final_graph.as_default() as g:
                self_embedding, noise_added, gradients, _, l = adver_session.run([embedding_output, perturb_noise, grads, train, loss],
                                                            feed_dict={
                                                            raw_data: feed_in_data[np.newaxis, :, :, np.newaxis],
                                                            goal_of_embedding: other_embedding,
                                                            noise_size: np.array([noise_level]),
                                                            utility_score_error: np.array([utility_error]),
                                                            one_step_score_error: np.array([one_step_error])})

            # print("Curr Loss is {}".format(l)) # For debugging
        # one cluster is done, generated_data = ori_data + learned_noise
        noise_added = np.squeeze(noise_added)
        for each in cluster_data:
            generated_data.append(np.add(each, noise_added))

    adver_session.close()
    return np.array(generated_data)
