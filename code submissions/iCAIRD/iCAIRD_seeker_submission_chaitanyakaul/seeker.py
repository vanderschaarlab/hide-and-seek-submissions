"""
The seeker module containing the `seeker(...)` function.
"""
# pylint: disable=fixme
from typing import Dict
import numpy as np
from utils.data_preprocess import preprocess_data


def seeker(input_dict: Dict) -> np.ndarray:
    """Solution seeker function.

    Args:
        input_dict (Dict): Dictionary that contains the seeker function inputs, as below:
            * "seed" (int): Random seed provided by the competition, use for reproducibility.
            * "generated_data" (np.ndarray of float): Generated dataset from hider data, 
                shape [num_examples, max_seq_len, num_features].
            * "enlarged_data" (np.ndarray of float): Enlarged original dataset, 
                shape [num_examples_enlarge, max_seq_len, num_features].
            * "generated_data_padding_mask" (np.ndarray of bool): Padding mask of bools, generated dataset, 
                same shape as "generated_data".
            * "enlarged_data_padding_mask" (np.ndarray of bool): Padding mask of bools, enlarged dataset, 
                same shape as "enlarged_data".

    Returns:
        np.ndarray: The reidentification labels produced by the seeker, expected shape [num_examples_enlarge].
    """

    # Get the inputs.
    seed = input_dict["seed"]
    generated_data = input_dict["generated_data"]
    enlarged_data = input_dict["enlarged_data"]
    generated_data_padding_mask = input_dict["generated_data_padding_mask"]
    enlarged_data_padding_mask = input_dict["enlarged_data_padding_mask"]

    # Get processed and imputed data, if desired:
    generated_data_preproc, generated_data_imputed = preprocess_data(generated_data, generated_data_padding_mask)
    enlarged_data_preproc, enlarged_data_imputed = preprocess_data(enlarged_data, enlarged_data_padding_mask)

    # TODO: Put your seeker code to replace Example 1 below.
    # Feel free play around with Examples 1 (knn) and 2 (binary_predictor) below.

    # --- Example 1: knn ---
    #from examples.seeker.knn import knn_seeker

    #reidentified_data = knn_seeker.knn_seeker(generated_data_imputed, enlarged_data_imputed)
    #return reidentified_data

    # --- Example 2: binary_predictor ---
    # from utils.misc import tf115_found
    # assert tf115_found is True, "TensorFlow 1.15 not found, which is required to run binary_predictor."
    # from examples.seeker.binary_predictor import binary_predictor
    # reidentified_data = binary_predictor.binary_predictor(generated_data_imputed, enlarged_data_imputed, verbose=True)
    # return generated_data
    
    import os
    import tempfile
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
    from tensorflow.keras.layers import Layer
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


    class Attention(Layer):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, hidden_states):
            """
            Many-to-one attention mechanism for Keras.
            """
            hidden_size = int(hidden_states.shape[2])
            # Inside dense layer
            #              hidden_states            dot               W            =>           score_first_part
            # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
            # W is the trainable weight matrix of attention Luong's multiplicative style score
            score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
            #            score_first_part           dot        last_hidden_state     => attention_weights
            # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
            h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
            score = dot([score_first_part, h_t], [2, 1], name='attention_score')
            attention_weights = Activation('softmax', name='attention_weight')(score)
            # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
            context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
            pre_activation = concatenate([context_vector, h_t], name='attention_output')
            attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
            return attention_vector
    
    # Parameters
    enl_no, seq_len, dim = enlarged_data.shape
    gen_no, _, _ = generated_data.shape
    max_seq_len = len(generated_data[0, :, 0])

    # Set training features and labels
    train_x = np.concatenate((generated_data_imputed.copy(), enlarged_data_imputed.copy()), axis=0)
    train_y = np.concatenate((np.zeros([gen_no, 1]), np.ones([enl_no, 1])), axis=0)

    idx = np.random.permutation(enl_no + gen_no)
    train_x = train_x[idx, :, :]
    train_y = train_y[idx, :]
    
    # Create the model
    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value=-1.0, input_shape=(100,71)))
    model.add(layers.LSTM(dim*2, return_sequences=True))
    model.add(layers.LSTM(2048, return_sequences=True))
    model.add(Attention())
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(train_y.shape[-1], activation="sigmoid"))
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    model.compile(loss=binary_cross_entropy_loss, optimizer=adam)
    
    # Model fitting
    valid_rate=0.1
    idx_train_test_split = np.random.permutation(len(train_x))
    train_idx = idx_train_test_split[: int(len(idx_train_test_split) * (1 - valid_rate))]
    valid_idx = idx_train_test_split[int(len(idx_train_test_split) * (1 - valid_rate)) :]

    train_data, train_label = train_x[train_idx], train_y[train_idx]
    valid_data, valid_label = train_x[valid_idx], train_y[valid_idx]
    
    
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
    
    model.fit(
        train_data,
        train_label,
        batch_size=128,
        epochs=10,
        validation_data=(valid_data, valid_label),
        callbacks=[save_best],
        verbose=1,
        )
    
    model.load_weights(save_file_name)
    
    model.pop()
    
    generated_data_imputed_embedding = model.predict(generated_data_imputed)
    
    enlarged_data_imputed_embedding = model.predict(enlarged_data_imputed)
    
    enl_no_emb, seq_len_emb = enlarged_data_imputed_embedding.shape
    gen_no_emb, _ = generated_data_imputed_embedding.shape
    
    # Output initialization
    distance = np.zeros([enl_no_emb,])

    # For each data point in enlarge dataset
    for i in range(enl_no_emb):
        temp_dist = list()
        # For each generated data point
        for j in range(gen_no_emb):
            # Check the distance between data points in enlarge dataset and generated dataset
            tempo = np.linalg.norm(enlarged_data_imputed_embedding[i, :] - generated_data_imputed_embedding[j, :])
            temp_dist.append(tempo)
        # Find the minimum distance from 1-NN generated data
        distance[i] = np.min(temp_dist)

    # Check the threshold distance for top gen_no for 1-NN distance
    thresh = sorted(distance)[gen_no_emb]

    # Return the decision for reidentified data
    reidentified_data = 1 * (distance <= thresh)

    return reidentified_data
    
    # # Measure the distance from synthetic data using the trained model
    # distance = model.predict(enlarged_data_imputed)
    
    # # Check the threshold distance for top gen_no for 1-NN distance
    # thresh = sorted(distance)[gen_no]
    
    # # Return the decision for reidentified data
    # reidentified_data = 1 * (distance <= thresh)
    
    #reidentification_score = reidentify_score(true_labels, reidentified_data)
    
    #return reidentified_data
    
    # import numpy as np
    # from sklearn import svm
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn. ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn import tree
    
    # #BG = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
    # #RF = RandomForestClassifier(max_depth=2, random_state=0)
    # #NN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(130,), random_state=1)
    # SV = svm.SVC(kernel='linear')
    # #ADA = AdaBoostClassifier()
    # KNN = KNeighborsClassifier()
    # #DTC = tree.DecisionTreeClassifier()
    # #MNB = GaussianNB()
    # GBC = GradientBoostingClassifier()
    
    # #evs = VotingClassifier(estimators=[('BG',BG),('RF',RF),('NN',NN),('SV',SV),('ADA',ADA),('KNN',KNN),('DTC',DTC),('MNB',MNB),('GBC',GBC)],voting='hard')
    # evs = VotingClassifier(estimators=[('SV',SV),('KNN',KNN),('GBC',GBC)],voting='hard')
    
    # enl_no, seq_len, dim = enlarged_data.shape
    # gen_no, _, _ = generated_data.shape

    # # Set training features and labels
    # train_x = np.concatenate((generated_data_imputed.copy(), enlarged_data_imputed.copy()), axis=0)
    # train_y = np.concatenate((np.zeros([gen_no, 1]), np.ones([enl_no, 1])), axis=0)

    # idx = np.random.permutation(enl_no + gen_no)
    # train_x = train_x[idx, :, :]
    # train_y = train_y[idx, :]
    
    # valid_rate=0.0
    # idx_train_test_split = np.random.permutation(len(train_x))
    # train_idx = idx_train_test_split[: int(len(idx_train_test_split) * (1 - valid_rate))]

    # train_data, train_label = train_x[train_idx], train_y[train_idx]
    
    # nsamples, nx, ny = train_data.shape
    # train_data = train_data.reshape((nsamples,nx*ny))
    
    # evs.fit(train_data,train_label)
    
    # reidentified_data = evs.predict(enlarged_data_imputed)
    
    #return reidentified_data
    
    
    