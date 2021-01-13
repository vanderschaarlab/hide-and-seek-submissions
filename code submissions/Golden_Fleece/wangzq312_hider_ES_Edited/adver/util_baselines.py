'''
   Most functions in this file are copied or adapted from the given one_step_ahead
   or feature_pred models. We warpped them with self-defined API so its easier to
   use.

   In the final version of the hider, we did not use them.
'''


import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Determine if TF 1.15 is present.
tf115_found = False
try:
    import tensorflow as tf

    if tf.__version__[:4] == "1.15":  # pylint: disable=no-member
        tf115_found = True
except ModuleNotFoundError:
    pass

# Resolve general_rnn module.
if tf115_found:
    try:
        from general_rnn import GeneralRNN
    except ModuleNotFoundError:
        try:
            from utils.general_rnn import GeneralRNN  # type: ignore
        except ModuleNotFoundError:
            from .general_rnn import GeneralRNN  # type: ignore  # pylint: disable=relative-beyond-top-level


def rmse_error(y_true, y_pred):
    """User defined root mean squared error.

    Args:
        - y_true: true labels
        - y_pred: predictions

    Returns:
        - computed_rmse: computed rmse loss
    """
    # Exclude masked labels
    idx = (y_true >= 0) * 1
    # Mean squared loss excluding masked labels
    computed_mse = np.sum(idx * ((y_true - y_pred) ** 2)) / np.sum(idx)
    computed_rmse = np.sqrt(computed_mse)
    return computed_rmse


def feature_pred_model_fit(train_data, index, verbose=False, debug=False):
    """Use the other features to predict a certain feature.
    """

    if not tf115_found:
        raise ModuleNotFoundError("TF 1.15 is required for running this function but was not found.")

    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters
    model_parameters = {
        "task": "regression",
        "model_type": "gru",
        "h_dim": dim,
        "n_layer": 3,
        "batch_size": 128,
        "epoch": 20 if not debug else 1,
        "learning_rate": 0.001,
    }

    # Output initialization
    models = []

    # For each index
    for idx in index:
        # Set training features and labels
        train_x = np.concatenate((train_data[:, :, :idx], train_data[:, :, (idx + 1) :]), axis=2)
        train_y = np.reshape(train_data[:, :, idx], [no, seq_len, 1])

        # Train the predictive model
        if len(np.unique(train_y)) == 2:
            model_parameters["task"] = "classification"
        general_rnn = GeneralRNN(model_parameters)
        general_rnn.fit(train_x, train_y, verbose=verbose)

        models.append(general_rnn)
    return models


def feature_pred_model_predict(models, test_data, index, verbose=False, debug=False):
    """Use the other features to predict a certain feature.
    """

    if not tf115_found:
        raise ModuleNotFoundError("TF 1.15 is required for running this function but was not found.")

    # Parameters
    if len(test_data.shape) == 2:
        test_data = test_data[np.newaxis, :, :]
    assert len(test_data.shape) == 3
    no, seq_len, dim = test_data.shape

    # Set model parameters
    model_parameters = {
        "task": "regression",
        "model_type": "gru",
        "h_dim": dim,
        "n_layer": 3,
        "batch_size": 128,
        "epoch": 20 if not debug else 1,
        "learning_rate": 0.001,
    }

    # Output initialization
    scores = [[] for i in range(len(index))]

    # For each index
    for i in range(len(index)):
        idx = index[i]
        # Set training features and labels
        test_x = np.concatenate((test_data[:, :, :idx], test_data[:, :, (idx + 1):]), axis=2)
        test_y = np.reshape(test_data[:, :, idx], [no, seq_len, 1])

        # Train the predictive model
        # if len(np.unique(test_y)) == 2:
        #     model_parameters["task"] = "classification"
        general_rnn = models[i]
        test_y_hat = general_rnn.predict(test_x)

        # Evaluate the trained model
        # test_y = np.reshape(test_y, [-1])
        # test_y_hat = np.reshape(test_y_hat, [-1])

        # if model_parameters["task"] == "classification":
        #     temp_perf = roc_auc_score(test_y, test_y_hat)
        if model_parameters["task"] == "regression":
            temp_perf = []
            for j in range(len(test_y)):
                temp_perf.append(rmse_error(np.squeeze(test_y[j]), np.squeeze(test_y_hat[j])))

        scores[i] = temp_perf
    scores = np.transpose(scores)
    if scores.shape[0] == 1:
        scores = np.squeeze(scores)
    return scores


def one_step_ahead_pred_model_fit(train_data, verbose=False, debug=False):
    """Use the previous time-series to predict one-step ahead feature values.
    """

    if not tf115_found:
        raise ModuleNotFoundError("TF 1.15 is required for running this function but was not found.")

    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters
    model_parameters = {
        "task": "regression",
        "model_type": "gru",
        "h_dim": dim,
        "n_layer": 3,
        "batch_size": 128,
        "epoch": 20 if not debug else 1,
        "learning_rate": 0.001,
    }

    # Set training features and labels
    train_x = train_data[:, :-1, :]
    train_y = train_data[:, 1:, :]

    # Train the predictive model
    if len(np.unique(train_y)) == 2:
        model_parameters["task"] = "classification"

    general_rnn = GeneralRNN(model_parameters)
    general_rnn.fit(train_x, train_y, verbose=verbose)

    return general_rnn


def one_step_ahead_pred_model_predict(models, test_data, verbose=False, debug=False):
    """Use the previous time-series to predict one-step ahead feature values.
    """

    if not tf115_found:
        raise ModuleNotFoundError("TF 1.15 is required for running this function but was not found.")

    # Parameters
    if len(test_data.shape) == 2:
        test_data = test_data[np.newaxis, :, :]
    assert len(test_data.shape) == 3
    no, seq_len, dim = test_data.shape

    # Set model parameters
    model_parameters = {
        "task": "regression",
        "model_type": "gru",
        "h_dim": dim,
        "n_layer": 3,
        "batch_size": 128,
        "epoch": 20 if not debug else 1,
        "learning_rate": 0.001,
    }

    # Set testing features and labels
    test_x = test_data[:, :-1, :]
    test_y = test_data[:, 1:, :]

    # Train the predictive model
    if len(np.unique(test_y)) == 2:
        model_parameters["task"] = "classification"

    general_rnn = models
    test_y_hat = general_rnn.predict(test_x)

    # Evaluate the trained model
    test_y = np.reshape(test_y, [-1])
    test_y_hat = np.reshape(test_y_hat, [-1])

    if model_parameters["task"] == "classification":
        perf = roc_auc_score(test_y, test_y_hat)
    elif model_parameters["task"] == "regression":
        perf = rmse_error(test_y, test_y_hat)

    return perf
