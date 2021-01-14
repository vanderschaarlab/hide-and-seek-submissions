""" Module containing classes for generation of values for sparse features.

The data contains features that are so sparse that modelling them as a
time series in a dynamical model is not possible. We treat them differently
by trying to predict them based on a set of generated dense features.
"""


import torch
import torch.nn as nn
from typing import List, Optional, Callable, Union, Tuple, Any, Dict
import numpy as np
import logging
from data_pipeline_transformers import DataTransformer, MSG_PADDING_MASK, StandardScaler

def get_default_device():
    dev = torch.device('cpu')
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    return dev

DO_PIN_MEMORY=False


class SimplerNetwork(nn.Module):
    """ NOT USED """

    def __init__(
            self,
            num_predictors: int,
            num_predictees: int,
            window_size: int,
            pre_conv_hidden: List[int]=None,
            post_conv_hidden: List[int]=None,
            nonlinearity=nn.ReLU):
        super().__init__()
        self.window_size = window_size
        self.num_predictors = num_predictors
        self.layers = nn.Sequential(
            nn.Linear(window_size*num_predictors, num_predictors), nonlinearity(),
            nn.Linear(num_predictors, num_predictors), nonlinearity(),
            nn.Linear(num_predictors, num_predictees)
        )

    def forward(self, x):
        z = x.reshape(-1, self.window_size * self.num_predictors)
        return self.layers(z)

class PredictionNetwork(nn.Module):
    """ Network for predicting sparse features.

    Predicts the sparse feature values from a sliding window of observed common values,
    i.e., $y_t = f(x_{t-w/2}, ..., x_t, ..., x_{t+w/2})$.

    The network consists of up to three stages:
        1. (Optional) Linear layers applied independently to each predictors time step
        (number of layers and layer sizes are defined in `pre_conv_hidden`).
        2. Layer that convolves windows of predictors.
        3. (Optional) Linear layers applied to the resulting hidden state time step
        (number of layers and layer sizes are defined in `post_conv_hidden`).
        4. Linear output layer producing values for predictees for each time step at a
        window center.

    The network is set up as to allow feeding only relevant windows during
    training but an entire sequence during prediction. This is to reduce
    computational overhead during training since at many locations values
    for predictees will be absent.

    Due to the above, this module does not perform any padding on the sequence.
    Callers must make sure to add padding (if required) to the data prior
    to invoking the module. Callers must also ensure to impute any missing
    values in the predictors.
    """

    def __init__(self,
            num_predictors: int,
            num_predictees: int,
            window_size: int=5,
            pre_conv_hidden: List[int]=None,
            post_conv_hidden: List[int]=None,
            nonlinearity=nn.ReLU):
        """
        Args:
            num_predictors: Number of common features to use as predictors.
            num_predictees: Number of sparse features to predict.
            window_size: The window size of time steps in the predictor sequence that
                is considered for predicting an output time step.
            pre_conv_hidden: Sizes for hidden unbiased linear layers before convolving the window.
            post_conv_hidden: Sizes for hidden linear layers after convolving the window.
            nonlinearity: The nonlinearity to use after every hidden layer.
        """
        super().__init__()

        self.window_size = window_size

        pre_conv_hidden = [] if pre_conv_hidden is None else pre_conv_hidden

        pre_conv_hidden = [num_predictors] + pre_conv_hidden

        post_conv_hidden = [pre_conv_hidden[-1]] if post_conv_hidden is None else post_conv_hidden

        self.pre_conv_linear = nn.Sequential(
            *(nn.Sequential(nn.Linear(h_in, h_out, bias=False), nonlinearity())
            for h_in, h_out in zip(pre_conv_hidden[:-1], pre_conv_hidden[1:]))
        )

        self.conv = nn.Sequential(
            nn.BatchNorm1d(pre_conv_hidden[-1]),
            nn.Conv1d(pre_conv_hidden[-1], post_conv_hidden[0], kernel_size=window_size, stride=1, padding=0),
            nonlinearity(),
            nn.BatchNorm1d(post_conv_hidden[0])
        )

        self.post_conv_linear = nn.Sequential(
            *(nn.Sequential(nn.Linear(h_in, h_out) , nonlinearity())
            for h_in, h_out in zip(post_conv_hidden[:-1], post_conv_hidden[1:]))
        )

        self.final_layer = nn.Linear(post_conv_hidden[-1], num_predictees)


    def forward(self, x):
        """
        Args:
            x (tf.Tensor): Input batch of shape (batch_size, seq_length, num_feat_common)
        Returns:
            (tf.Tensor): Predicted sparse values of shape (batch_size, seq_length-(window_size-1), num_feat_sparse)
        """
        z = x
        z = self.pre_conv_linear(z)

        z = z.transpose(-1, -2)
        z = self.conv(z)
        z = nn.functional.relu(z)
        z = z.transpose(-1, -2)

        z = self.post_conv_linear(z)
        y = self.final_layer(z)
        return torch.squeeze(y, dim=1)

class SparseFeatureTrainingDataSet():
    """ Isolates sites with at least one sparse feature value and the corresponding
    window of predictors for training the prediction network.

    Constructs torch.DataLoader instances for training the sparse feature predictor network.
    """

    def __init__(self, data, padding_mask, predictor_feature_idxs, sparse_feature_idxs, window_size):
        """
        Args:
            data: np.array containing train data; shape (num_data, max_seq_len, num_features).
            padding_mask: boolean np.array of shape (num_data, max_seq_len), with True indicating padding position.
            predictor_feature_idxs: List or np.array containing feature ids of predictors.
            sparse_feature_idxs: List or np.array containing feature ids of sparse features (predictees/regressors).
            window_size: The size of the window of predictors corresponding to a sparse feature site.
        """
        window_half = window_size // 2

        #### we first need to treat the data to extract sites where there are values
        #### for sparse features and the corresponding windows in the predictors

        data = np.where(np.expand_dims(padding_mask, -1), np.nan, data)
        # treating padding as nan will cause it to be filtered away in the next steps

        target_slice = data[:,:,sparse_feature_idxs]
        predictor_slice = data[:,:,predictor_feature_idxs]

        self.target_scaler = StandardScaler()
        target_slice, _ = self.target_scaler.preprocess(np.copy(target_slice), {})
        self.predictor_scaler = StandardScaler()
        predictor_slice, _ = self.predictor_scaler.preprocess(np.copy(predictor_slice), {})

        logging.debug('predictor means {},\n\tstds {}'.format(np.nanmean(predictor_slice, axis=(0,1)), np.nanstd(predictor_slice, axis=(0,1))))
        logging.debug('target means {},\n\tstds {}'.format(np.nanmean(target_slice, axis=(0,1)), np.nanstd(target_slice, axis=(0,1))))

        # pad the time sequences for predictors so that we get full windows at the start and end
        num_data, max_seq_len, num_predictors = predictor_slice.shape
        pad_len = (window_size - 1) // 2 # how much to pad on each side of the sequence for extracting prediction windows, unrelated to the competition padding
        predictor_slice = np.pad(predictor_slice, ((0,0), (pad_len, pad_len), (0,0)))

        # identify sites where at least one sparse feature is present
        target_slice_mask = ~np.isnan(target_slice)
        usable_target_time_step_mask = np.any(target_slice_mask, axis=-1) # array (num_data, max_seq_len) indicating time steps with at least one target value
        usable_target_time_step_idxs = np.where(usable_target_time_step_mask)

        # assemble a data set of sparse feature targets and corresponding windows
        num_targets = len(usable_target_time_step_idxs[0])
        targets = np.zeros(shape=(num_targets, target_slice.shape[-1]))
        predictors = np.zeros(shape=(num_targets, window_size, num_predictors))
        for i, (n, t) in enumerate(zip(*usable_target_time_step_idxs)):
            targets[i] = target_slice[n, t]
            predictors[i] = predictor_slice[n, t:t+window_size]


        # remove nans from data (replace with 0 but remember locations of nans in targets to filter in loss)
        target_masks = ~np.isnan(targets)
        targets[np.isnan(targets)] = 0.
        predictors[np.isnan(predictors)] = 0.

        self.target_masks = target_masks
        self.targets = targets
        self.predictors = predictors

        logging.info("identified {} targets for sparse feature training".format(num_targets))

    def get_data_loaders(self, train_ratio: float, batch_size: int, device: torch.device=None):
        if device is None:
            device = get_default_device()

        # hand over to pytorch
        target_masks = torch.tensor(self.target_masks, device=device, dtype=torch.float32)
        targets = torch.tensor(self.targets, device=device, dtype=torch.float32)
        predictors = torch.tensor(self.predictors, device=device, dtype=torch.float32)

        # get random indices for train/test split
        num_targets = len(self.targets)
        permutation = np.random.permutation(np.arange(num_targets))
        train_idx = permutation[:int(train_ratio * num_targets)]
        test_idx = permutation[-int(train_ratio * num_targets):]

        # select train data
        targets_train = targets[train_idx]
        target_masks_train = target_masks[train_idx]
        predictors_train = predictors[train_idx]

        # select test data
        targets_test = targets[test_idx]
        target_masks_test = target_masks[test_idx]
        predictors_test = predictors[test_idx]

        # set up pytorch data loaders for minibatch generation
        train_loader = torch.utils.data.DataLoader(
            list(zip(predictors_train, targets_train, target_masks_train)),
            shuffle=True, batch_size=batch_size, pin_memory=DO_PIN_MEMORY, drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            list(zip(predictors_test, targets_test, target_masks_test)),
            shuffle=True, batch_size=batch_size, pin_memory=DO_PIN_MEMORY, drop_last=True
        )
        return train_loader, test_loader

class SparseFeaturePredictionDataSet():
    """ Prepares data streams for prediction of sparse feature values.

    Constructs a torch.DataLoader instance for prediction in minibatches.
    """

    def __init__(self, data, padding_mask, predictor_feature_idxs, window_size, predictor_scaler):
        window_half = window_size // 2

        # isolate predictors
        predictor_slice = np.where(
            np.expand_dims(padding_mask, -1),
            np.nan, data[:,:,predictor_feature_idxs]
        )

        predictor_slice, _ = predictor_scaler.preprocess(np.copy(predictor_slice), {})
        logging.debug('predictor means {},\n\tstds {}'.format(np.nanmean(predictor_slice, axis=(0,1)), np.nanstd(predictor_slice, axis=(0,1))))

        # pad the time sequences for predictors so that we get full windows at the start and end
        num_data, max_seq_len, num_predictors = predictor_slice.shape
        pad_len = (window_size - 1) // 2 # how much to pad on each side of the sequence for extracting prediction windows, unrelated to the competition padding
        predictors = np.pad(predictor_slice, ((0,0), (pad_len, pad_len), (0,0)))

        predictors[~np.isfinite(predictors)] = 0. # replace nan's with 0

        self.predictors = predictors

    def get_data_loader(self, batch_size: int, device: torch.device=None):
        if device is None:
            device = get_default_device()

        # hand over to pytorch
        predictors = torch.tensor(self.predictors, device=device, dtype=torch.float32)

        # set up pytorch data loaders for minibatch generation
        data_loader = torch.utils.data.DataLoader(
            predictors,
            shuffle=False, batch_size=batch_size, pin_memory=DO_PIN_MEMORY, drop_last=False
        )

        return data_loader

def train_sparse_feature_predictor(
        data: np.ndarray,
        padding_mask: np.ndarray,
        predictor_feature_idxs: List[int],
        sparse_feature_idxs: List[int],
        predictor: PredictionNetwork,
        batch_size: int=32,
        conv_crit_threshold: float=1e-5,
        conv_crit_len: int=3,
        conv_crit_window: int=10,
        max_epochs: int=100,
        training_device: Optional[torch.device]=None):
    """
    Trains with minibatch size batch_size until average loss on test over last conv_crit_window
    episodes changes by less than conv_crit_threshold for conv_crit_len epochs, but for a maximum of
    max_epochs epochs.

    Args:
        data: np.array containing train data; shape (num_data, max_seq_len, num_features).
        padding_mask: boolean np.array of shape (num_data, max_seq_len), with True indicating padding position.
        predictor_feature_idxs: List or np.array containing feature ids of predictors.
        sparse_feature_idxs: List or np.array containing feature ids of sparse features (predictees/regressors).
        predictor: Instance of PredictionNetwork.
        batch_size: Size of minibatches in training.
        conv_crit_threshold: Treshold below which we consider deviations in averaged loss over episodes to be converging.
        conv_crit_len: Number of consecutive epochs of average loss below threshold until we consider the model converged.
        conv_crit_window: Length of window for averaging losses for convergence criterium.
        max_epochs: Maximum number of iterations over the whole data.
        training_device: torch.device on which the training will take place.
    """

    window_size = predictor.window_size

    if training_device is None:
        training_device = get_default_device()

    # create SparseFeatureDataSet which extracts windows for training
    dataset = SparseFeatureTrainingDataSet(data, padding_mask, predictor_feature_idxs, sparse_feature_idxs, window_size)
    train_loader, test_loader = dataset.get_data_loaders(train_ratio=0.98, batch_size=batch_size, device=training_device)
    predictor.to(training_device)

    # train the predictor
    lr = 1e-3
    optimizer = torch.optim.Adam(predictor.parameters(), lr)

    e = 0
    recent_avg_test_losses = np.ones(conv_crit_len + 1) * np.inf
    recent_test_losses = np.ones(conv_crit_window) * np.nan
    while (not np.all((np.abs(np.diff(recent_avg_test_losses)) <= conv_crit_threshold))
            and e < max_epochs):
        predictor.train()
        train_loss = 0.
        num_train = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y, y_mask = batch

            y_hat = predictor(x)
            y_hat_masked = y_hat*y_mask
            delta = y_hat_masked - y
            loss = torch.mean(delta**2)

            loss.backward()
            train_loss += loss.detach()
            num_train += 1
            optimizer.step()

        predictor.eval()
        eval_loss = 0.
        num_eval = 0
        for batch in test_loader:
            x, y, y_mask = batch
            y_hat = predictor(x)
            loss = torch.mean((y_hat*y_mask - y)**2)
            eval_loss += loss.detach()
            num_eval += 1

        recent_test_losses[:-1] = recent_test_losses[1:]
        recent_test_losses[-1] = eval_loss/num_eval
        recent_avg_test_losses[:-1] = recent_avg_test_losses[1:]
        recent_avg_test_losses[-1] = np.nanmean(recent_test_losses)

        logging.info("Sparse feature predictor training: Epoch {}, loss = {:.5f} ({} epoch avg.: {:.5f}) (train loss = {:.5f})".format(
            e, recent_test_losses[-1], conv_crit_window, recent_avg_test_losses[-1], train_loss/num_train)
        )
        e += 1

    return predictor, dataset.predictor_scaler, dataset.target_scaler

def calculate_avg_feature_density(data, padding_mask):
    data = np.where(np.expand_dims(padding_mask, -1), np.nan, data)
    data_count = np.sum(~np.isnan(data), axis=1)

    data_freq = data_count / np.expand_dims(np.sum(~padding_mask, axis=1), -1)
    avg_feature_freq = np.mean(data_freq, axis=0)
    return avg_feature_freq

class SparseFeaturePredictionTransformer(DataTransformer):
    """ Removes sparse features for training the dynamic models during preprocessing and
    generates them based on a simple NN prediction model from generated dense features during
    postprocessing.

    Features are consideres as sparse if their average frequency of occurence per
    sequence is less then parameter `sparsity_treshold`.

    During preprocessing, fits a neural network to predict sparse feature observation
    from denser features. The network considers dense features in a window around
    the time step of the sparse feature as predictors. During postprocessing,
    the network is used to generate values for the sparse features based on
    the generated values in the dense features.
    """

    def __init__(self,
            prediction_network_creator: Optional[Callable[[int, int, int], nn.Module]]=PredictionNetwork,
            sparsity_threshold: Optional[float]=0.05,
            density_threshold: Optional[float]=0.2,
            window_size: Optional[int]=5,
            batch_size: Optional[int]=32,
            conv_crit_threshold: Optional[float]=1e-5,
            conv_crit_len: Optional[int]=5,
            conv_crit_window: Optional[int]=10,
            max_epochs: Optional[int]=100
        ):
        """
        Args:
            prediction_network_creator: A callable with arguments num_predictors, num_predictees, window_size that returns a prediction network.
            sparsity_threshold: Average frequency below which features will be considered sparse and thus predicted by this transformer.
            density_threshold: Average frequency above which features will be considered dense and thus used as predictors.
            window_size: The window size to consider for prediction a single site of sparse features.
            batch_size: Size of minibatches in training.
            conv_crit_threshold: Treshold below which we consider deviations in averaged loss over episodes to be converging.
            conv_crit_len: Number of consecutive epochs of average loss below threshold until we consider the model converged.
            conv_crit_window: Length of window for averaging losses for convergence criterium.
            max_epochs: Maximum number of iterations over the whole data.
        """
        super().__init__()
        self._prediction_network_factory = prediction_network_creator
        self._sparsity_treshold = sparsity_threshold
        self._density_threshold = density_threshold
        if self._density_threshold is None:
            self._density_threshold = self._sparsity_treshold
        self._window_size = window_size
        self._batch_size = batch_size
        self._conv_crit_threshold = conv_crit_threshold
        self._conv_crit_len = conv_crit_len
        self._conv_crit_window = conv_crit_window
        self._max_epochs = max_epochs


    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(padded_sequences.shape[:-1], dtype=bool)

        # identify sparse features to predict and dense feature to predict from according to the thresholds
        avg_feature_density = calculate_avg_feature_density(padded_sequences, padding_mask)
        sparse_feature_idxs = np.where(avg_feature_density < self._sparsity_treshold)[0]
        non_sparse_feature_idxs = [idx for idx in range(padded_sequences.shape[-1]) if idx not in sparse_feature_idxs]
        dense_feature_idxs = np.where(avg_feature_density >= self._density_threshold)[0]
        logging.info('identified {} sparse (with avg. frequency < {} per sequence) and {} dense features (avg. frequency >= {})'.format(
            len(sparse_feature_idxs), self._sparsity_treshold, len(dense_feature_idxs), self._density_threshold
        ))
        logging.debug('sparse feature ids: {}'.format(sparse_feature_idxs))
        logging.debug('dense feature ids: {}'.format(dense_feature_idxs))

        messages['sparse_features'] = sparse_feature_idxs
        messages['non_sparse_features'] = non_sparse_feature_idxs
        messages['dense_features'] = dense_feature_idxs

        # create prediction network
        prediction_network = self._prediction_network_factory(
            len(dense_feature_idxs), len(sparse_feature_idxs), self._window_size
        )
        logging.debug(prediction_network)

        training_device = get_default_device()

        # train prediction network
        prediction_network, train_scaler, test_scaler = train_sparse_feature_predictor(
            padded_sequences, padding_mask, dense_feature_idxs, sparse_feature_idxs, prediction_network,
            batch_size=self._batch_size,
            conv_crit_threshold=self._conv_crit_threshold,
            conv_crit_len=self._conv_crit_len,
            conv_crit_window=self._conv_crit_window,
            max_epochs=self._max_epochs,
            training_device=training_device
        )

        messages['sparse_feature_prediction_network'] = prediction_network
        messages['sparse_feature_prediction_scalers'] = {'predictors': train_scaler, 'sparse_features': test_scaler}

        # filter out sparse features for the remainder of the pipeline
        new_data = padded_sequences[:,:,non_sparse_feature_idxs]

        return new_data, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        sparse_feature_idxs = messages['sparse_features']
        non_sparse_feature_idxs = messages['non_sparse_features'] # gives the original feature indices in padded_sequences
        dense_feature_idxs = messages['dense_features']
        scalers = messages['sparse_feature_prediction_scalers']

        num_sequences, max_seq_len, num_features_current = padded_sequences.shape
        assert(num_features_current == len(non_sparse_feature_idxs))

        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros((num_sequences, max_seq_len), dtype=bool)

        # prepare data with predicted sparse features; shape and indices as original data during preprocessing
        num_features_total = len(non_sparse_feature_idxs) + len(sparse_feature_idxs)
        new_data = np.zeros((num_sequences, max_seq_len, num_features_total))
        new_data[:,:,non_sparse_feature_idxs] = padded_sequences

        # get the prediction network
        prediction_network = messages['sparse_feature_prediction_network'] # type: nn.Module
        prediction_network.eval()

        training_device = get_default_device()
        prediction_network.to(training_device)

        # prepare data loader for network and perform actual prediction of sparse features
        dataset = SparseFeaturePredictionDataSet(new_data, padding_mask, dense_feature_idxs, self._window_size, scalers['predictors'])

        target_scaler = scalers['sparse_features']

        offset = 0
        # had_infty = False
        with torch.no_grad():
            for x_batch in dataset.get_data_loader(self._batch_size, device=training_device):
                y_batch = prediction_network(x_batch)
                assert(y_batch.shape[-1] == len(sparse_feature_idxs))
                y_batch, _ = target_scaler.postprocess(y_batch.cpu(), {})

                # if not np.all(np.isfinite(y_batch)):
                #     had_infty = True
                #     y_batch[~np.isfinite(y_batch)] = np.nan

                new_data[offset : offset + len(y_batch), :, sparse_feature_idxs] = y_batch

        # if had_infty:
        #     logging.warn("sparse feature prediction produced non-finite values (which I replaced by NaN)")


        if np.any(padding_mask):
            padding_value = padded_sequences[padding_mask].reshape(-1)[0]
            new_data = np.where(np.expand_dims(padding_mask, -1), padding_value, new_data)

        del messages['sparse_features']
        del messages['non_sparse_features']
        del messages['dense_features']
        del messages['sparse_feature_prediction_network']
        del messages['sparse_feature_prediction_scalers']

        return new_data, messages

class ReplaceWithOriginal(DataTransformer):
    """ Replaces values in generated data with those from original data for a given
    set of feature names.

    We use this to deal with very sparse features and those that we perform badly on in evaluation.

    During preprocessing they are simply removed from the data (so that models don't have to worry
    about them). During postprocessing, the original data for these features is output with a small
    amount of noise added.
    """

    def __init__(self, sparsity_threshold: Optional[float]=0.01, sample_noise: Optional[float]=0.01, bad_feature_names=None):
        self._sparsity_treshold = sparsity_threshold
        self._sample_noise = sample_noise
        self._bad_feature_names = bad_feature_names

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(padded_sequences.shape[:-1], dtype=bool)

        # identify sparse features to replace by their median
        avg_feature_density = calculate_avg_feature_density(padded_sequences, padding_mask)
        sparse_feature_idxs = np.where(avg_feature_density < self._sparsity_treshold)[0]

        # hack!
        if self._bad_feature_names is not None:
            bad_feature_idxs = [i for i, feat in enumerate(messages['current_features']) if feat in self._bad_feature_names]
            sparse_feature_idxs = np.unique(np.concatenate((sparse_feature_idxs, bad_feature_idxs)))
        # end hack!

        non_sparse_feature_idxs = [idx for idx in range(padded_sequences.shape[-1]) if idx not in sparse_feature_idxs]
        logging.info('identified {} sparse (with avg. frequency < {} per sequence)'.format(
            len(sparse_feature_idxs), self._sparsity_treshold
        ))
        logging.debug('sparse feature ids: {}'.format(sparse_feature_idxs))

        messages['supersparse_features'] = sparse_feature_idxs
        messages['non_supersparse_features'] = non_sparse_feature_idxs

        # median_vals = np.nanmedian(np.where(np.expand_dims(padding_mask, -1), np.nan, padded_sequences)[:,:,sparse_feature_idxs], axis=(0,1))
        # assert(median_vals.shape == (len(sparse_feature_idxs), ))
        messages['supersparse_feature_values'] = np.where(np.expand_dims(padding_mask, -1), np.nan, padded_sequences)[:,:,sparse_feature_idxs]

        new_data = padded_sequences[:,:,non_sparse_feature_idxs]

        return new_data, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        sparse_feature_idxs = messages['supersparse_features']
        non_sparse_feature_idxs = messages['non_supersparse_features'] # gives the original feature indices in padded_sequences
        feature_values = messages['supersparse_feature_values']
        assert(feature_values.shape[-1] == len(sparse_feature_idxs))

        num_sequences, max_seq_len, num_features_current = padded_sequences.shape
        assert(num_features_current == len(non_sparse_feature_idxs))

        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros((num_sequences, max_seq_len), dtype=bool)

        # noise to mask original data a bit
        sparse_feature_sds = np.nanstd(feature_values.reshape(-1, len(sparse_feature_idxs)), axis=0)
        noise = np.random.randn(num_sequences, max_seq_len, len(sparse_feature_idxs)) * self._sample_noise * sparse_feature_sds

        num_features_total = len(non_sparse_feature_idxs) + len(sparse_feature_idxs)
        new_data = np.zeros((num_sequences, max_seq_len, num_features_total))
        new_data[:,:,non_sparse_feature_idxs] = padded_sequences
        new_data[:,:,sparse_feature_idxs] = feature_values + noise

        # enforcing padding
        if np.any(padding_mask):
            padding_value = padded_sequences[padding_mask].reshape(-1)[0]
            new_data = np.where(np.expand_dims(padding_mask, -1), padding_value, new_data)

        del messages['supersparse_features']
        del messages['non_supersparse_features']
        del messages['supersparse_feature_values']

        return new_data, messages


class SparseFeatureMedianInserter(DataTransformer):
    """ Replaces values for very sparse features in generated data with the median from original data.

    We use this to deal with very sparse features and those that we perform badly on in evaluation.

    NOT USED: THIS IS AWFUL WITH THE MIN-MAX SCALING PERFORMED BY THE COMPETITION CODE.
    """

    def __init__(self, sparsity_threshold: Optional[float]=0.01, sample_noise: Optional[float]=1e-6, bad_feature_names=None):
        self._sparsity_treshold = sparsity_threshold
        self._sample_noise = sample_noise
        self._bad_feature_names = bad_feature_names

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(padded_sequences.shape[:-1], dtype=bool)

        # identify sparse features to replace by their median
        avg_feature_density = calculate_avg_feature_density(padded_sequences, padding_mask)
        sparse_feature_idxs = np.where(avg_feature_density < self._sparsity_treshold)[0]

        # hack!
        if self._bad_feature_names is not None:
            bad_feature_idxs = [i for i, feat in enumerate(messages['current_features']) if feat in self._bad_feature_names]
            sparse_feature_idxs = np.unique(np.concatenate((sparse_feature_idxs, bad_feature_idxs)))
        # end hack!

        non_sparse_feature_idxs = [idx for idx in range(padded_sequences.shape[-1]) if idx not in sparse_feature_idxs]
        logging.info('identified {} sparse (with avg. frequency < {} per sequence)'.format(
            len(sparse_feature_idxs), self._sparsity_treshold
        ))
        logging.debug('sparse feature ids: {}'.format(sparse_feature_idxs))

        messages['supersparse_features'] = sparse_feature_idxs
        messages['non_supersparse_features'] = non_sparse_feature_idxs

        median_vals = np.nanmedian(np.where(np.expand_dims(padding_mask, -1), np.nan, padded_sequences)[:,:,sparse_feature_idxs], axis=(0,1))
        assert(median_vals.shape == (len(sparse_feature_idxs), ))
        messages['supersparse_feature_medians'] = median_vals

        new_data = padded_sequences[:,:,non_sparse_feature_idxs]

        return new_data, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        sparse_feature_idxs = messages['supersparse_features']
        non_sparse_feature_idxs = messages['non_supersparse_features'] # gives the original feature indices in padded_sequences
        median_vals = messages['supersparse_feature_medians']
        assert(median_vals.shape == (len(sparse_feature_idxs), ))

        num_sequences, max_seq_len, num_features_current = padded_sequences.shape
        assert(num_features_current == len(non_sparse_feature_idxs))

        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros((num_sequences, max_seq_len), dtype=bool)

        # tiny amount of noise on top of median values to ensure more than 1 value
        # note: the only purpose of this is to not trigger classification mode
        #   in the utility scoring - it HAS NOTHING TO DO WITH PRIVACY
        noise = np.random.randn(num_sequences, max_seq_len, len(sparse_feature_idxs)) * self._sample_noise

        num_features_total = len(non_sparse_feature_idxs) + len(sparse_feature_idxs)
        new_data = np.zeros((num_sequences, max_seq_len, num_features_total))
        new_data[:,:,non_sparse_feature_idxs] = padded_sequences
        new_data[:,:,sparse_feature_idxs] = median_vals[np.newaxis, np.newaxis, :] + noise

        # enforcing padding
        if np.any(padding_mask):
            padding_value = padded_sequences[padding_mask].reshape(-1)[0]
            new_data = np.where(np.expand_dims(padding_mask, -1), padding_value, new_data)

        del messages['supersparse_features']
        del messages['non_supersparse_features']
        del messages['supersparse_feature_medians']

        return new_data, messages

class SparseFeatureRemover(DataTransformer):
    """ NOT USED """

    def __init__(self, sparsity_threshold: Optional[float]=0.01):
        self._sparsity_treshold = sparsity_threshold

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(padded_sequences.shape[:-1], dtype=bool)

        # identify sparse features to replace by their median
        avg_feature_density = calculate_avg_feature_density(padded_sequences, padding_mask)
        sparse_feature_idxs = np.where(avg_feature_density < self._sparsity_treshold)[0]
        non_sparse_feature_idxs = [idx for idx in range(padded_sequences.shape[-1]) if idx not in sparse_feature_idxs]
        logging.info('identified {} sparse (with avg. frequency < {} per sequence)'.format(
            len(sparse_feature_idxs), self._sparsity_treshold
        ))
        logging.debug('sparse feature ids: {}'.format(sparse_feature_idxs))

        messages['supersparse_features'] = sparse_feature_idxs
        messages['non_supersparse_features'] = non_sparse_feature_idxs

        new_data = padded_sequences[:,:,non_sparse_feature_idxs]

        return new_data, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        sparse_feature_idxs = messages['supersparse_features']
        non_sparse_feature_idxs = messages['non_supersparse_features'] # gives the original feature indices in padded_sequences

        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(num_sequences, max_seq_len, dtype=bool)

        num_sequences, max_seq_len, num_features_current = padded_sequences.shape
        assert(num_features_current == len(non_sparse_feature_idxs))

        num_features_total = len(non_sparse_feature_idxs) + len(sparse_feature_idxs)
        new_data = np.zeros((num_sequences, max_seq_len, num_features_total))
        new_data[:,:,non_sparse_feature_idxs] = padded_sequences
        new_data[:,:,sparse_feature_idxs] = np.nan

        # enforcing padding
        padding_value = padded_sequences[padding_mask].reshape(-1)[0]
        new_data = np.where(np.expand_dims(padding_mask, -1), padding_value, new_data)

        del messages['supersparse_features']
        del messages['non_supersparse_features']

        return new_data, messages


if __name__ == '__main__':
    # some irrelevant early testing code of sparse feature prediction with NN
    logging.getLogger().setLevel(logging.INFO)
    num_data = 1000
    seq_len = 5

    predictor_feature_idxs = [0, 3]
    sparse_feature_idxs = [1, 4, 5]


    np.random.seed(0)
    data = np.random.randn(num_data, seq_len, 6)
    for i in sparse_feature_idxs: # create some linear dependency of sparse features on predictors
        coeffs = np.random.randn(len(predictor_feature_idxs))
        data[:,:,i] = np.sum([data[:,:,j]*c for j, c in zip(predictor_feature_idxs, coeffs)], axis=0)

    data[np.random.uniform(size=data.shape) < .6] = np.nan
    paddings = np.random.uniform(0, seq_len//2+1, size=num_data).astype(int)
    padding_mask = np.squeeze(np.array(np.cumsum(np.eye(seq_len+1, seq_len, dtype=bool), axis=1))[seq_len-paddings])

    net = PredictionNetwork(len(predictor_feature_idxs), len(sparse_feature_idxs), 3)
    # net = SimplerNetwork(len(predictor_feature_idxs), len(sparse_feature_idxs), 3, [len(predictor_feature_idxs)], [2, 2])
    print(net)
    train_sparse_feature_predictor(data, padding_mask, predictor_feature_idxs, sparse_feature_idxs, net)