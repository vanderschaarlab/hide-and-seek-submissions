""" DataTransformers for pre- and postprocessing.

Our models make certain assumptions of the data that are not met by
the data in the form handed to us by competition code. This module contains
classes implementing pre- and postprocessing steps to take the competition-provided
data to a format that works with our models, and transform generated data back
to competition format.
"""

import logging
from typing import List, Tuple, abstractmethod, Iterable, Dict, Any, TypeVar, Optional
import numpy as np
import pandas as pd
import sklearn.preprocessing
import warnings
from os import path
import pickle
import hashlib

# message site names
MSG_PADDING_MASK = 'padding_mask'
MSG_MISSING_VALUE_MASK = 'missing_feature_values'
MSG_SEQUENCE_START_MASK = 'sequence_start_mask'
MSG_MEDIANS = 'median_vals'
# end of messages site names

## NOTE: The following DataTransformer class is intended to be the base class
##       for implementing our preprocessing steps. It features a preprocess and
##       postprocess function for invertible transformations and is intended to
##       compose well with other transformers.
##       The DataPipeline class then applies a chain of transformers to take us
##       from raw to model-ready data (and back).
##       Further below is block with implementation stubs for our data pipeline
##       that need to be implemented and at the end there is a small usage example.

class DataTransformer:
    """ A step in our data transformation pipeline. Implements methods preprocess and postprocess. """

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Perform data preprocessing. Takes us from original raw data to something nice for our models to work with.

        Args:
            padded_sequences: A numpy array of shape (num_sequences x max_sequence_length x num_features) holding the data.
            messages: A keyed dictionary. Any DataTransformer in the pipeline may add, change or remove messages to receive
                pass on additional information about the data from earlier / to later steps in the data pipeline.
        Returns: a 2-tuple with the following contents:
            - a numpy array of shape (num_sequences x max_sequence_length_new x num_features_new) holding the preprocessing results.
            - a keyed dictionary of messages to be passed on to later stages.
        """
        return padded_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Perform data postprocessing. Takes us from the data our models work with to something that looks like the original raw data.

        Args:
            padded_sequences: A numpy array of shape (num_sequences x max_sequence_length x num_features) holding the data.
            messages: A keyed dictionary. Any DataTransformer in the pipeline may add, change or remove messages to receive
                pass on additional information about the data from earlier / to later steps in the data pipeline.
        Returns: a 2-tuple with the following contents:
            - a numpy array of shape (num_sequences x max_sequence_length_new x num_features_new) holding the postprocessing results.
            - a keyed dictionary of messages to be passed on to later stages.
        """
        return padded_sequences, messages


class DataPipeline(DataTransformer):
    """ A data transformation pipeline implementation based on chaining of DataTransformer instances. """

    def __init__(self, transformers: Iterable[DataTransformer], pipeline_seed: Optional[int]=None, caching_dir: Optional[str]=None) -> None:
        """
        Args:
            transformers: Iterable of DataTransformer instances that form the pipeline.
            pipeline_seed: An optional integer to fix pipeline randomness. If None (default), the pipeline
                will use global randomness.
            caching_dir: An optional path to a directory. If present, DataPipeline will store intermediate (and final)
                pipeline results and reuse them in later runs to avoid redoing the same preprocessing. Note that there
                is currently no way to detect changes in input data, so use this option only with the same data.
        """
        self.transformers = transformers
        self.seed = pipeline_seed
        self.caching_dir = caching_dir

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Perform data preprocessing. Takes us from original raw data to something nice for our models to work with.

        Applies the `preprocess` method of the `DataTransformer` instances this `DataPipeline` consists of in order, passing
        along messages emitted by each `DataTransformer`. If a caching directory was specified, the output of each
        `DataTransformer` in the pipeline will be stored in the caching directory after initial processing and
        reused in subsequent preprocessing runs.

        Args:
            padded_sequences: A numpy array of shape (num_sequences x max_sequence_length x num_features) holding the data.
            messages: A keyed dictionary. Any DataTransformer in the pipeline may add, change or remove messages to receive
                pass on additional information about the data from earlier / to later steps in the data pipeline.
        Returns: a 2-tuple with the following contents:
            - a numpy array of shape (num_sequences x max_sequence_length_new x num_features_new) holding the preprocessing results.
            - a keyed dictionary of messages to be passed on to later stages.
        """
        random_state = None
        if self.seed is not None:
            seed = 11*self.seed # ensuring different randomness in pre- and postprocessing
            random_state = np.random.get_state()
            np.random.seed(seed)

        running_hash = hashlib.md5("hash_init_value".encode('utf8')).hexdigest()

        for i, transformer in enumerate(self.transformers):
            running_hash = hashlib.md5("{}_seed_{}_{}_{}".format(i, self.seed, type(transformer), running_hash).encode('utf8')).hexdigest()

            # check if there is cached data available and skip computation if so
            if self.caching_dir is not None:
                file_path = path.join(self.caching_dir, "{}_seed_{}_hash_{}.cache".format(i, self.seed, running_hash))
                if path.exists(file_path):
                    with open(file_path, "rb") as f:
                        stored_data = pickle.load(f)
                    padded_sequences = stored_data['data']
                    messages = stored_data['messages']
                    continue

            padded_sequences, messages = transformer.preprocess(padded_sequences, messages)
            if not np.all(np.isnan(padded_sequences) | np.isfinite(padded_sequences)):
                logging.warn(f'preprocessing: infinite values in data after {type(transformer)}')

            # store preprocessed data in cache if desired
            if self.caching_dir is not None:
                file_path = path.join(self.caching_dir, "{}_seed_{}_hash_{}.cache".format(i, self.seed, running_hash))
                with open(file_path, "wb") as f:
                    pickle.dump({'data': padded_sequences, 'messages': messages}, f)

        if random_state is not None:
            np.random.set_state(random_state)
        return padded_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Perform data postprocessing. Takes us from the data our models work with to something that looks like the original raw data.

        Applies the `postprocess` method of the `DataTransformer` instances this `DataPipeline` consists of in reverse order,
        i.e., `DataTransformer`s that were applied first during preprocessing are applied last during postprocessing (and vice-versa).

        Caching is not used during postprocessing.

        Args:
            padded_sequences: A numpy array of shape (num_sequences x max_sequence_length x num_features) holding the data.
            messages: A keyed dictionary. Any DataTransformer in the pipeline may add, change or remove messages to receive
                pass on additional information about the data from earlier / to later steps in the data pipeline.
        Returns: a 2-tuple with the following contents:
            - a numpy array of shape (num_sequences x max_sequence_length_new x num_features_new) holding the postprocessing results.
            - a keyed dictionary of messages to be passed on to later stages.
        """
        random_state = None
        if self.seed is not None:
            seed = 13*(self.seed+1) # ensuring different randomness in pre- and postprocessing
            random_state = np.random.get_state()
            np.random.seed(seed)

        for transformer in reversed(self.transformers):
            padded_sequences, messages = transformer.postprocess(padded_sequences, messages)
            if not np.all(np.isnan(padded_sequences) | np.isfinite(padded_sequences)):
                logging.warn(f'postprocessing: infinite values in data after {type(transformer)}')

        if random_state is not None:
            np.random.set_state(random_state)
        return padded_sequences, messages

##############################################################################################
###### Preprocessing step implementations ####################################################
##############################################################################################

array_like = TypeVar('array_like', np.array, Tuple[float])

class PaddingDetector(DataTransformer):
    """ Detects padding time steps by looking for time steps where all features are the padding value. Does not change the data.

    Not used any longer since we now get a padding mask from the competition framework.
    """

    def __init__(self, padding_value=-1.) -> None:
        self.padding_value = padding_value

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            emits `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
        """
        zero_padded_sequences = padded_sequences - self.padding_value # changing padding to 0 (exactly!)
        padded_steps = np.sum(zero_padded_sequences, axis=-1) == 0
        # note(lumip): strict != 0 would usually be an issue with floats, but here
        #   it is a sensible filter: if(f) a time step is padded, all feature values are
        #   now exactly 0 and thus their sum will be exactly 0 as well
        messages[MSG_PADDING_MASK] = padded_steps
        return padded_sequences, messages

class PaddingMaskReshaper(DataTransformer):
    """ Converts between 3d and 2d padding masks.

    The competition framework uses 3d-padding mask, however our code expects the
    padding mask to be 2d (shape: num_instances, max_seq_len). This DataTransformer
    reshapes the padding mask accordingly during pre- and postprocessing.
    """


    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            reads `padding_mask`: a boolean mask of shape (num_data, max_seq_len, num_features)
            emits `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
        """

        if MSG_PADDING_MASK in messages:
            messages[MSG_PADDING_MASK] = np.all(messages[MSG_PADDING_MASK], axis=-1)

        return padded_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            emits `padding_mask`: a boolean mask of shape (num_data, max_seq_len, num_features)
            reads `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
        """
        if MSG_PADDING_MASK in messages:
            messages[MSG_PADDING_MASK] = np.broadcast_to(
                np.expand_dims(messages[MSG_PADDING_MASK], -1), np.shape(padded_sequences)
            )

        return padded_sequences, messages

class FlipPaddingTransfomer(DataTransformer):
    """ Moves the padding to the end of each sequence.

    CAUTION: Will not transform any messages other than `padding_mask`.
    Any transformers that introduce messages depending on positions on the data should be
    placed in the pipeline AFTER the FlipPaddingTransformer, or their messages will
    be out of sync with the data.
    """
    # todo(lumip,all): need to figure out how to transform messages?

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            reads/writes `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
        """
        padding_mask = messages[MSG_PADDING_MASK]
        padding_lens = np.sum(padding_mask, axis=-1)

        flipped_sequences = np.empty_like(padded_sequences)
        flipped_padding_mask = np.empty_like(padding_mask)

        for i in range(len(padded_sequences)):
            if padding_lens[i] == 0:
                flipped_sequences[i] = padded_sequences[i]
                flipped_padding_mask[i] = False
            else:
                flipped_sequences[i, :-padding_lens[i]] = padded_sequences[i, padding_lens[i]:]
                flipped_sequences[i, -padding_lens[i]:] = padded_sequences[i, :padding_lens[i]]
                flipped_padding_mask[i, :-padding_lens[i]] = False
                flipped_padding_mask[i, -padding_lens[i]:] = True

        messages[MSG_PADDING_MASK] = flipped_padding_mask
        return flipped_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            reads/writes `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
        """
        padding_mask = messages[MSG_PADDING_MASK]
        padding_lens = np.sum(padding_mask, axis=-1)

        flipped_sequences = np.empty_like(padded_sequences)
        flipped_padding_mask = np.empty_like(padding_mask)

        for i in range(len(padded_sequences)):
            if padding_lens[i] == 0:
                flipped_sequences[i] = padded_sequences[i]
                flipped_padding_mask[i] = False
            else:
                flipped_sequences[i, padding_lens[i]:] = padded_sequences[i, :-padding_lens[i]]
                flipped_sequences[i, :padding_lens[i]] = padded_sequences[i, -padding_lens[i]:]
                flipped_padding_mask[i, padding_lens[i]:] = False
                flipped_padding_mask[i, :padding_lens[i]] = True

        messages[MSG_PADDING_MASK] = flipped_padding_mask
        return flipped_sequences, messages

class QuantileFilterTransformer(DataTransformer):
    """ Filters data based on quantiles.

    For each feature, determines the lower and upper value corresponding to the
    given quantile bounds. Any value in the data exceeding these will be clipped
    to match.

    Ignores padded values if a `padding_mask` message is present.

    Messages:
        emits `quantiles`: a (2, num_features) array containing the quantile values (and thus new min/max bounds for the data).
            quantiles[0] contains the lower, quantiles[1] the upper quantile value for each feature dimension.
        emits `outlier_mask`: a Boolean mask of shape (num_data, max_seq_len, num_features), True where values have been
            detected as outliers by this filter.
        reads `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
    """

    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        super().__init__()
        self.quantiles = [lower_quantile, upper_quantile]

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_data, max_seq_len, num_features = padded_sequences.shape

        # determining quantile values
        timestep_features = np.reshape(padded_sequences, (-1, num_features))

        padding_mask = np.zeros((num_data, max_seq_len), dtype=np.bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        # we replace padding with nan here, so np.nanquantile will ignore it
        timestep_features = np.where(np.reshape(padding_mask, (-1, 1)), np.nan, timestep_features)

        quantiles = np.nanquantile(timestep_features, self.quantiles, interpolation='nearest', axis=0)
        messages['quantiles'] = quantiles

        # detecting and clipping low outliers
        lower_mask = (padded_sequences < quantiles[0]) & ~padding_mask.reshape((num_data, max_seq_len, 1))
        filtered_sequences = np.where(lower_mask, quantiles[0], padded_sequences)

        # detecting and clipping high outliers
        upper_mask = (padded_sequences > quantiles[1]) & ~padding_mask.reshape((num_data, max_seq_len, 1))
        filtered_sequences = np.where(upper_mask, quantiles[1], filtered_sequences)

        messages['outlier_mask'] = lower_mask ^ upper_mask

        return filtered_sequences, messages

class OutlierFilterTransformer(DataTransformer):
    """
    Filters data based on given boundary values.

    For each feature, outliers are clipped to boundary values that are defined
    based on prior knowledge about the feature's domain.

    Padded values are ignored by using padding_mask (from messages!)

    Messages:
        reads `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
        whether (True) or not (False) a time step in a sequence is padded.
    """

    def __init__(self, data_dictionary: pd.core.frame.DataFrame, replace_with_nans=False) -> None:
        """
        Args:
            - data_dictionary: a pandas DataFrame including the predetermined lower and upper
                limit values for each feature
            - replace_with_nans: If True, values exceeding the boundaries will
                be replaced with NaN instead of being clipped to the boundary value.
        """
        self.data_dictionary = data_dictionary
        self.replace_with_nans = replace_with_nans

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:

        # get boundary values
        data_bounds = np.zeros((71,2))
        data_bounds[:,0] = self.data_dictionary['lower_bound'].to_numpy()[1:] # drop admissionid
        data_bounds[:,1] = self.data_dictionary['upper_bound'].to_numpy()[1:] # drop admissionid

        from data_processing_utils import remove_outliers

        padded_sequences = remove_outliers(
            data=padded_sequences,
            data_bounds= data_bounds,
            padding_mask= messages['padding_mask'],
            replace_with_nans=self.replace_with_nans
        )

        return padded_sequences, messages

class ApplyInPostprocessing(DataTransformer):
    """ Wrapper to apply the preprocessing step of a given `DataTransformer` during postprocessing instead.

    Has no effect during preprocessing.
    """

    def __init__(self, transformer: DataTransformer) -> None:
        self.transformer = transformer
    def postprocess(self, data: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.transformer.preprocess(data, messages)

class CompetitionPreprocessor(DataTransformer):
    '''
    Implementation of the normalisation and imputation done in the competition preprocessing script: impute, then normalise.

    OUTDATED: CURRENTLY NOT USED ANY LONGER.

    Messages:
        reads   'median_vals' containing medians for all features
                'competition_prep_scaler' fitted standard scaler to be used after imputing values
    '''
    def preprocess(self, sequences: List[np.ndarray], messages: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        warnings.warn("CompetitionPreprocessor has not yet been tested!! You probably want to use the competition code for this anyways.")
        from data_processing_utils import competition_imputation

        logging.debug('Competition preprocessor input data length: {}'.format(len(sequences)))
        sequences = competition_imputation(sequences, messages['median_vals'])

        scaler = messages['competition_prep_scaler']
        for i in range(len(sequences)):
            sequences[i] = scaler.transform(sequences[i])

        return sequences, messages


from data_processing_utils import imputation_per_feature
class BetterImputer(DataTransformer):
    """ Imputes missing values in the same way that competition code does.

    A faster implementation of `CompetitionImputer`.
    """
    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:

        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(padded_sequences.shape[:-1], dtype=np.bool)

        if MSG_MEDIANS in messages:
            median_vals = messages[MSG_MEDIANS]
        else:
            # obtain feature medians, replace padding to nan for this
            nan_padded = np.where(np.expand_dims(padding_mask, -1), np.nan, padded_sequences)
            median_vals = np.nanmedian(nan_padded, axis=[0,1])

        imputed_sequences = imputation_per_feature(padded_sequences, median_vals, padding_mask)

        # reinsert padding
        padded_sequences = np.where(np.expand_dims(padding_mask, -1), padded_sequences, imputed_sequences)

        if MSG_MISSING_VALUE_MASK in messages:
            del messages[MSG_MISSING_VALUE_MASK]

        return padded_sequences, messages

class CompetitionImputer(DataTransformer):
    """ Imputes missing values in the same way that competition code does. """

    def __init__(self, forward_first=False):
        """
        Arguments:
            forward_first: If True, will perform forward fill (repeating last seen value)
                before backward fill (repeating next seen value). Otherwise, backward fill first.
        """
        self._forward_first = forward_first

    '''
    Implementation of the imputation done in the competition preprocessing script. This only imputes values, doesn't do any normalization.
    Messages:
        removes 'missing_feature_values' if present
        reads 'median_vals' if present: An np.array of shape (feature_dims,) with the median value for each feature (over individuals and time)
    '''
    def preprocess(self, sequences: List[np.ndarray], messages: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        from data_processing_utils import competition_imputation
        logging.debug('Competition imputer only input data length: {}'.format(len(sequences)))

        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]
        else:
            padding_mask = np.zeros(sequences.shape[:-1], dtype=np.bool)

        # remove padding to prevent it from affecting imputation
        unpadded_sequences = np.where(np.expand_dims(padding_mask, -1), np.nan, sequences)

        if MSG_MEDIANS in messages:
            median_vals = messages[MSG_MEDIANS]
        else:
            # obtain feature medians for filling sequences without any observation for a feature
            unpadded_observations = np.reshape(unpadded_sequences, (-1, sequences.shape[-1]))
            median_vals = np.nanmedian(unpadded_observations, axis=0)

        if self._forward_first:
            unpadded_sequences = np.flip(unpadded_sequences, axis=1)

        unpadded_sequences = competition_imputation(unpadded_sequences, median_vals)

        if self._forward_first:
            unpadded_sequences = np.flip(unpadded_sequences, axis=1)

        # reinsert padding
        unpadded_sequences = np.where(np.expand_dims(padding_mask, -1), sequences, unpadded_sequences)

        if MSG_MISSING_VALUE_MASK in messages:
            del messages[MSG_MISSING_VALUE_MASK]

        return unpadded_sequences, messages



class KalmanFilterImputer(DataTransformer):
    """ INCOMPLETE AND NOT USED """

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        warnings.warn("KalmanFilterImputer has not yet been tested!!")
        # NOTE: assume modified col names are supplied in the message?

        from data_processing_utils import Kalman_Gaussian_RW_filter as KF

        warnings.warn('Kalman filtering not done!')

        logging.debug('Starting Kalman filtering...')

        N,T,D = padded_sequences.shape
        #print('KF got shapes N={}, T={}, D={}'.format(N,T,D))

        for cols, prior_mean in zip(messages['cols_to_KF'],messages['prior_means_to_KF']):
            for i in range(N):
                #print('single shape: {}'.format(padded_sequences[i,:,cols].shape))
                warnings.warn('KF currently replaces all temp measurements with a single estimated mean! Should probably create a new feature for this')
                # replaces all temp measurements with the single estimated mean signal
                #
                padded_sequences[i,:,cols] = KF(np.transpose(padded_sequences[i,:,cols]), prior_mean=prior_mean)[0]
                #from matplotlib import pyplot as plt
                #plt.plot(np.transpose(padded_sequences[i,:,cols]))
                #plt.show()

        logging.debug('Kalman filtering done.')
        return padded_sequences, messages

        #def Kalman_Gaussian_RW_filter(y, prior_mean, prior_std=2, obs_noise=1, latent_noise=1, first_obs_as_prior=True):


class FeatureCombiner(DataTransformer):
    """ Eliminates feature dimensions in the data based on prior knowledge of
    feature correlations.

    During preprocessing removes features based on given functions:
    - remove_missing_features function removes features without any observations
    - remove_superfluous_features function removes features that are highly
      dependent on other features and can be estimated using them in a simple
      regression.

    During postprocessing reconstructs the removed features and either returns the
    original values (padding values and NaN's) or estimates the values based on
    other features (return_superfluous_features)
    """

    def __init__(self, feature_functions: List[str]) -> None:
        self.feature_functions = feature_functions
        self.padding_value = 999

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            reads 'mod_col_names': list with names of feature columns
            emits padding_value
            emits 'original_dimensions': the original shape of the data
            emits 'orig_features': list with name of feature columns before the feature combination = mod_col_names
            emits 'current_features': list with names of feature columns after the feature combination
            emits 'removed_features': dictory mapping names of removed feature columns to their original position
            emits 'reg_coef_map': coefficients to reconstruct removed features during postprocessing
        """
        # get padding value during preprocessing as generated data does not include padded values
        padding_mask = messages['padding_mask']
        messages['padding_value'] = padded_sequences[padding_mask][0,0]

        # save original dimensions of the array to the messages
        messages['original_dimensions'] = list(padded_sequences.shape)

        # get original feture names
        mod_col_names = messages['mod_col_names']


        # get features in the original data before anything is removed, and save
        orig_features = mod_col_names
        # add original features to messages
        messages['orig_features'] = orig_features


        # add list of features without removed features to messages
        # -> this reflects the current features in the data
        current_features = orig_features.copy()
        messages['current_features'] = current_features


        # make dictionary for removed features and their column indices
        removed_features = {}
        # add removed_features dictionary to messages
        messages['removed_features'] = removed_features

        # import functions
        from data_processing_utils import remove_superfluous_features, remove_missing_features

        # call listed functions
        if 'remove_missing_features' in self.feature_functions:
            padded_sequences, messages = remove_missing_features(self, padded_sequences, messages)


        if 'remove_superfluous_features' in self.feature_functions:
            try:
                padded_sequences, messages = remove_superfluous_features(padded_sequences, messages)
            except ValueError as ve:
                raise Exception("A feature required for regression prediction is missing.") from ve


        return padded_sequences, messages


    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        #logging.info('Starting featureComb postprocess..')

        # create array with original dimensions and fill columns from
        # synthetic data and leave places for removed features
        from data_processing_utils import create_array_for_returning_features
        padded_sequences, messages = create_array_for_returning_features(padded_sequences, messages)

        # import functions
        from data_processing_utils import return_superfluous_features

        if 'remove_superfluous_features' in self.feature_functions:
            padded_sequences, messages = return_superfluous_features(self, padded_sequences, messages)

        if 'remove_missing_features' in self.feature_functions:
            ind_removed_empty_columns = messages['ind_removed_empty_columns']
            padding_value = messages['padding_value']
            padding_mask = messages['padding_mask']
            for i in ind_removed_empty_columns:
                padded_sequences[:,:,i][~padding_mask] = np.nan
                padded_sequences[:,:,i][padding_mask] = padding_value

            del messages['ind_removed_empty_columns']

        del messages['current_features']
        del messages['removed_features']

        return padded_sequences, messages


class LogarithmicScaler(DataTransformer):

    """
    Transform non-negative values of right-skewed variables to their natural
    logarithm before modelling and back to original scale after modeling
    in the synthetic data.

    NOTE, that scaling assumes values to be scaledd between 0 and 1,
    and missing value to be -1 OR padding_mask to be in the messages

    Arguments:
        data_dictionary, a pandas DataFrame having information which features are right-skewed
    """

    def __init__(self, data_dictionary, epsilon = 1e-15):
        self.data_dictionary = data_dictionary
        self.epsilon = epsilon # to avoid issues with zeros
        self.ind_right_skewed = []
        self.names_right_skewed = []
        self.mask = []


    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:

        '''
        Scale rigth-skewed variables to their natural logatirhms before modelling
        '''

        from data_processing_utils import ln_scale

        padded_sequences, messages = ln_scale(self, padded_sequences, messages)

        return padded_sequences, messages

    def postprocess(self, synt_data: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:

        ''' Re-scale ln-scaled features '''

        from data_processing_utils import re_scale

        synt_data, messages = re_scale(self, synt_data, messages)

        return synt_data, messages




class ListTimeAligner(DataTransformer):
    """ Transforms list of np arrays such that all begin at time 0; no effect in postprocessing. Assume time is feature 0.

    NO LONGER IN USE
    """

    def preprocess(self, sequences: List[np.ndarray], messages: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        warnings.warn("Start times are already aligned by the competition code, you probably don't want to use the ListTimeAligner!!")
        for i in range(len(sequences)):
            t0 = np.nanmin(sequences[i][:,0], axis=-1)
            sequences[i][:,0] -= np.expand_dims(t0, -1)
        return sequences, messages


class TimeAligner(DataTransformer):
    """ Transforms sequences such that all begin at time 0; no effect in postprocessing """

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        warnings.warn("Start times are already aligned by the competition code, you probably don't want to use the TimeAligner!!")
        padding_mask = np.zeros(padded_sequences.shape[:-1], dtype=bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]

        t0 = np.nanmin(np.where(np.expand_dims(padding_mask, -1), np.nan, padded_sequences)[:,:,0], axis=-1)
        transformed_sequences = np.copy(padded_sequences)
        transformed_sequences[:,:,0] -= np.expand_dims(t0, -1)

        # reinstate proper padding
        transformed_sequences[:,:,0] = np.where(padding_mask, transformed_sequences[:,:,1], transformed_sequences[:,:,0])

        return transformed_sequences, messages

class TimeDifferenceTransformer(DataTransformer):
    """ Replaces absolute time values with logs of time deltas.

    The start time is turned into a missing value and the `missing_feature_values`
    message is adjusted accordingly (or emitted, if not previously present).

    Works with any padding (front/back).

    Can optionally emit a message containing the original start time of each sequence.
    """

    def __init__(self, time_feature_id: int=0, save_start_times: bool=True, epsilon=1e-15):
        """
        Args:
            time_feature_id (int): The index of the feature designating the time (default=0).
            save_start_times (bool): If `True`, will emit a message containing start times for each sequence.
        """
        super().__init__()
        self.save_start_times = save_start_times
        self.time_idx = time_feature_id
        self.epsilon = epsilon

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            reads `padding_mask` (if present)
            reads/emits `missing_feature_values`
            emits `sequence_start_mask`: a Boolean mask of shape (num_data, max_seq_len) indicating with a True value
                the start of each sequence
            emits `start_times`: an array of shape (num_data,) with the start time for each sequence
                (only if the `save_start_times` argument was set to `True` during class initialization)
        """
        time_idx = self.time_idx
        padding_mask = np.zeros(padded_sequences.shape[-1:], dtype=np.bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]

        # missing_values = np.zeros(padded_sequences.shape, dtype=np.bool)
        # if MSG_MISSING_VALUE_MASK in messages:
        #     missing_values = messages[MSG_MISSING_VALUE_MASK]

        transformed_sequences = np.copy(padded_sequences)
        time_sequences = transformed_sequences[:, :, time_idx]

        # find out where the sequences actually start
        seq_start_mask = np.cumsum(np.cumsum(~padding_mask, axis=-1), axis=-1)==1 # single cumsum has pathological case with sequence length 1, double cumsum fixes that
        assert(np.sum(seq_start_mask) == len(padded_sequences))
        messages[MSG_SEQUENCE_START_MASK] = seq_start_mask
        data_idxs, seq_start_idxs = np.where(seq_start_mask)
        if self.save_start_times:
            start_times = np.ones(len(time_sequences))*np.nan
            start_times[data_idxs] = np.diag(np.swapaxes(time_sequences[data_idxs], 0, 1)[seq_start_idxs])
            messages['start_times'] = start_times

        # transform time values to log of diff
        time_sequences[:, 1:] = np.log(np.diff(time_sequences)+self.epsilon)

        # set start value to missing and restore padding
        time_sequences = np.where(seq_start_mask, np.nan, time_sequences)
        time_sequences = np.where(padding_mask, padded_sequences[:,:,time_idx], time_sequences)

        transformed_sequences[:, :, time_idx] = time_sequences

        # missing_values[:,:,time_idx] = np.where(seq_start_mask, True, missing_values[:,:,time_idx])
        # messages[MSG_MISSING_VALUE_MASK] = missing_values

        return transformed_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        time_idx = self.time_idx
        padding_mask = np.zeros(padded_sequences.shape[-1:], dtype=np.bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]

        seq_start_mask = messages[MSG_SEQUENCE_START_MASK]
        # get start times for each sequence from messages (or default to 0.)
        if 'start_times' in messages:
            start_times = messages['start_times']
            start_times = start_times[~np.isnan(start_times)] # ignore start times for empty sequences
        else:
            start_times = np.zeros(np.sum(seq_start_mask))

        # revert log
        transformed_sequences = np.copy(padded_sequences)
        time_sequences = transformed_sequences[:, :, time_idx]
        time_sequences = np.exp(time_sequences)

        # set start times in sequences
        time_sequences[seq_start_mask] = start_times

        # force padding to 0 (to avoid it contributing to the cumsum)
        time_sequences = np.where(padding_mask, 0., time_sequences)

        # convert diff times back to absolute times
        time_sequences = np.cumsum(time_sequences, axis=-1)

        # restore padding
        time_sequences = np.where(padding_mask, padded_sequences[:,:,time_idx], time_sequences)

        transformed_sequences[:, :, time_idx] = time_sequences

        # fix missing value message
        # missing_values = messages[MSG_MISSING_VALUE_MASK]
        # missing_values[:,:,time_idx] = np.where(seq_start_mask, False, missing_values[:,:,time_idx])
        # messages[MSG_MISSING_VALUE_MASK] = missing_values

        return transformed_sequences, messages

class StandardScaler(DataTransformer):
    """ Performs standard z-scaling for each feature dimension (over all sequences and times)."""

    def __init__(self) -> None:
        super().__init__()
        self.scaler = sklearn.preprocessing.StandardScaler()

    """
    Messages:
        reads `padding_mask`: a boolean mask of shape (num_data, max_seq_len) indicating
                whether (True) or not (False) a time step in a sequence is padded.
    """
    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_data, max_seq_len, num_features = padded_sequences.shape
        #logging.info('padded_sequences shape in StdScaler preprocess: {}'.format(padded_sequences.shape))

        padding_mask = np.zeros((num_data, max_seq_len), dtype=np.bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]

        # replace padding with nan so StandardScaler will ignore it
        padding_removed = np.where(np.expand_dims(padding_mask, 2), np.nan, padded_sequences)

        scaled_padding_removed = self.scaler.fit_transform(
            padding_removed.reshape(-1, num_features)
        ).reshape(padded_sequences.shape)

        # put the original padding back in
        scaled_sequences = np.where(np.expand_dims(padding_mask, 2), padded_sequences, scaled_padding_removed)

        return scaled_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:

        num_data, max_seq_len, num_features = padded_sequences.shape
        #logging.info('padded_sequences shape in StdScaler postprocess: {}'.format(padded_sequences.shape))

        padding_mask = np.zeros((num_data, max_seq_len), dtype=np.bool)
        if MSG_PADDING_MASK in messages:
            #logging.info('MSG_PADDING_MASK found in messages')
            padding_mask = messages[MSG_PADDING_MASK]

        scaled_sequences = self.scaler.inverse_transform(
            padded_sequences.reshape(-1, num_features)
        ).reshape(padded_sequences.shape)

        # put the original padding back in
        scaled_sequences = np.where(np.expand_dims(padding_mask, 2), padded_sequences, scaled_sequences)

        return scaled_sequences, messages


class MissingValueDetector(DataTransformer):
    """ Leaves data untouched but adds a message `missing_feature_values` that holds a mask indicating missing values in the data.

    Missing values are identified by being nonfinite (either nan or +/-inf).

    Messages:
        emits `missing_feature_values`: Boolean mask of shape (num_data, max_seq_len, num_features) indicating missing values (Note: padding is not detected as missing by this detector)
    """

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        messages[MSG_MISSING_VALUE_MASK] = ~np.isfinite(padded_sequences)
        return padded_sequences, messages

class MaskedValueReplacer(DataTransformer):
    """ Replaces values indicated by a mask in a given message with a given value.

    NOT USED.
    """

    def __init__(self, message_name: str, preprocess_replacement_value: float=.0, postprocess_replacement_value: float=None, negate_mask=False) -> None:
        """
        Args:
            message_name: Which message site holds the mask to use for identifying values that should be replaced.
            preprocess_replacement_value: Which value to use to replace.
            postprocess_replacement_value: Which value to use for replacment during postprocessing. If None,
                data will not be modified during postprocessing.
            negate_mask: If True, uses the inverse of the mask given by message_name for identifying values that should be replaced.
        """
        super().__init__()
        self._message_name = message_name
        self._pre_replacement_value = preprocess_replacement_value
        self._post_replacement_value = postprocess_replacement_value
        self._negate = negate_mask

    def preprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._message_name in messages:
            mask = messages[self._message_name]
            if self._negate:
                mask = ~mask
            padded_sequences = np.where(mask, self._pre_replacement_value, padded_sequences)
        #logging.info('MVR preprocess shape: {}'.format(padded_sequences.shape))
        return padded_sequences, messages

    def postprocess(self, padded_sequences: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        #logging.info('MVR postprocess start shape: {}'.format(padded_sequences.shape))
        if self._post_replacement_value is None:
            return padded_sequences, messages

        if self._message_name in messages:
            mask = messages[self._message_name]
            if self._negate:
                mask = ~mask
            padded_sequences = np.where(mask, self._post_replacement_value, padded_sequences)
        #logging.info('MVR postprocess end shape: {}'.format(padded_sequences.shape))
        return padded_sequences, messages

########## The following are for transforming raw data into a nicely padded numpy array.
########## They do not strictly implement DataTransformer because they take differnt types of data inputs...
########## .. but hey, it's python....

class DataPadder(DataTransformer):
    """ Assembles a list of sequences into a numpy array of shape (num_sequences x max_sequence_length x num_features) and pads
        with nan or cuts a sequence if necessary.

    NOT USED
    """

    def __init__(self, max_sequence_length: int = 100, invert_padding: bool = True) -> None:
        """
        Args:
            max_sequence_length: the maximum sequence length in the transformed data
        """
        self.max_sequence_length = max_sequence_length
        self.invert_padding = invert_padding

    def preprocess(self, sequences: List[np.ndarray], messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Assembles a list of sequences into a numpy array of shape (num_sequences x max_sequence_length x num_features) and pads
            with nan or cuts a sequence if necessary.

        Messages:
            emits `sequence_masks`: (num_sequences x max_sequence_length) indicating present observations with True, absent (padded) with False
        """
        warnings.warn("Sequences are already padded by the competition code, you probably don't want to use the DataPadder!!")
        data = np.ones((len(sequences), self.max_sequence_length, sequences[0].shape[-1])) * np.nan
        masks = np.zeros((len(sequences), self.max_sequence_length), dtype=np.bool)
        for i in range(len(sequences)):
            sequence = sequences[i]
            sequence_length = len(sequence)
            if sequence_length >= self.max_sequence_length:
                data[i] = sequence[:self.max_sequence_length]
                masks[i] = np.ones(self.max_sequence_length)
            else:
                if self.invert_padding:
                    data[i][:sequence_length] = sequence
                    masks[i][:sequence_length] = np.ones(sequence_length, dtype=np.bool)
                else:
                    data[i][-sequence_length:] = sequence
                    masks[i][-sequence_length:] = np.ones(sequence_length, dtype=np.bool)

        messages['sequence_masks'] = masks
        return data, messages

    def postprocess(self, padded_data: np.ndarray, messages: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """ Transforms padded sequences back into list of sequences with variable lengths.
        If a `sequence_masks` message is present, sequence lengths will be inferred from that. Otherwise, full lengths sequences will be emitted.

        Messages:
            optionally reads `sequences_masks`: (num_sequences x max_sequence_length) to determine where to cut off padding from sequences
        """
        if 'sequence_masks' in messages:
            sequence_lengths = self.sequence_mask_to_lengths(messages['sequence_masks'])
        else:
            sequence_lengths = np.ones(len(padded_data)) * padded_data.shape[1]

        sequences = []
        for i in range(len(padded_data)):
            sequences.add(padded_data[i][:sequence_lengths[i]])
        return sequences, messages

    @staticmethod
    def sequence_mask_to_lengths(sequence_masks: np.ndarray) -> np.ndarray:
        return np.sum(sequence_masks, axis=-1)


class DataGrouper(DataTransformer):
    """ Groups incoming array of raw data points (time step measurements) into time sequences for each subject/admission.

    NOT USED
    """

    def preprocess(self, raw_data: np.ndarray, messages: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """ Groups incoming array of raw data points (time step measurements) into time sequences for each subject/admission.

        Messages:
            emits `subject_ids`: assigning data admission ids to positions in the returned list (shape: (num_sequences,))
        """
        warnings.warn("Timesteps are already grouped by admission by the competition code, you probably don't want to use the DataGrouper!!")
        subject_ids = np.unique(raw_data[:,0])

        data_by_subject = []
        for i in subject_ids:
            idxs = np.where(raw_data[:,0] == i)
            data_by_subject.append(raw_data[idxs][:,1:])

        messages['subject_ids'] = subject_ids
        return data_by_subject, messages

    def postprocess(self, sequences: Iterable[np.ndarray], messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Messages:
            optionally reads `subject_ids`: assigning data admission ids to positions in the returned list (shape: (num_sequences,))
        """
        if 'subject_ids' in messages:
            subjects_ids = messages['subject_ids']
        else:
            subjects_ids = np.arange(len(sequences))

        num_data = 0
        num_features = 0
        for sequence in sequences:
            num_data = len(sequence)
            num_features = sequence.shape[-1]

        data = np.empty((num_data, num_features + 1))
        offset = 0
        for i, sequence in enumerate(sequence):
            end = offset + len(sequence)
            data[offset:end, 0] = subjects_ids[i]
            data[offset:end, 1:] = sequence

        return data, messages

class FinitenessEnforcer(DataTransformer):
    """ Replaces all infinity values with nans in pre- and postprocessing. """

    def preprocess(self, data: np.ndarray, messages: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if np.any(~np.isfinite(data) & ~np.isnan(data)):
            logging.debug("infinite values in data replaced with nan")
        filtered = np.where(np.isfinite(data), data, np.nan)
        return filtered, messages

    def postprocess(self, data: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if np.any(~np.isfinite(data) & ~np.isnan(data)):
            logging.debug("infinite values in data replaced with nan")
        filtered = np.where(np.isfinite(data), data, np.nan)
        return filtered, messages

class AddNoise(DataTransformer):
    """ Perturbs specified features in the data by adding a small amount of iid noise.

    Adds given noise std*N(0,1)*feature_std noise as postprosessing to the specified features in the generated data.

    Use this wrapped in `ApplyInPostprocessing` transformer to apply it to generated data in the pipeline.
    """
    def __init__(self, noise_std: float = 0.0, feature_list: list = []) -> None:
        """
        Args:
            noise_std: noise level
            feature_list: list of features where noise is added (indices in the full data)
        """
        self.noise_std = noise_std
        self.feature_list = feature_list

    def preprocess(self, generated_data: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not len(self.feature_list) > 0 or self.noise_std == 0:
            logging.debug('Skipping add noise in postprocess..')
            return generated_data, messages
        N,T,D = generated_data.shape
        assert D == 71, 'Add noise should have full dim data, instead got {} dims!'.format(D)
        logging.debug('Running add noise with noise std {}, feature_list: {}, full data shape {}'.format(self.noise_std, self.feature_list ,generated_data.shape))
        generated_data = generated_data.reshape((N*T,D))
        stds = np.nanstd(generated_data,0)
        for dim in self.feature_list:
            generated_data[:,dim] = generated_data[:,dim] + self.noise_std*stds[dim]*np.random.rand(N*T)
        generated_data = generated_data.reshape((N,T,D))
        return generated_data, messages



class BadFeatureReplacer(DataTransformer):
    """Replace typically bad features with related better performing ones.

    We have identified a number of features that perform bad in the competition evaluation
    no matter which model we use or how we tune it, but for which highly
    correlated/equivalent features exist in the data. This `DataTransformer`
    replaced the bad features with the better.

    Use this wrapped in `ApplyInPostprocessing` transformer to apply it to generated data in the pipeline.
    """
    def __init__(self, feature_replacements: dict={}, orig_feature_names: list=[]) -> None:
        """
        Args:
            feature_replacements: dict of features, in format 'what to replace' : 'with what'
            orig_feature_names: list of names for the original features
        """
        self.feature_replacements = feature_replacements
        self.orig_feature_names = orig_feature_names

    def preprocess(self, generated_data: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if len(self.feature_replacements) == 0 or len(self.orig_feature_names) == 0 :
            logging.debug('Skipping BadFeatureReplacer in postprocessing!')

        N,T,D = generated_data.shape
        assert D == len(self.orig_feature_names), 'BadFeatureReplacer: Data dims {} do not match orig features len {}!'.format(D,len(self.orig_feature_names))

        #print(self.orig_feature_names)
        generated_data = generated_data.reshape((N*T,D))
        #print(generated_data[-5:,63])
        #print(generated_data[-5:,9])

        for k in self.feature_replacements:
            # inds: (what to replace , with what)
            inds = (self.orig_feature_names.index(k), self.orig_feature_names.index(self.feature_replacements[k]))
            logging.debug('BadFeatureReplacer: replacing {} with {}'.format(k, self.feature_replacements[k]))
            generated_data[:,inds[0]] = generated_data[:,inds[1]]

        #print(generated_data[-5:,63])
        generated_data = generated_data.reshape((N,T,D))
        return generated_data, messages


class ZeroFeatureRemover(DataTransformer):
    """ Removes all features that only have 0 values.

    Used after `CompetitionImputer` and `StandardScaler` in the pipeline, a feature with all
    0 values is indicative of a constant feature after imputation.
    We easen the task of our models by removing these during preprocessing
    and re-inserting during postprocessing.
    """

    def preprocess(self, data: np.ndarray, messages: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        padding_mask = np.zeros(data.shape[0:1], dtype=bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]

        zero_feature_mask = np.all(
            np.isclose(np.where(np.expand_dims(padding_mask, -1), 0., data), 0.),
            axis=(0,1)
        )

        zero_feature_idxs = np.where(zero_feature_mask)[0]
        nonzero_feature_idxs = [i for i in range(data.shape[-1]) if i not in zero_feature_idxs]

        messages['zero_feature_idxs'] = zero_feature_idxs
        messages['nonzero_feature_idxs'] = [i for i in range(data.shape[-1]) if i not in zero_feature_idxs]
        new_data = data[:,:,nonzero_feature_idxs]

        return new_data, messages

    def postprocess(self, data: np.ndarray, messages: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        zero_feature_idxs = messages['zero_feature_idxs']
        nonzero_feature_idxs = messages['nonzero_feature_idxs']

        padding_mask = np.zeros(data.shape[0:1], dtype=bool)
        if MSG_PADDING_MASK in messages:
            padding_mask = messages[MSG_PADDING_MASK]

        feature_count = len(zero_feature_idxs) + len(nonzero_feature_idxs)
        assert(feature_count >= data.shape[-1])

        new_data = np.zeros((data.shape[0], data.shape[1], feature_count))
        new_data[:,:,nonzero_feature_idxs] = data

        if np.any(padding_mask):
            padding_value = data[padding_mask].reshape(-1)[0]
            new_data = np.where(np.expand_dims(padding_mask, -1), padding_value, new_data)

        del messages['zero_feature_idxs']
        del messages['nonzero_feature_idxs']

        return new_data, messages
