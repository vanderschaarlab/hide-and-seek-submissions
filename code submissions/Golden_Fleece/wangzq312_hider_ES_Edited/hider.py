"""
The hider module containing the `hider(...)` function.
"""
# pylint: disable=fixme
from typing import Dict, Union, Tuple, Optional
import numpy as np
from utils.data_preprocess import preprocess_data


def MinMaxScaler(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data [users, time, features]

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(data, axis=1)
    for i in range(data.shape[1]):
        data[:, i, :] -= min_val

    max_val = np.max(data, axis=1)
    for i in range(data.shape[1]):
        data[:, i, :] = np.true_divide(data[:, i, :], (max_val + 1e-7))

    return data, min_val, max_val


def MinMaxRecovery(data, min_val, max_val):
    """Reverse Operation (renormalization) of the Min-Max Normalizer.
        Args:
          - data: data [users, time, features]
          - min_val: minimum value of each dimension [featues, 1]
          - max_val: maximum value of each dimension [featues, 1]
    """
    for i in range(data.shape[1]):
        data[:, i, :] = np.multiply(data[:, i, :], max_val)

    for i in range(data.shape[1]):
        data[:, i, :] += min_val

    return data


def hider(input_dict: Dict) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Solution hider function.

    Args:
        input_dict (Dict): Dictionary that contains the hider function inputs, as below:
            * "seed" (int): Random seed provided by the competition, use for reproducibility.
            * "data" (np.ndarray of float): Input data, shape [num_examples, max_seq_len, num_features].
            * "padding_mask" (np.ndarray of bool): Padding mask of bools, same shape as data.

    Returns:
        Return format is:
            np.ndarray (of float) [, np.ndarray (of bool)]
        first argument is the hider generated data, expected shape [num_examples, max_seq_len, num_features]);
        second optional argument is the corresponding padding mask, same shape.
    """

    # Get the inputs.
    seed = input_dict["seed"]  # Random seed provided by the competition, use for reproducibility.
    data = input_dict["data"]  # Input data, shape [num_examples, max_seq_len, num_features].
    padding_mask = input_dict["padding_mask"]  # Padding mask of bools, same shape as data.


    # Get processed and imputed data, if desired:
    # data_preproc, data_imputed = preprocess_data(data, padding_mask)
    #
    # # TODO: Put your hider code to replace Example 1 below.
    # # Feel free play around with Examples 1 (add_noise) and 2 (timegan) below.
    #
    # # --- Example 1: add_noise ---
    # from examples.hider.add_noise import add_noise

    # Find where all the NaNs are
    #  ASSUMES DATA IS 3D
    nan_indices = np.argwhere(np.isnan(data))
    # Get user indexes, max seq len indexs, feat indexes
    nan_indices_users = [x[0] for x in nan_indices]
    nan_indices_seq = [x[1] for x in nan_indices]
    nan_indices_feats = [x[2] for x in nan_indices]

    # Turn all NaNs into 0
    # generated_data = data
    original_data = np.nan_to_num(data)


    # TODO: Put your hider code to replace Example 1 below.
    from adver import adver
    original_data, min_val, max_val = MinMaxScaler(original_data)

    generated_data = adver.adver(original_data) # Main hider function

    original_data = MinMaxRecovery(original_data, min_val, max_val)
    generated_data = MinMaxRecovery(generated_data, min_val, max_val)
    # data_preproc, generated_data = preprocess_data(generated_data, padding_mask)

    # Add back all the NaNs
    generated_data[nan_indices_users, nan_indices_seq, nan_indices_feats] = np.nan

    # generated_data = data
    generated_padding_mask = np.copy(padding_mask)

    return generated_data, generated_padding_mask
