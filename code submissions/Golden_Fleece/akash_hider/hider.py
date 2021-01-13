"""
The hider module containing the `hider(...)` function.
"""
# pylint: disable=fixme
from typing import Dict, Union, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from utils.data_preprocess import preprocess_data
from tqdm import tqdm

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
    no, seq_len, dim = original_data.shape

    # TODO: Put your hider code to replace Example 1 below.
    ##############################
    # Use minmax scaling to nomalize the data into [0,1]
    original_data, min_val, max_val = MinMaxScaler(original_data) # original_data is [no,100,71]
    generated_data = np.zeros(original_data.shape)

    for i in range(no):
        for j in range(seq_len):
            generated_data[i, j, 0] = original_data[i, j, 0] + np.random.normal(loc=0, scale=0.0008)

    for feature in tqdm(range(1, dim)):
        slices = np.squeeze(original_data[:,:,feature]).reshape([-1])
        # the data is normalized into [0,1], we focus on the non-extreme value only
        meaningful_vals = slices[slices>0.0001]
        meaningful_vals = meaningful_vals[meaningful_vals<0.9999]
        # cluster_model = KMeans(n_clusters=n_clusters, random_state=0)
        # cluster_model.fit(feature_embeddings)
        # labels = cluster_model.labels_
        if len(meaningful_vals):
            number_of_bins = 33
            # find the boundary of value bins.
            bounds = np.percentile(meaningful_vals, np.linspace(0,100,number_of_bins+1))
            # register: which bin does each data belong to?
            val_of_each_bin = [[] for i in range(number_of_bins)]
            # import pdb; pdb.set_trace()
            for each in meaningful_vals:
                idx = np.argmax(bounds > each) - 1
                val_of_each_bin[idx].append(each)

        for i in range(no):
            for j in range(seq_len):
                temp = original_data[i, j, feature]
                if temp <= 0.0001 or temp >= 0.9999:
                    generated_data[i, j, feature] = temp  # extreme value remain unchanged
                else:
                    # find the bin the current data belongs to
                    idx = np.argmax(bounds > temp) - 1
                    if len(val_of_each_bin[idx]) >= 1:
                        # randomly select another value data value in the same bin to replace the current data
                        generated_data[i, j, feature] = np.random.choice(val_of_each_bin[idx])
                    else:
                        generated_data[i, j, feature] = temp

        # import matplotlib.pyplot as plt
        # hist, bin_edges = np.histogram(meaningful_vals, 100, density=True)
        # plt.plot(bin_edges[:-1], hist)
        # plt.show()

        # number_of_bins = int(no/15.0)
        # bin_boundaries = np.linspace(bound[0], bound[1], num=number_of_bins)
        # import pdb; pdb.set_trace()
    # Undo the minmax scaling, i.e., renormalization
    original_data = MinMaxRecovery(original_data, min_val, max_val)
    generated_data = MinMaxRecovery(generated_data, min_val, max_val)
    ##############################



    # data_preproc, generated_data = preprocess_data(generated_data, padding_mask)
    # Add back all the NaNs
    generated_data[nan_indices_users, nan_indices_seq, nan_indices_feats] = np.nan

    # generated_data = data
    generated_padding_mask = np.copy(padding_mask)

    return generated_data, generated_padding_mask
