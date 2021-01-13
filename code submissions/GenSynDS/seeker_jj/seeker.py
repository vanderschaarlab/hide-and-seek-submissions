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
    from examples.seeker.knn import knn_seeker

    generated = [
        np.mean(generated_data_imputed, axis=1, keepdims=True),
        np.std(generated_data_imputed, axis=1, keepdims=True)
    ]
    enlarged = [
        np.mean(enlarged_data_imputed, axis=1, keepdims=True),
        np.std(enlarged_data_imputed, axis=1, keepdims=True)
    ]
    if (generated_data_padding_mask is not None and
        enlarged_data_padding_mask is not None):
        # If applicable, append to features the sum of mask for each time series (which is the number of non-missing data)
        generated.append(generated_data_padding_mask.sum(axis=1, keepdims=True))
        enlarged.append(enlarged_data_padding_mask.sum(axis=1, keepdims=True))
    
    generated_features = np.concatenate(generated, axis=1)
    enlarged_features = np.concatenate(enlarged, axis=1)
    
    reidentified_data = knn_seeker.knn_seeker(generated_features, enlarged_features)
    return reidentified_data
