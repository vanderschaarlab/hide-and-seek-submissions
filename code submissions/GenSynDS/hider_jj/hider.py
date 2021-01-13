"""
The hider module containing the `hider(...)` function.
"""
# pylint: disable=fixme
from typing import Dict, Union, Tuple, Optional
import numpy as np
from utils.data_preprocess import preprocess_data


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
        
        Alternatively, may return a str "rescore" if there has been a previous successful submission and wishing to 
        just re-run the vs-seekers scoring step.
    """

    # Get the inputs.
    seed = input_dict["seed"]  # Random seed provided by the competition, use for reproducibility.
    data = input_dict["data"]  # Input data, shape [num_examples, max_seq_len, num_features].
    padding_mask = input_dict["padding_mask"]  # Padding mask of bools, same shape as data.

    # Get processed and imputed data, if desired:
    data_preproc, data_imputed = preprocess_data(data, padding_mask)

    # TODO: Put your hider code to replace Example 1 below.
    # Feel free play around with Examples 1 (add_noise) and 2 (timegan) below.

    # Rescale
    mus = np.mean(data_imputed, axis=1, keepdims=True)
    allstds = np.std(data_imputed, axis=1, keepdims=True)
    allstds[allstds == 0] = 1
    X_scaled = (data_imputed - mus) / allstds

    # Discriminate over variability into 2 clusters
    variability = (X_scaled[:, 1:, 1] - X_scaled[:, :-1, 1]).std(axis=1)
    all_people = np.array(list(range(len(data_imputed))))
    cat1 = all_people[variability[all_people] < 0.5]
    cat2 = all_people[variability[all_people] >= 0.5]
    print(len(set(cat1) | set(cat2)))
    indices = np.concatenate([cat1, cat2])
    
    from tslearn.barycenters import softdtw_barycenter
    generated_data = data_imputed.copy()
    for i in range(len(data_imputed)):
        # For each pair of consecutive time series within a cluster
        if i < len(cat1):
            a, b = cat1[i], cat1[(i + 1) % len(cat1)]
        else:
            j = i - len(cat1)
            a, b = cat2[j], cat2[(j + 1) % len(cat2)]
        # Compute their SoftDTW barycenter and rescale back to the mean and std of the first element in the pair
        generated_data[i] = softdtw_barycenter([X_scaled[a], X_scaled[b]]) * allstds[indices[i]] + mus[indices[i]]
    return generated_data#, np.ones_like(padding_mask)
