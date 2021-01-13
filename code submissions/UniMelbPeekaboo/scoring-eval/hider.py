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


    # use modified timegan ---
    from utils.misc import tf115_found
    assert tf115_found is True, "TensorFlow 1.15 not found, which is required to run timegan."
    from utils.timegan import timegan
    generated_data = timegan(data_imputed)
    return generated_data
