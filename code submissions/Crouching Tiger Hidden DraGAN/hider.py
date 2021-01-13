"""
The hider module containing the `hider(...)` function.
"""
# pylint: disable=fixme
from typing import Dict, Union, Tuple, Optional
import numpy as np
from utils.data_preprocess import preprocess_data, get_medians_and_scaler, process_and_impute
from sklearn.preprocessing import StandardScaler



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

    # Preprocessing
    # The columns at indices 39 & 69, and 37 & 38 are pairs of variables with the only difference appearing to be
    # related to the date of admission. Patients in the dataset had non-missing observations for at most one variable
    # in each of the pairs of columns.
    pair1 = [69, 39]
    pair2 = [37, 38]
    # Combine data from both pairs into single column
    data[:,:,pair1[0]] = np.where(np.isnan(data[:,:,pair1[0]]),
                            data[:,:,pair1[1]],
                            data[:,:,pair1[0]])
    data[:,:,pair2[0]] = np.where(np.isnan(data[:,:,pair2[0]]),
                            data[:,:,pair2[1]],
                            data[:,:,pair2[0]])
    data = np.concatenate(
      [data[:,:,:pair2[0]+1], data[:,:,pair1[1]+1:]], axis=2
    )
    padding_mask = np.concatenate(
      [padding_mask[:,:,:pair2[0]+1], padding_mask[:,:,pair1[1]+1:]], axis=2
    )
    # Mean Arterial Pressure can be well estimated from heart rate, diastolic and systolic blood pressure, so the MAP
    # column is removed and it will be calculated later using one of the formulas.
    data = np.concatenate(
      [data[:,:,:3], data[:,:,4:]], axis=2
    )
    padding_mask = np.concatenate(
      [padding_mask[:,:,:3], padding_mask[:,:,4:]], axis=2
    )

    # # Get processed and imputed data, if desired:
    scaler, median_vals = get_medians_and_scaler(data, padding_mask)
    data_preproc, data_imputed = process_and_impute(data, padding_mask, scaler, median_vals)

    # --- Wasserstein GAN ---
    from examples.hider.timegan import model

    from utils.misc import tf115_found
    assert tf115_found is True, "TensorFlow 1.15 not found, which is required to run timegan."

    # model parameters
    iterations = 50000
    batch_size = 256
    generated_data = model.wgangp(data_imputed, iterations=iterations, codalab=True, dp=False, batch_size=batch_size)

    # Inverse scaling - can speed up by applying
    for i in range(len(generated_data)):
      generated_data[i, :, 1:] = scaler.inverse_transform(generated_data[i])[:, 1:]

    # Calculate MAP using generalised formula
    def generalised_map(data):
      """
      Mean arterial pressure can be well estimated with several formulas.
      dp: diastolic bp
      sp: systolic bp
      hr: heart rate
      References:
        https://www.jstage.jst.go.jp/article/ahs1995/14/6/14_6_293/_article
      """
      dp = data[:,:,[3]]
      sp = data[:,:,[2]]
      hr = data[:,:,[1]]
      # return dp + 0.01 * (sp - dp) * np.e ** (4.14 - 40.74/hr) # Generalised formula
      return (2 * sp + dp)/3 # Not generalised estimate but wont blow up for some hr
    generated_data = np.concatenate([
      generated_data[:,:,:4], generalised_map(generated_data), generated_data[:,:,4:]
    ], axis=2)
    # Add paired columns back in so data is correct shape
    generated_data = np.concatenate([
      generated_data[:, :, :37+1], generated_data[:, :, [37]], generated_data[:,:,[67]], generated_data[:, :, 38:]
    ], axis=2)
    return generated_data