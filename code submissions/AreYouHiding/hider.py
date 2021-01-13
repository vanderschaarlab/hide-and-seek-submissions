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
    """

    # NOTE: ES ADDED: Environment info #################################################################################
    
    # 1. Environment info:
    import sys
    import platform
    import tensorflow as tf
    import torch
    import subprocess
    print("=" * 80)
    print(f"OS [platform.platform()]:\n{platform.platform()}")
    print(f"\nPython [sys.version]:\n{sys.version}")
    print(f"\nTF [tf.__version__]:\n{tf.__version__}")  # pylint: disable=no-member
    print("\nnvidia-smi:\n")
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode("utf-8"))
    print(f"\nCUDA driver [torch._C._cuda_getDriverVersion()]:\n{torch._C._cuda_getDriverVersion()}")
    print(f"\nCUDA compiled version [torch.version.cuda]:\n{torch.version.cuda}")
    print(f"\nCuDNN version [torch.backends.cudnn.version()]:\n{torch.backends.cudnn.version()}")
    print("=" * 80)
    
    # 2. Disable TensorFlow GPU use (CPU will be used):
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    
    ####################################################################################################################

    # Get the inputs.
    seed = input_dict["seed"]  # Random seed provided by the competition, use for reproducibility.
    data = input_dict["data"]  # Input data, shape [num_examples, max_seq_len, num_features].
    padding_mask = input_dict["padding_mask"]  # Padding mask of bools, same shape as data.

    # Get processed and imputed data, if desired:
    data_preproc, data_imputed = preprocess_data(data, padding_mask)

    # TODO: Put your hider code to replace Example 1 below.
    # Feel free play around with Examples 1 (add_noise) and 2 (timegan) below.

    # --- Example 1: add_noise ---
    # from examples.hider.add_noise import add_noise

    # generated_data = add_noise.add_noise(data_imputed, noise_size=0.1)
    # generated_padding_mask = np.copy(padding_mask)
    # return generated_data, generated_padding_mask

    # --- Example 2: timegan ---
    # from utils.misc import tf115_found
    # assert tf115_found is True, "TensorFlow 1.15 not found, which is required to run timegan."
    from examples.hider.timegan import timegan as timegan
    generated_data = timegan.train_timegan(data_imputed)
    return generated_data
