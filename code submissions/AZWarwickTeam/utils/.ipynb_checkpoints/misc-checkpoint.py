import os
import random
import contextlib
import logging
import json

import numpy as np

# Determine if TF 1.15 is present.
tf115_found = False
try:
    import tensorflow as tf

    if tf.__version__[:4] == "1.15":  # pylint: disable=no-member
        tf115_found = True
except ModuleNotFoundError:
    pass
if not tf115_found:
    print("TensorFlow 1.15 not found.")


def fix_all_random_seeds(random_seed):
    """
    Fix random seeds etc. for experiment reproducibility.
    
    Args:
        random_seed (int): Random seed to use.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    if tf115_found:
        tf.random.set_random_seed(random_seed)


@contextlib.contextmanager
def temp_seed_numpy(seed):
    """Set a temporary numpy seed: set the seed at the beginning of this context, then at the end, restore random 
    state to what it was before.

    Args:
        seed (int): Random seed to use.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def tf_set_log_level(level):
    if tf115_found:
        if level >= logging.FATAL:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
        if level >= logging.ERROR:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        if level >= logging.WARNING:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        logging.getLogger("tensorflow").setLevel(level)


@contextlib.contextmanager
def in_progress(stage):
    print(f"{stage}...")
    try:
        yield
    finally:
        print(f"{stage} DONE")


def read_competition_config(config_path):
    config_path = os.path.realpath(config_path)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = dict()
    return config
