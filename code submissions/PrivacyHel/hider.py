from generative_svd import generative_svd
import data_pipeline as dpl
import os
import numpy as np
from post_process import NAImputer, Padder, Perturber

def hider(input_dict, kwargs={}):
    """
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

    # setup preprocessing pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dictionary_path = os.path.join(script_dir, 'data_dictionary_modified.csv')
    pipeline, messages = dpl.make_default_pipeline(data_dictionary_path, use_feature_combiner=True, do_imputation=True, seed=input_dict['seed'])

    # preprocess!
    messages[dpl.MSG_PADDING_MASK] = np.copy(input_dict['padding_mask'])
    #print(messages[dpl.MSG_PADDING_MASK].shape)
    train_data, messages = pipeline.preprocess(input_dict['data'], messages)


    assert(not np.any(np.all(np.isclose(np.where(np.expand_dims(messages[dpl.MSG_PADDING_MASK], -1), 0., train_data), 0.), axis=(0,1))))
    assert(not np.any(np.isnan(train_data)))

    # add padding mask
    #kwargs['pad_mask'] = messages[dpl.MSG_PADDING_MASK]
    # get mask value
    kwargs['mask_val'] = train_data[messages[dpl.MSG_PADDING_MASK]][0,0]
    # generated data!
    generated_data = generative_svd(train_data, **kwargs)

    #print('checking recon accuracy..')
    #print('recon accurate: {}'.format( np.allclose(train_data, generated_data)))
    #import sys
    #sys.exit()

    # postprocess! (so that data looks like imputed with the competition preprocessing)
    messages[dpl.MSG_PADDING_MASK] = np.zeros_like(messages[dpl.MSG_PADDING_MASK], dtype=bool) # no padding for synthetic data
    generated_data, messages = pipeline.postprocess(generated_data, messages)

    #na_imputer = NAImputer()
    #generated_data = na_imputer.fit_transform(input_dict['data'], generated_data)

    perturber = Perturber()
    generated_data = perturber.perturb(generated_data, messages[dpl.MSG_PADDING_MASK])

    return generated_data, input_dict['padding_mask']

__all__ = ['hider']
