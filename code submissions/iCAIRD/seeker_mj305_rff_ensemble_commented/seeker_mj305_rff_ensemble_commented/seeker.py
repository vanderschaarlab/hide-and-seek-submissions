"""
The seeker module containing the `seeker(...)` function.
"""
# pylint: disable=fixme
from typing import Dict
import numpy as np
from utils.data_preprocess import preprocess_data
from sklearn.ensemble import RandomForestClassifier

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

    '''Aggregate the re-id results from
        - the starting kit versions of KNN
        - and binary predictor,
        - and both types of LOGAN: DC-LOGAN
        - and Time-LOGAN: '''
    from examples.seeker.knn import knn_seeker
    from examples.seeker.DC_LOGAN.seeker import dclogan_seeker
    from examples.seeker.time_logan.time_logan import time_logan
    '''LOGAN    - a type of a membership inference method,
                that trains a GAN on hider-generated data and then
                uses its discriminator for telling apart training data
                and additional data.
                The assumption is that a good generative model would capture
                enough about the distribution of its training data. So 
                attacker/seeker can use its generated data as a reference
                to what the training data distribution is.
                Hence the discriminator trained on these generated data
                would losely distinguish between the training data distribution
                and everything else. Hence it would be useful in a seeker's quest.
                GANs used in LOGAN can be different - we tried DCGANs and Time-GANs.'''

    re_id = np.zeros((np.shape(enlarged_data)[0], 1, 4))
    re_id[:, :, 0] = dclogan_seeker(generated_data_imputed, enlarged_data_imputed)
    re_id[:, 0, 1] = knn_seeker.knn_seeker(generated_data_imputed, enlarged_data_imputed)
    re_id[:, 0, 3] = time_logan(generated_data_imputed, enlarged_data_imputed)    
    from utils.misc import tf115_found
    assert tf115_found is True, "TensorFlow 1.15 not found, which is required to run binary_predictor."
    from examples.seeker.binary_predictor import binary_predictor
    re_id[:, :, 2] = binary_predictor.binary_predictor(generated_data_imputed, enlarged_data_imputed, verbose=True)
    
    # This is the majority vote result, that is later used to complete the random forests re-id result:
    reidentified_data = ((np.sum(re_id, (1, 2)))>=3).astype(int)
    
    
    '''Random Forest result:'''
    from examples.hider.add_noise import multiply_noise
    # creating a larger training data for the random forest by
    # using a modification of hider adding noise function (multiplying instead):
    generated_data_additional = multiply_noise.multiply_noise(generated_data_imputed, noise_size=0.2)
    generated_data_additional_2 = multiply_noise.multiply_noise(generated_data_imputed, noise_size=0.2)
    generated_data_large = np.concatenate((generated_data_imputed, generated_data_additional, generated_data_additional_2))
    
    # training random forest (RF) with 1000 trees:
    clf = RandomForestClassifier(n_estimators=1000)
    print(np.shape(generated_data_imputed)[0], np.shape(generated_data_additional)[0])
    lbl = np.concatenate((np.ones(np.shape(generated_data_imputed)[0]), np.zeros(np.shape(generated_data_additional)[0]), np.zeros(np.shape(generated_data_additional)[0])))
    print(np.shape(lbl))
    generated_data_large_ = generated_data_large.reshape((len(lbl),-1))
    clf.fit(generated_data_large_, lbl)
    
    # getting the re-id results of the trained RF:
    enlarged_data_imputed_  = np.concatenate((enlarged_data_imputed, enlarged_data_imputed))[:len(lbl), :, :]
    print(np.shape(enlarged_data_imputed))
    enlarged_data_imputed_ = enlarged_data_imputed_.reshape((np.shape(enlarged_data_imputed_)[0],-1))
    print(np.shape(enlarged_data_imputed_))
    reidentified_data_rf=clf.predict(enlarged_data_imputed_)[:np.shape(enlarged_data_imputed)[0]]
    
    
    '''Combining the re-id results:
            When the majority vote of all other models agrees on 1,
            replace the re-id result of the random forest (whatever it is) by 1:'''
    reidentified_data_rf[np.where(reidentified_data == 1)] = np.ones(np.shape(reidentified_data_rf[np.where(reidentified_data == 1)]))

    return reidentified_data_rf

