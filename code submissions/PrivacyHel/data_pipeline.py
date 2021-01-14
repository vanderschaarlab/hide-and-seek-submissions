""" Implementation of default data pipeline.

Our models make certain assumptions of the data that are not met by
the data in the form handed to us by competition code. This module sets up
our default data processing pipeline that transforms between these two
data formats, using the transformers implemented in `data_pipeline_transformers.py`.
"""
from data_pipeline_transformers import *
from sparse_feature_prediction import SparseFeatureMedianInserter, SparseFeaturePredictionTransformer, PredictionNetwork, ReplaceWithOriginal
from typing import Optional
import logging

def make_default_pipeline(
        data_dictionary_path,
        seed=None,
        verbose=False,
        use_feature_combiner=True,
        do_imputation=True,
        forward_fill_first=False,
        caching_dir: Optional[str]=None,
        do_time_align_zero=False,
        do_time_logdiff=False,
        use_prediction_network=True,
        prediction_network_train_epochs=250,
        sparsity_threshold=0.05,
        super_sparsity_threshold=0.01,
        add_noise_std=0.0,
        use_bad_feature_replacer=True,
        ):
    """
    Sets up our preprocessing pipeline.

    Arguments:
        data_dictionary_path: Full file path to the `data_dictionary_modified.csv`.
        seed: (Optional) Fixed seed to use within the pipeline.
        verbose: (Optional, False) If True, will print debug values.
        use_feature_combiner: (Optional, True) If True, will include the `FeatureCombiner` in the pipeline.
        do_imputation: (Optional, True) If True, will perform imputation for holes in the data.
        forward_fill_first: (Optional, False) If True and `do_imputation` is True, imputation will use forward filling first.
        caching_dir: (Optional) If given, the pipeline will cache in and reuse results from this directory.
        do_time_align_zero: (Optional) If True, will align all sequences to start at time 0.
        do_time_logdiff: (Optional) If True, will transform time feature values by taking the log of their differences.
        use_prediction_network: (Optional) If True, will use simple network-based prediction for sparse (but not super-sparse) features from dense ones.
        prediction_network_train_epochs: (Optional) Number of epochs to train the sparse feature prediction network for (only relevant if use_prediction_network is True).
        sparsity_threshold: (Optional) Average per-sequence frequency below which features are considered to be sparse.
        super_sparsity_threshold: (Optional) Average per-sequence frequency below which features are considered to be super-sparse (these will be predicted by their median).
        add_noise_std: add given std*feature std*N(0,1) noise to generated features, the list of features is defined below
        use_bad_feature_replacer: (bool) replace usually failing features directly by related other features, which tend to pass
    """


    # common messages for all pipelines
    messages = {}


    data_dictionary = pd.read_csv(data_dictionary_path, sep=';')

    # column names for feature combination
    messages['mod_col_names'] = list(data_dictionary['feature_name'][1:]) # drop admissionid
    messages['verbose'] = verbose # for testing, changed to True

    # create message with current features, try if this is enough to not crash
    messages['current_features'] = messages['mod_col_names'].copy()

    logging.info('Using custom pipeline')

    # feature list where to add noise in postprocessing, [] to skip, this should be run on the full generated data
    # these are feature our models are typically doing quite well on, so we add some additional
    # perturbation to increase privacy
    add_noise_feature_list = [0, 1, 2, 3, 7, 8, 13, 15, 17, 30, 47, 70] # Note: these are based on the ssm full runs in scores

    # these are the names of features that just don't want to be modelled well
    # their evaluation error in generated data is consistently too higher, no matter which model
    bad_features = [
        'dbp',
        'sodium',
        'glucose',
        'sodium_abg',
        'potassium_abg',
        'hb_abg',
        'exp_mv',
        'rr_set',
        'ph',
        'spo2_abg',
        'temp_rectal',
        'temp_groin',
        'temp_oesophagus'
    ]

    # input to pipeline is data as handed over by the competition script:
    #   1. series aligned to all start at time zero (not anymore?)
    #   2. all features min-max scaled (with scaler fit before step 1)
    #   3. sequences padded in front to max_seq_len=100 (longer sequences truncated), padding value -1
    #   result is a 3d numpy array (num_subjects, max_seq_len, num_features)

    transformers = [
        PaddingMaskReshaper(), # transforms 3d-padding mask into 2d (num_data, max_seq_len), effectively removing the feature dimension (other transformers expect this now)
        FinitenessEnforcer(), # ensures that no infinite values go in or out of the pipeline (get replaced by nan)
        ApplyInPostprocessing(OutlierFilterTransformer(data_dictionary, replace_with_nans=False)), # during postprocessing, clip all outliers to predefined boundary values, assumes data to be in orginial range (not scaled), and padding_mask to be saved in messages
        ApplyInPostprocessing(AddNoise(add_noise_std, add_noise_feature_list)), # during postprocessing, perturb features given in add_noise_feature_list for privacy (we do typically well on these, so there's some slack to perturb for more privacy)
        FlipPaddingTransfomer(), # move padding to the end of each sequence (also transforms `padding_mask` message [but no other message, so keep this at the start if needed])
    ]

    # replace some bad features with ones that tend to do better (in postprocessing)
    if use_bad_feature_replacer:
        # in format 'what to replace' : 'with what'
        feature_replacements = {
                                    # 'sodium_abg' : 'sodium',
                                    # 'potassium_abg' : 'potassium',
                                    # 'hb_abg' : 'hb',
                                    # 'exp_mv' : 'mv_spontaneous',
                                    # 'rr_set' : 'rr_ventilator',
                                    # check temp vars here, everything else : 'temp_blood'
                                    'temp_rectal' : 'temp_blood',
                                    'temp_groin' : 'temp_blood',
                                    'temp_axillary' : 'temp_blood',
                                    'temp_ear' : 'temp_blood',
                                    'temp_skin' : 'temp_blood',
                                    'temp_bladder' : 'temp_blood',
                                    'temp_oesophagus' : 'temp_blood',
                                    'exp_tidal' : 'tidal_volume',
                                    }
        bad_features = [feat for feat in bad_features if feat not in feature_replacements.keys()]
        transformers.append(ApplyInPostprocessing(BadFeatureReplacer(feature_replacements, messages['mod_col_names'].copy())))


    # change to log time diff
    if do_time_align_zero:
        transformers.append(TimeAligner())
    if do_time_logdiff:
        transformers.append(TimeDifferenceTransformer(epsilon=1e-15))

    transformers.append(QuantileFilterTransformer(0.05, 0.95)) # clip all data lying outside of [0.05,0.95] quantile interval to its boundaries
    # transformers.append(OutlierFilterTransformer(data_dictionary)) # clip all outliers to predefined boundary values, assumes data to be in orginial range (not scaled), and padding_mask to be saved in messages

    if use_feature_combiner:
        transformers.append(FeatureCombiner(['remove_missing_features', 'remove_superfluous_features'])) # remove_missing_features: remove features were all values are NaN or padding value, remove_superfluous_features: use prior knowledge to remove highly dependent features

    transformers.extend((
        # MissingValueDetector(), # detect all nans (does not detect padding, which is already set to -1 in input data); needs to come after FeatureCombiner or its output message ('missing_value_mask') will not match the data after removing features
        LogarithmicScaler(data_dictionary, epsilon = 1e-15), # does ln-scaling for right-skewed features, assumes values to be [0,1], and padded values to be -1 or padding_mask to be in messages
        ReplaceWithOriginal(sparsity_threshold=super_sparsity_threshold, sample_noise=1e-10, bad_feature_names=bad_features), # replace very sparse feature and those that are consistently bad in evaluation with the original data values + a bit of noise
        # SparseFeatureMedianInserter(sparsity_threshold=super_sparsity_threshold, sample_noise=1e-10, bad_feature_names=bad_features), # removes features with less than 1% observations (on average per sequence); during postprocessing, inserts median of observations perturbed with a bit of noise
    ))

    if use_prediction_network:
        # sets up a sparse feature prediction network given number of dense and sparse features
        def sparse_feature_prediction_network_factory(num_dense, num_sparse, window_size):
            pre_conv_hidden_layers = [num_dense//2]
            post_conv_hidden_layers = [num_dense//2, num_sparse]
            return PredictionNetwork(num_dense, num_sparse, window_size,
                pre_conv_hidden_layers, post_conv_hidden_layers)

        transformers.append(
            SparseFeaturePredictionTransformer( # predicts (remaining) sparse features (less than 5% observations per sequence on average) via NN
                sparse_feature_prediction_network_factory,
                sparsity_threshold=sparsity_threshold, density_threshold=0.2, window_size=5,
                max_epochs=prediction_network_train_epochs, conv_crit_len=5
            )
        )

    if do_imputation:
        transformers.append(
            CompetitionImputer(forward_first=forward_fill_first) # impute (remaining) missing values
        )


    transformers.append(StandardScaler()) # transforms all data to zero mean and unit variance (leaving nan untouched)
    transformers.append(ZeroFeatureRemover()) # removes all features that have only a single unique value left (which is now 0)

    # our pipeline currently additionally only detects missing values (nans) and replaces them with 0.
    pipeline = DataPipeline(transformers, pipeline_seed=seed, caching_dir=caching_dir)

    logging.info('Pipeline for train data:')
    for p in pipeline.transformers:
        logging.info(p)

    return pipeline, messages
