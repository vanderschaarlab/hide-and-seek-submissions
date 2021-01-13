import numpy as np
from utils.metric_utils import feature_prediction, one_step_ahead_prediction
from utils.data_preprocess import preprocess_data
from utils.general_rnn import GeneralRNN  # type: ignore
from typing import Dict, Union, Tuple, Optional
import time
from sklearn.cluster import MiniBatchKMeans

def get_pred_params(dim):
    """
    Simply returns a set of hyperparameters for our 'fitness' models

    Args:
        - dim: Dimensions for the model's hidden layer - identical to general_rnn.py in metric_utils
    Returns:
        - model_parameters: a python dictionary containing the hyperparameters for a 'fitness' model
    """

    # Set model parameters
    model_parameters = {
        "task": "regression",
        "model_type": "gru",
        "h_dim": dim,
        "n_layer": 3,
        "batch_size": 128,
        "epoch": 20,
        "learning_rate": 0.001,
    }

    return model_parameters

def rmse_error(y_true, y_pred):
    """User defined root mean squared error.

    Args:
        - y_true: true labels
        - y_pred: predictions

    Returns:
        - computed_rmse: computed rmse loss
    """
    # Exclude masked labels
    idx = (y_true >= 0) * 1
    # Mean squared loss excluding masked labels
    computed_mse = np.sum(idx * ((y_true - y_pred) ** 2)) / np.sum(idx)
    computed_rmse = np.sqrt(computed_mse)
    return computed_rmse


# Train utility models
def train_utility(input_data, n_users, max_seq_length, n_features, n_feature_predictors):

    """
    Trains and returns a list of 'fitness' models to be later used in our genetic algorithm

    Args:
        - input_data: real clinical time series data (Dimensions are [# users, max_seq_length, # features])
        - n_users: number of users in the input_data
        - max_seq_length: number of the datapoints in each user's time series (in input_data)
        - n_features: number of features for each user (should be 71)
        - n_feature_predictors: how many fitness models to train for the 'feature prediction' fitness task
    Returns:
        - feature_predictors: a list of TF Keras models that are trained on the feature prediction task for each feature index
        - one_step_predictors: a list of TF keras models that are trained on the one-step prediction task (should only be one model)
        - chosen_feature_idxs: A list of feature indexes used to train the feature_predictors
        - feature_gt: All data for a particular feature index (i.e. the ground truth used as the label for training this feature predictor)
        - one_step_gt: All data for the last time-step (i.e. the ground truth used as the label for training one-step-ahead prediction)
    """

    num_feature_predictors = n_feature_predictors
    feature_predictors = []
    one_step_predictors = []

    feature_gt = [] # Ground truth for feature prediction
    one_step_gt = [] #Ground truth for one-step

    # Train K feature predictors and 1 one-step-ahead predictor
    chosen_feature_idxs = np.random.choice(np.arange(n_features), num_feature_predictors, replace=False)
    for feature_idx in chosen_feature_idxs:

        # Set training features and labels
        train_x = np.concatenate((input_data[:, :, :feature_idx], \
            input_data[:, :, (feature_idx + 1) :]), axis=2)
        train_y = np.reshape(input_data[:, :, feature_idx], [n_users, max_seq_length, 1])
        feature_gt.append(train_y)

        general_rnn = GeneralRNN(get_pred_params(n_features))
        general_rnn.fit(train_x, train_y, verbose=False)
        feature_predictors.append(general_rnn)


    # Train one-step-ahead prediction
    # Set training features and labels
    train_x = input_data[:, :-1, :]
    train_y = input_data[:, 1:, :]
    one_step_gt.append(train_y)

    general_rnn = GeneralRNN(get_pred_params(n_features))
    general_rnn.fit(train_x, train_y, verbose=False)
    one_step_predictors.append(general_rnn)

    return feature_predictors, one_step_predictors, chosen_feature_idxs, feature_gt, one_step_gt

# Apply prediction on the features
def predict_features(input_data, feature_idx, predictor):

    """
    Runs prediction on a feature prediction model given some real or synthesized data

    Args:
        - input_data: real/synthesized data (Dimensions are [# users, max_seq_length, # features])
        - feature_idx: What feature index to predict on
        - predictor: TF Keras model to be used for feature prediction
    Returns:
        - input_data_y: The predicted feature values for the input_data
    """

    # Remove the feature index from our input data, and attempt to predict it.
    input_data_x = np.concatenate((input_data[:, :, :feature_idx], \
        input_data[:, :, (feature_idx + 1) :]), axis=2)
    input_data_y = predictor.predict(input_data_x)

    # input_data_y = np.reshape(input_data_y, [-1])

    return input_data_y

# Apply prediction on the features
def predict_one_step(input_data, predictor):

    """
    Runs prediction on the one_step_ahead prediction models

    Args:
        - input_data: real/synthesized data (Dimensions are [# users, max_seq_length, # features])
        - predictor: TF Keras model to be used for one-step-ahead prediction
    Returns:
        - input_data_y: The predicted one-step-ahead values for the input_data
    """

    input_data_x = input_data[:, :-1, :]
    input_data_y = predictor.predict(input_data_x)

    # input_data_y = np.reshape(input_data_y, [-1])

    return input_data_y

# Get all scores for an input data (using all predictors)
def get_scores(input_data, chosen_feature_idxs, feature_predictors, one_step_predictors):

    """
    Get the predictions from all our prediction models, both feature prediction and one-step-ahead models

    Args:
        - input_data: real/synthesized data (Dimensions are [# users, max_seq_length, # features])
        - chosen_feature_idxs: What feature indexes to predict on
        - feature_predictors: TF Keras models to be used for feature prediction
        - one_step_predictors: TF Keras models to be used for one-step-ahead prediction
    Returns:
        - feature_pred_scores: A list of predictions for all feature prediction models
        - one_step_scores: A list of one-step-ahead predictions for all one-step-ahead prediction models
    """

    # Get feature prediction scores and one_step_scores
    feature_pred_scores = []
    one_step_scores = []

    # Get scores for feature prediction original dataset
    for i, feature_idx in enumerate(chosen_feature_idxs):
        predictions = predict_features(input_data, feature_idx, feature_predictors[i])
        feature_pred_scores.append(predictions)

    for one_step_predictor in one_step_predictors:
        # Get scores for one step prediction
        predictions = predict_one_step(input_data, one_step_predictor)
        one_step_scores.append(predictions)

    return feature_pred_scores, one_step_scores

# Get average user
#  Assumes input is (n_users, max_seq_len, features)
def get_starting_synthesized_data(input_data):

    """
    An UNUSED function originally meant to create some initial synthesized data
    The method was simple - just create an 'average user' and copy it for all users.

    Args:
        - input_data: real/synthesized data (Dimensions are [# users, max_seq_length, # features])
    Returns:
        - synthesized_data: An initial ndarray of user time series data based on an average user
    """

    # Get average user
    avg_user = input_data.mean(axis=0)
    avg_user = np.expand_dims(avg_user, 0)

    # Copy average user into original shape
    synthesized_data = np.repeat(avg_user, input_data.shape[0], axis=0)

    return synthesized_data


# Check if we can pass the utility tests FOR A USER
#  Problem - we used training data as the ground truth, and compare against output
#  for training data.  we might not do a good job at hiding this way.
def check_if_pass(feat_pred_ori_user, one_step_pred_ori_user, feat_pred_syn_user, \
    one_step_pred_syn_user, feature_gt, one_step_gt):

    """
    Given the feature and one-step-ahead predictions for some synthesized/transformed clinical data,
    check if it passes a utility threshold.

    Args:
        - feat_pred_ori_user: Feature predictions for a particular user (list of predictions for several feature indexes)
            These predictions are predicted from the original data
        - one_step_pred_ori_user: one-step-ahead predictions for a particular user
            These predictions are predicted from the original data
        - feat_pred_syn_user: Feature predictions for a particular user (list of predictions for several feature indexes)
            These predictions are predicted from synthesized data
        - one_step_pred_syn_user: one-step-ahead predictions for a particular user
            These predictions are predicted from synthesized data
        - feature_gt: feature prediction labels for a user, for a particular set of features
            These are the ground truth values for the original data
        - one_step_gt: one-step-ahead labels for a particular user
            These are the ground truth values for the original data

    Returns:
        - user_utility_pass: A boolean function that determines whether ALL the synthesized predictions
            are within an MSE bounds of the original data predictions.  True if all of the predictions are within the threshold, False otherwise.
    """

    bounds = 4.5 # Have to be within X of the utility bounds
    passes = []
    user_utility_pass = True
    # Iterate through each feature prediction, and get RMSE
    for i, single_feature_gt in enumerate(feature_gt):  # 100, 1

        feat_pred_ori = feat_pred_ori_user[i]
        feat_pred_syn = feat_pred_syn_user[i]

        ori_rmse = rmse_error(single_feature_gt, feat_pred_ori)
        # print(ori_rmse)
        syn_rmse = rmse_error(single_feature_gt, feat_pred_syn)
        # print(syn_rmse)
        if ori_rmse * bounds < syn_rmse:
            user_utility_pass = False
            passes.append(False)
        else:
            passes.append(True)

    # Iterate through each one_step prediction, get RMSE
    for i, single_step_gt in enumerate(one_step_gt):  # 99, 71
        one_step_pred_ori = one_step_pred_ori_user[i]
        one_step_pred_syn = one_step_pred_syn_user[i]

        ori_rmse = rmse_error(single_step_gt, one_step_pred_ori)
        syn_rmse = rmse_error(single_step_gt, one_step_pred_syn)
        if ori_rmse * bounds < syn_rmse:
            user_utility_pass = False
            passes.append(False)
        else:
            passes.append(True)

    # print("check_pass: " + str(passes))

    return user_utility_pass


# Create an initial population of users
def generate_initial_population(starting_data_nonuser, starting_original_user_data, initial_noise, \
    population_starting_size):

    """
    Create an initial population of synthesized user data for our genetic algorithm (basically samples some X user data and add small noise to them)
    It is worth noting that the noise is added by first scaling a user's data by a small float value, and then adding it as noise to another user.

    Args:
        - starting_data_nonuser: An ndarray of the original data that does NOT include the user who's data we are attempted to synthesize
        - starting_original_user_data: An ndarray of the original data for a particular user we are synthesizing data for
        - initial_noise: a small float value used to scale a user's data, to be used as noise
        - population_starting_size: How much data we create for a particular user, used as the starting population for a user's data. (i.e. if value is 10, that means
            we have 10 synthesized ndarrays of data for a particular user)

    Returns:
        - initial_population: A list of ndarrays that are the initial synthesized data for a particular user
    """

    initial_population = []

    # Randomly take X users, add them as noise to the user
    random_user_idxs = np.random.choice(starting_data_nonuser.shape[0], \
        population_starting_size, replace=False)
    # print(random_user_idxs)
    # Then scale them to a certain amount (i.e. take them as 0.05 noise)
    for random_user_idx in random_user_idxs:
        population_item = starting_original_user_data + \
            (initial_noise*starting_data_nonuser[random_user_idx:random_user_idx+1])
        # print(population_item.shape)
        initial_population.append(population_item)

    return initial_population


# create children from the parents
def crossover_parents(current_population, population_max_size):

    """
    Used to merge two different synthesized data for a particular user in a genetic algorithm.  Crossover/child-making is performed by
      taking half of all feature data from one synthesized data, and half from another, and creating a new synthesized data from it.

    Args:
        - current_population: A list of synthesized data for a particular user (each one represents potential synthesized data for this user)
        - population_max_size: How many synthesized data we can create for one user (determines how many crossovers/children we create )

    Returns:
        - new_population: A list of ndarrays that are the synthesized data for a particular user, now including some crossover/children members
    """

    new_population = []
    new_population.extend(current_population)

    current_population_size = len(current_population)
    print("crossparents: pop size: " + str(current_population_size))
    num_children = population_max_size - current_population_size

    # Iterate through all the children we need to create
    for i in range(num_children):

        # Randomly select a pair of parents
        parents = np.random.choice(current_population_size, 2)
        parents = [current_population[parents[0]], current_population[parents[1]]]

        # Get some dimensions
        _, max_seq_len, num_features = parents[0].shape

        # Randomly select a set of features to take from each parent
        feature_idxs_parent1 = np.random.choice(num_features, num_features//2, replace=False)

        # Though the child is initially parent2, they have 35 features substituted from parent 1
        child = parents[1]
        child[:,:,feature_idxs_parent1] = parents[0][:,:,feature_idxs_parent1]

        new_population.append(child)

    return new_population


# Add noise to a population
def population_add_noise(current_population, starting_data_nonuser, noise_scale, \
    population_max_size):

    """
    Used to mutate members of the population (basically, add more noise to each synthesized data candidate for a particular user)

    Args:
        - current_population: A list of synthesized data for a particular user (each one represents potential synthesized data for this user)
        - starting_data_nonuser: An ndarray of the original data that does NOT include the user who's data we are attempted to synthesize
                here, it is used to add noise to synthesized data, since our definition of 'noise' is scaling another user's data and adding it.
        - noise_scale: a float value used to scale a user's data, to be used as noise
        - population_max_size: How many synthesized data we can create for one user

    Returns:
        - current_population: A list of ndarrays that are the synthesized data for a particular user, prior to mutation
        - new_population: A list of ndarrays that are the synthesized data for a particular user, after mutation
    """

    new_population = []

    population_size = len(current_population)
    random_user_idxs = np.random.choice(starting_data_nonuser.shape[0], population_size)

    # Update current population to have population_max_size elements

    # Then scale them to a certain amount (i.e. take them as 0.05 noise)
    for i, random_user_idx in enumerate(random_user_idxs):
        population_item = current_population[i] + \
            (noise_scale*starting_data_nonuser[random_user_idx:random_user_idx+1])
        new_population.append(population_item)

    return current_population, new_population


# Check fitness of a population
def population_check_fitness(current_population, chosen_feature_idxs, feature_predictors, \
    one_step_predictors, feature_pred_scores_ori_user, one_step_scores_ori_user, \
        feature_gt_user, one_step_gt_user, population_max_size):

    """
    Used to prune our synthesized data candidates for each user.  To check fitness, we compare the synthesized data against
     the utility thresholds, and return the synthesized data candidates that actually pass the thresholds.

    Args:
        - current_population: A list of synthesized data for a particular user (each one represents potential synthesized data for this user)
        - chosen_feature_idxs:  Feature indexes we are using for feature prediction
        - feature_predictors: List of TF Keras models for performing feature prediction
        - one_step_predictors: List of TF Keras models for performing one-step-ahead prediction
        - feature_pred_scores_ori_user: Feature predictions for a particular user (list of predictions for several feature indexes)
            These predictions are predicted from the original data
        - one_step_scores_ori_user: one-step-ahead predictions for a particular user
            These predictions are predicted from the original data
        - feature_gt_user: feature prediction labels for a user, for a particular set of features
            These are the ground truth values for the original data
        - one_step_gt_user: one-step-ahead labels for a particular user
            These are the ground truth values for the original data
        - population_max_size: How many synthesized data we can create for one user

    Returns:
        - selected_population: A list of synthesize data that have passed the fitness test (i.e. passed all utility thresholds)
        - all_failed: True if all synthesized data candidates for this user have failed, otherwise return False.
    """

    fitness_tracker = []
    selected_population = []
    all_failed = False  # Did all members of the population fail?

    print("p check fitness: current pop size: " + str(len(current_population)))

    for synthesized_user_data in current_population:

        # Score the synthesized data
        feature_pred_scores_syn_user, one_step_scores_syn_user = \
            get_scores(synthesized_user_data, chosen_feature_idxs, feature_predictors, one_step_predictors)

        # Flatten out predictions
        # feature_pred_scores_syn_user = np.squeeze(feature_pred_scores_syn_user)
        # one_step_scores_syn_user = np.squeeze(one_step_scores_syn_user)

        # print("Sizes: ")
        # print(feature_pred_scores_syn_user[0].shape)
        # print(one_step_scores_syn_user[0].shape)
        # print(feature_pred_scores_ori_user[0].shape)
        # print(one_step_scores_ori_user[0].shape)

        # Check if this user passes the test
        syn_pass = check_if_pass(feature_pred_scores_ori_user, one_step_scores_ori_user, \
            feature_pred_scores_syn_user, one_step_scores_syn_user, feature_gt_user, one_step_gt_user)


        fitness_tracker.append((synthesized_user_data, syn_pass))

    # Sort the fitness tracker.  Be sure to have best first.
    fitness_tracker = sorted(fitness_tracker, key=lambda x : x[1])[::-1]

    # How many passed?  If fewer than the max_size passed, then
    #   we take more samples
    num_passes = sum([x[1] for x in fitness_tracker])
    print("p check fitness passes: " + str(num_passes))
    if num_passes > 0:
        missing_members = population_max_size - num_passes
        fitness_tracker = fitness_tracker[:num_passes]

        # Get the passing population
        selected_population = [x[0] for x in fitness_tracker]
        # Take random selections from the selected_population
        if missing_members > 0:
            repeated_population = np.random.choice(np.arange(len(selected_population)),\
                missing_members)
            repeated_population = [selected_population[i] for i in repeated_population]
            selected_population.extend(repeated_population)
    else: # No passes
        selected_population.extend(current_population[:population_max_size])
        all_failed = True

    return selected_population, all_failed




# Run repetitive algorithm for a single user (100, 71)
#  Goal is to progressively add noise to an original user
#  We have an initial population (where the expectation is that each candidate can still)
#   pass all the requirements
#  We then progressively add noise until we reach the max iterations (or we've been failing for a while)
def user_repetitive_algorithm(feature_pred_scores_ori_user, one_step_scores_ori_user, \
    feature_gt_user, one_step_gt_user, num_users, feature_predictors, one_step_predictors, chosen_feature_idxs, \
    starting_original_user_data, starting_data_nonuser):

    """
    Runs the genetic algorithm for a particular user to determined the best synthesized data for this user (i.e. passes the utility thresholds and has a large amount of noise added.)
    More specifically, it requires several steps:
        - Create the initial population of synthesize data
        - Check if this initial population passes the fitness test
        - Then loop:
            Crossover candidates (creating children candidates )
            Mutate all candidates
            Check fitness again
        - Until we reach some maximum number of generations (or all candidates fail the fitness test.)

    Args:
        - feature_pred_scores_ori_user: Feature predictions for a particular user (list of predictions for several feature indexes)
            These predictions are predicted from the original data
        - one_step_scores_ori_user: one-step-ahead predictions for a particular user
            These predictions are predicted from the original data
        - feature_gt_user: feature prediction labels for a user, for a particular set of features
            These are the ground truth values for the original data
        - one_step_gt_user: one-step-ahead labels for a particular user
            These are the ground truth values for the original data
        - feature_predictors: List of TF Keras models for performing feature prediction
        - one_step_predictors: List of TF Keras models for performing one-step-ahead prediction
        - chosen_feature_idxs:  Feature indexes we are using for feature prediction
        - starting_original_user_data: An ndarray of the original data for a particular user we are synthesizing data for
        - starting_data_nonuser: An ndarray of the original data that does NOT include the user who's data we are attempted to synthesize

    Returns:
        - selected_population: A list of synthesize data that have passed the fitness test (i.e. passed all utility thresholds)
        - failout: True if all synthesized data candidates for this user have failed during running the genetic algorithm, otherwise return False
        - initialfailout: True if the initial population of data candidates for this user have failed, prior to even running the genetic algorithm.
        - current_iter: The iteration/generation that the algorithm stopped at.
    """

    # synthesized_user_data = starting_original_user_data

    max_iter = 10
    current_iter = 0
    synthetic_pass = False
    noise_scale = 0.000000001
    noise_increments = 0.2

    population_starting_size = 5
    population_max_size = 5
    population_selection_size = 2

    max_num_fails = 5
    num_fails = 0

    failout = False
    initialfailout = False

    # Have to make sure at least some members of the initial population pass fitness

    # Create initial population
    current_population = \
        generate_initial_population(starting_data_nonuser, starting_original_user_data, \
            noise_scale, population_starting_size)

    # current_population = [starting_original_user_data for i in range(population_starting_size)]

    # Check fitness
    selected_population, all_failed = \
        population_check_fitness(current_population, chosen_feature_idxs, feature_predictors, \
        one_step_predictors, feature_pred_scores_ori_user, one_step_scores_ori_user, \
            feature_gt_user, one_step_gt_user, population_selection_size)

    if all_failed:
        print("Initial population failed!")
        initialfailout = True
    else:
        print("Initial population passed!")

    #  At this point, some members of the population MUST have passed the fitness check.


    # While we are within the iterations and we haven't passed the synthetic data check
    while current_iter < max_iter:

        if num_fails > max_num_fails:
            failout = True
            break

        print("Iter: " + str(current_iter))
        print("Noise: " + str(noise_scale))

        # Create children from the selected population
        current_population = crossover_parents(selected_population, population_max_size)

        # Mutate the population
        old_population, current_population, = \
            population_add_noise(current_population, starting_data_nonuser, noise_scale, \
                population_max_size)

        # Check fitness
        selected_population, all_failed = \
            population_check_fitness(current_population, chosen_feature_idxs, feature_predictors, \
            one_step_predictors, feature_pred_scores_ori_user, one_step_scores_ori_user, \
                feature_gt_user, one_step_gt_user, population_selection_size)

        # If none of the population succeeds
        if all_failed:
            num_fails += 1
            print("fail " + str(num_fails) + " at iter=" + str(current_iter))
            # Make sure we return the current population to the old population
            selected_population = old_population
            # Decrease our noise
            noise_scale = noise_scale / 2
        else:  # If we have some success - add noise
            noise_scale += noise_increments # Add to the noise increments

        current_iter += 1 # Add to the iterations

    # If the initial population failed AND we failed out, return no noise
    if all_failed and initialfailout:
        selected_population = [starting_original_user_data for j in range(population_max_size)]

    return selected_population, failout, initialfailout, current_iter

# Get closest user datapoint to a cluster centroid
def get_best_cluster_representative(user_ids_in_cluster, user_utility_values):

    """
    When clustering the users by their feature values (not one-step-ahead), we are trying to find the
      user closest to the cluster's center.  This makes more sense in the context of the @func generate_user_clusters

    Args:
        - user_ids_in_cluster: Which users ids are in this cluster (user data is identified by their index in the original data)
        - user_utility_values: List of corresponding feature values (for a set of feature indexes), which are ndarrays, for each user id.

    Returns:
        - best_representative_user_id: Returns the index of the user whose utility ndarray is closest to the center for a particular set of values.
    """

    # Get utility values for all user ids in this cluster
    cluster_average_utility_value = user_utility_values[user_ids_in_cluster]
    # Get centroid (I'll define as the average user)
    cluster_average_utility_value = cluster_average_utility_value.mean(axis=0)

    # Get distance for this average to all other users
    distances = []  # tuple of (user id, distance to centroid)
    for user_id in user_ids_in_cluster:
        user_utility_val = user_utility_values[user_id]
        distance = np.linalg.norm(user_utility_val - cluster_average_utility_value)
        distances.append((user_id, distance))

    # Sort the distances
    distances = sorted(distances, key=lambda x: x[1])
    best_representative_user_id = distances[0][0]

    return best_representative_user_id


# Generate clusters of users
def generate_user_clusters(num_clusters, feature_gt, one_step_gt, num_users, starting_data):

    """
    A bit of context: we are creating user clusters because we can't run the genetic algorithm for all users in the dataset - that would take far too long.
    So instead, we opt to run the genetic data on a few users, whose data are representive of a cluster of other users - that way, we can apply the same
      synthesization/transofrmation for this user's data to all other users in the same cluster.
    The goal of this function is to create these user clusters, where clustering is done by K-means for some subset of feature values for each user.

    Args:
        - num_clusters: How many clusters we want
        - feature_gt: All data for a particular feature index
        - one_step_gt: All data for the last time-step (here we didn't end up using it)
        - num_users: How many users are in the dataset
        - starting_data: Basically, the original preprocessed data.

    Returns:
        - best_cluster_representatives: A set of user ids that best represent each cluster.  List of tuples (cluster id, user id)
        - cluster_predictions: A list that is the same length as the number of users, where each item is the cluster id corresponding to that user.

    """

    user_utility_values = []

    # Iterate through every user, and get each utility prediction, and concatenate the data.
    for user_idx in range(num_users):

        user_data = []

        # First, aggregate all the prediction data together
        for i, single_feature_gt in enumerate(feature_gt):  # 100, 1
            user_data.append(single_feature_gt[user_idx].flatten())
        # # Iterate through each one_step prediction, get RMSE
        # for i, single_step_gt in enumerate(one_step_gt):  # 99, 71

        # Then we create a single datapoint for this user
        user_data = np.concatenate(user_data)
        user_utility_values.append(user_data)

    # Reshape user data into ndarray
    user_utility_values = np.array(user_utility_values)

    user_utility_values = starting_data
    user_utility_values = np.reshape(user_utility_values, \
        [user_utility_values.shape[0], user_utility_values.shape[1] * user_utility_values.shape[2]])

    # print("USER UTILITY CLUSTERING SHAPE")
    # print(user_utility_values.shape)

    cluster_start_t = time.time()

    # Once we've created all the data, we apply clustering
    cluster_model = MiniBatchKMeans(n_clusters=num_clusters)
    cluster_model.fit(user_utility_values)
    cluster_predictions = cluster_model.predict(user_utility_values)
    print("Cluster Time: " + str(time.time() - cluster_start_t))

    # print("CLUSTER PREDICTIONS:")
    # print(cluster_predictions.shape) # Guessing to be (num_users, 1) or (num_users, )
    # print(cluster_predictions[:10])

    # Keep track - for each cluster ID, determine which users are part of it.
    cluster_user_dict = {}
    # For each unique cluster, get a user index
    for user_idx in np.arange(cluster_predictions.shape[0]):

        cluster_id = cluster_predictions[user_idx]
        if cluster_id not in cluster_user_dict:
            cluster_user_dict[cluster_id] = [user_idx]
        else:
            cluster_user_dict[cluster_id].append(user_idx)

    best_cluster_representatives = [] # tuples of (cluster_id, user_id)
    # For each cluster id, get the user closest to the centroid of that cluster
    for cluster_id in cluster_user_dict:
        user_ids_in_cluster = cluster_user_dict[cluster_id]
        user_representative_id = \
            get_best_cluster_representative(user_ids_in_cluster, user_utility_values)
        best_cluster_representatives.append( (cluster_id, user_representative_id) )

    return best_cluster_representatives, cluster_predictions

# Fill the dataset with the synthesized values
def get_synthetic_data(cluster_synthetic_noise, user_clusters, ori_data):

    """
    Once we have run the genetic algorithm for all cluster representatives, we take the transformation/noise for each
    representative user and apply it to all other users' data in the same cluster.

    Args:
        - cluster_synthetic_noise: A dictionary of {cluster id : ndarray of noise }, so we can apply the same noise to all users in a cluster.
        - user_clusters: A list that is the same length as the number of users, where each item is the cluster id corresponding to that user.
        - ori_data: Basically, the original preprocessed data.

    Returns:
        - ori_data: Original preprocessed data with noise added (should've been named synthesized_data)

    """

    # Iterate through the users of the dataset
    for user_idx in np.arange(ori_data.shape[0]):

        # Get the cluster id of this user
        cluster_id = user_clusters[user_idx]
        # Get the cluster noise for this cluster id
        cluster_noise = cluster_synthetic_noise[cluster_id]

        # Apply the cluster noise to this user
        ori_data[user_idx] += cluster_noise

    return ori_data

# Evaluate synthesized data scores against original
def generate_sythetic_dataset(feat_pred_ori, one_step_pred_ori, \
    feature_gt, one_step_gt, num_users, feature_predictors, one_step_predictors, chosen_feature_idxs, \
        ori_data, num_clusters):

    """
    First, generates a set of clusters for each user (reason mentioned in @func generate_user_clusters)
    Then, iterate through the user representatives for each cluster, and run the genetic algorithm to determine
      the noise to be added for each user representative's data.
     Then, apply the noise for all users in the same cluster to get our final synthesized data.

    Args:
        - feature_pred_scores_ori: Feature predictions (list of predictions for several feature indexes)
            These predictions are predicted from the original data
        - one_step_scores_ori: one-step-ahead predictions
            These predictions are predicted from the original data
        - feature_gt: feature prediction labels, for a particular set of features
            These are the ground truth values for the original data
        - one_step_gt: one-step-ahead labels
            These are the ground truth values for the original data
        - num_users: Number of users in the dataset
        - feature_predictors: List of TF Keras models for performing feature prediction
        - one_step_predictors: List of TF Keras models for performing one-step-ahead prediction
        - chosen_feature_idxs:  Feature indexes we are using for feature prediction
        - ori_data: Basically, the original preprocessed data.
        - num_clusters: How many clusters we want, in order to cluster users together.

    Returns:
        - synthetic_data: Final synthesized data for our hider

    """

    # Keep track of how many users are within bounds
    full_fails = 0
    initial_fails = 0

    fail_dict = {}

    synthetic_data = ori_data

    cluster_ids = {}  # Cluster IDs are keys, values are user indexes
    # Generate the clusters of users
    best_user_representatives, user_clusters = \
        generate_user_clusters(num_clusters, feature_gt, one_step_gt, num_users, ori_data)

    # Dictionary that stores the synthetic user data for anyone in a cluster
    cluster_synthetic_noise = {}

    # Get all the tuples of users representing each cluster
    #  (cluster_id, user_index)
    # chosen_user_indexes = list(cluster_ids.items())

    print("Generating data!")

    # Iterate through each user
    # for user_idx in np.arange(num_users):
    for i, tuple in enumerate(best_user_representatives):

        cluster_id = tuple[0]
        user_idx = tuple[1]

        print("Generating data for user " + str(i) + " out of " + str(len(best_user_representatives)))

        # Get values for this user
        feature_pred_scores_ori_user = [x[user_idx] for x in feat_pred_ori]
        one_step_scores_ori_user = [x[user_idx] for x in one_step_pred_ori]
        feature_gt_user = [x[user_idx] for x in feature_gt]
        one_step_gt_user = [x[user_idx] for x in one_step_gt]

        # Get current user data, and other user data (not current user)
        starting_data_user = ori_data[user_idx]
        starting_data_nonuser = ori_data[[x for x in np.arange(num_users) if x != user_idx]]
        starting_data_user = np.expand_dims(starting_data_user, 0)

        # Perform genetic alg for users
        start_t = time.time()
        user_data_candidates, failout, initialfailout, end_iter = user_repetitive_algorithm(feature_pred_scores_ori_user, one_step_scores_ori_user, \
            feature_gt_user, one_step_gt_user, num_users, feature_predictors, one_step_predictors, chosen_feature_idxs, \
            starting_data_user, starting_data_nonuser)
        print("User repetitive time: " + str(time.time() - start_t))

        if failout:
            full_fails += 1
        if initialfailout:
            initial_fails +=1

        # Add to dictionary
        if end_iter not in fail_dict:
            fail_dict[end_iter] = 1
        else:
            fail_dict[end_iter] += 1

        # Choose a random user data candidate
        user_data_candidate = user_data_candidates[0]

        # Determine the noise to add for this cluster
        cluster_noise = starting_data_user - user_data_candidate
        cluster_noise = cluster_noise.squeeze()

        # # If you want 0 noise as each cluster
        # cluster_noise = np.zeros(starting_data_user.shape)
        cluster_synthetic_noise[cluster_id] = cluster_noise.squeeze()

    # Get synthetic data from cluster ids and predicted clusters
    synthetic_data = get_synthetic_data(cluster_synthetic_noise, user_clusters, ori_data)

    # How many users fully failed?
    print("Full Fails: " + str(full_fails) + "/" + str(len(best_user_representatives)))
    print("Initial population Fails: " + str(initial_fails) + "/" + str(len(best_user_representatives)))
    print("fail dict: " + str(fail_dict))

    return synthetic_data



# Goal is to start fully synthesizing a dataset
def hider(input_dict: Dict) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:

    """
    Simply calls our actual data synthesis process, @func generate_sythetic_dataset

    Args:
        - input_dict: Dictionary of seed, input_data, and padding_mask

    Returns:
        - synthetic_data: Final synthesized data for our hider
        - padding_mask: Simply copies the padding mask from the input

    """

    seed = input_dict["seed"]  # Random seed provided by the competition, use for reproducibility.
    input_data = input_dict["data"]  # Input data, shape [num_examples, max_seq_len, num_features].
    padding_mask = input_dict["padding_mask"]  # Padding mask of bools, same shape as data.

    n_users, max_seq_length, n_features = input_data.shape

    # Number of feature predictors to use
    n_feature_predictors = 20
    num_clusters = 500

    # Get NaN indices
    nan_indices = np.argwhere(np.isnan(input_data))
    # Get user indexes, max seq len indexs, feat indexes
    nan_indices_users = [x[0] for x in nan_indices]
    nan_indices_seq = [x[1] for x in nan_indices]
    nan_indices_feats = [x[2] for x in nan_indices]

    # Get imputed data
    # _, input_data = preprocess_data(input_data, padding_mask)
    input_data = np.nan_to_num(input_data)

    # Train the utility predictors
    start_t = time.time()
    feature_predictors, one_step_predictors, chosen_feature_idxs, feature_gt, one_step_gt = \
        train_utility(input_data, n_users, max_seq_length, n_features, n_feature_predictors)
    print("Train time for utility functions: " + str(time.time() - start_t))

    # Get the scores for this original data
    feature_pred_scores_ori, one_step_scores_ori = \
        get_scores(input_data, chosen_feature_idxs, feature_predictors, one_step_predictors)


    # Evaluate synthesized scores against ori scores
    start_t = time.time()
    synthetic_data = generate_sythetic_dataset(feature_pred_scores_ori, one_step_scores_ori, \
        feature_gt, one_step_gt, n_users, \
        feature_predictors, one_step_predictors, chosen_feature_idxs, input_data, num_clusters)
    print("Dataset generation time: " + str(time.time() - start_t))

    # Add back all the NaNs
    synthetic_data[nan_indices_users, nan_indices_seq, nan_indices_feats] = np.nan


    print("final shape: " + str(synthetic_data.shape))

    return synthetic_data, padding_mask


# Another way to do it: For each user, get K candidates.  At the end, search through all users'
#  candidates and find the candidates that have minimum distance from each other.


# You ended up using the original data for the user_utility_values (maybe featurewise is better?)
# Changed feature distance bounds to 3

# If the cluster noise method doesn't work well, it's hard to tell why
#  maybe it's because we don't have enough clusters to accurately represent users
#  , maybe it's because our bounds are too high,
#  maybe its because we don't have enough predictors.
