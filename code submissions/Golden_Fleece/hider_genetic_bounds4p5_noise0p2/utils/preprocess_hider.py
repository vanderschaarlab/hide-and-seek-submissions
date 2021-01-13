import numpy as np
import pandas as pd


# Optimize time buckets - determines intervals for our datapoints
#  Max seq length determines how many buckets we have
# Buckets is a list of pairs, where each pair is the interval bounds
#  Basically, here we do equal frequency binning
def get_time_buckets(all_time_values, max_seq_len):


    # Get the maximum and minimum values
    max_val = max(all_time_values)
    min_val = min(all_time_values)
    # Get number of values
    num_values = len(all_time_values)

    # Get the bin intervals - equal frequency binning
    bin_intervals = np.interp(np.linspace(0, num_values, max_seq_len + 1),
                     np.arange(num_values),
                     np.sort(all_time_values))

    # Convert intervals into pairs, replacing the first and last intervals with
    # positive and negative infinity (open bounds)
    bin_intervals[0] = -np.inf
    bin_intervals[-1] = np.inf
    time_buckets = [(bin_intervals[i], bin_intervals[i+1]) for i in np.arange(bin_intervals.shape[0]-1)]

    # print(len(time_buckets))  # Should be max_seq_len

    return time_buckets

# Get all the time values for all users
#  all_times is the column of the preprocessed data for all time data.
#   it should be (1000, 100), with many zeroes prepended for each user.
#  all_data is the complete input dataset (1000, 100, 71),
#  which is used to derive the original nonzero user data
def get_time_values(all_time_columns, all_data):

    # Remember, we have to ignore the prepended empty values
    #  except for the first zero prior to non-zero values
    # So instead we work backwards and append to a list
    user_datapoints = [] # actual original rows of user data (ignoring zero rows)
    all_time_values = [] # Time values for all users for all data points

    # Get all the time values
    for user_idx in np.arange(all_time_columns.shape[0]):
        # Count values backwards
        user_data = all_time_columns[user_idx]  # Should be (100,)
        starting_data_idx = user_data.shape[0]-1

        while True:

            # Current time value
            datapoint_time_val = user_data[starting_data_idx]
            if datapoint_time_val == 0.0 or starting_data_idx == 0:
                # all_time_values.append(datapoint_time_val)
                user_datapoints.append(all_data[user_idx, starting_data_idx:,:])
                break

            all_time_values.append(datapoint_time_val)
            starting_data_idx -= 1

    return all_time_values, user_datapoints


# Dump datapoints into each time bin, while retaining user information
def dump_datapoints_into_timebins(time_buckets, user_datapoints, input_data_shape):

    # For each interval (total of max_seq_len intervals), record all user datapoints landing in it.
    interval_dict = {}

    user_time_indexes = {} # for each user, record where their data was placed.

    reorganized_data = np.zeros(input_data_shape)  # Reorganized data based on input data, which is (num_users, max_seq_len, num_features)

    # For each time bucket, check which datapoints for each user land in it.
    #  If multiple datapoints for a user land in the same bucket, take the average.
    for bucket_index, bucket in enumerate(time_buckets):

        bucket_t_start = bucket[0]
        bucket_t_end = bucket[1]

        all_user_landed_data = [] # Collect all user datapoints that lands in this time bucket.


        # Iterate through each user
        for user_idx, user_data in enumerate(user_datapoints):

            # Collect data for this user that lands in the bucket - if multiple, average them.
            user_landed_data = []

            # Append to the user time indexes
            if user_idx not in user_time_indexes:
                user_time_indexes[user_idx] = [None for x in time_buckets]

            # Iterate through each datapoint for this user
            for data_row_idx in np.arange(user_data.shape[0]):

                # Get the time value for this user data
                user_t_val = user_data[data_row_idx][0]

                # Check if the time value falls in the bucket. If so, add it
                #  to the landed data.  Also add it to the reorganized data.
                if user_t_val >= bucket_t_start and user_t_val <= bucket_t_end:
                    user_landed_data.append(user_data[data_row_idx])

                    # Mark this time bucket
                    user_time_indexes[user_idx][bucket_index] = bucket_index

                    # Add to reorganized data where it belongs
                    reorganized_data[user_idx,bucket_index,:] = user_data[data_row_idx]



            # Take the average of this user's landed data.
            user_landed_data = np.array(user_landed_data)
            # print(user_landed_data.shape)
            average_landed_data = user_landed_data
            if len(user_landed_data.shape) > 1:
                average_landed_data = np.mean(user_landed_data, axis=0)

            # print(average_landed_data)
            # if any([np.isnan(x) for x in average_landed_data]):
            #     print(user_data)

            # Append this user to all landed data.
            if user_landed_data != []:
                all_user_landed_data.append((user_idx, average_landed_data))

        # Append the all_user_landed data for this time bucket dict
        interval_dict[bucket_index] = all_user_landed_data

    # Update all the bucket indexes to ignore Nones
    for user_idx in user_time_indexes.keys():
        indexes = user_time_indexes[user_idx]
        indexes = [x for x in indexes if x]
        user_time_indexes[user_idx] = indexes

    return interval_dict, user_time_indexes, reorganized_data


# Impute values by forward and backward filling
# Data is of shape (num_users, max_seq_length, num_features)
#  Empty values are nan.
def impute(data):

    imputed_data = np.empty(data.shape)
    # For every user, we impute data in.
    for user_idx in np.arange(data.shape[0]):

        current_user_data = data[user_idx] # (max_seq_len, num_features)
        # Pandas ffill
        current_user_data = pd.DataFrame(current_user_data)
        current_user_data = current_user_data.ffill(axis = 'rows')
        current_user_data = current_user_data.bfill(axis = 'rows')
        current_user_data = current_user_data.to_numpy()

        imputed_data[user_idx] = current_user_data

    return imputed_data


# Reformat data (i.e. pulls correct rows for each user, and prepends with zeroes)
def reformat_data(user_time_indexes, generated_data):

    new_generated_data = np.zeros(generated_data.shape)

    # For every user, pull only the rows we are interested in
    for user_idx in np.arange(generated_data.shape[0]):

        rows_of_interest = user_time_indexes[user_idx]
        pulled_data = generated_data[user_idx][rows_of_interest]

        new_generated_data[user_idx, -len(rows_of_interest):, :] = pulled_data

    return new_generated_data

# Open up minitest files and check best recorded results
def check_results(minitest_filepaths):

    best_results = [] # (Architecture number, score)

    for minitest_file in minitest_filepaths:

        # Get the arch number
        arch_number = minitest_file.split(",")[-2].split(")")[0].strip()
        # print(arch_number)

        with open(minitest_file, "r") as f:
            score_data = f.read()
            score = score_data.split("(")[-1].split(")")[0]
            # print(score)
            # print(arch_number)
            best_results.append((int(arch_number), float(score)))

    best_results = sorted(best_results, key=lambda x: x[1])
    return best_results[0][0]
