"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar,
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data,"
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------

(1) data_preprocess: Load the data and preprocess for 3d numpy array
(2) imputation: Impute missing data using bfill, ffill and median imputation
"""

## Necessary packages
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# Plot the values for a feature
#  assumes a flattened np array that we can plot.
def plot_values(feature_values, feature_name):

    # We have to count the number of values for each value
    feature_val_count = {}
    for x in feature_values:
        if np.isnan(x):
            continue
        if x in feature_val_count:
            feature_val_count[x] += 1
        else:
            feature_val_count[x] = 1

    # Remove the largest and smallest values
    feature_values_max = max(feature_val_count.keys())
    feature_values_min = min(feature_val_count.keys())
    feature_val_count.pop(feature_values_max)
    feature_val_count.pop(feature_values_min)

    feature_values = list(feature_val_count.keys())


    feature_counts = list(feature_val_count.values())

    # Values are x axis, counts are y axis
    plt.scatter(feature_values, feature_counts, s=1)
    # Add title and axis names
    plt.title("Feature: " + str(feature_name))
    plt.xlabel('Feature Values')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join("plot_data", str(feature_name) + '.png'))
    plt.close()



def data_preprocess(file_name, max_seq_len):
  """Load the data and preprocess for 3d numpy array.

  Args:
    - file_name: CSV file name
    - max_seq_len: maximum sequence length

  Returns:
    - processed_data: preprocessed data
  """

  # Load data
  ori_data = pd.read_csv(file_name)

  # Plot each feature
  # print("Plotting Feature Value Frequencies!")
  # for feat_idx in np.arange(ori_data.shape[1][:2]):  # Ignore admission ID and time
  #     plot_values(ori_data.to_numpy()[:,feat_idx], ori_data.columns[feat_idx])


  # Take half the dataset since it is too large to fit all at once
  # ori_data = ori_data.head(ori_data.shape[0]//2)
  print(ori_data.shape)
  # example_data = ori_data.head(1)
  # feature_ids = feature_ids.insert(0, "time")
  # random_user_orig_data = pd.DataFrame(old_ori_data[0])
  # random_user_orig_data.columns = feature_ids
  # random_user_orig_data.to_csv("data/example_orig.csv")

  # Parameters
  uniq_id = np.unique(ori_data['admissionid'])
  no = len(uniq_id)
  dim = len(ori_data.columns) - 1
  # print(ori_data.shape)
  median_vals = ori_data.median()
  # print(median_vals.shape)

  # Preprocessing
  scaler = StandardScaler()
  scaler.fit(ori_data)
  # print(scaler.mean_.shape)
  # asdf

  # Output initialization
  processed_data = np.zeros([no, max_seq_len, dim])

  # For each uniq id
  for i in tqdm(range(no)):
    # Extract the time-series data with a certain admissionid
    idx = ori_data.index[ori_data['admissionid'] == uniq_id[i]]
    curr_data = ori_data.iloc[idx]

    # if i < 2:
    #     curr_data_csv = pd.DataFrame(curr_data)
    #     curr_data_csv.columns = ori_data.columns
    #     curr_data_csv.to_csv("data/example_preprocessed" + str(i) + "-1.csv")

    # Preprocess time
    curr_data['time'] = curr_data['time'] - np.min(curr_data['time'])
    # plot_values(ori_data.to_numpy()[:,1], ori_data.columns[1])

    # if i < 2:
    #     curr_data_csv = pd.DataFrame(curr_data)
    #     curr_data_csv.columns = ori_data.columns
    #     curr_data_csv.to_csv("data/example_preprocessed" + str(i) + "-2.csv")

    # Impute missing data
    curr_data = imputation(curr_data, median_vals)

    # if i < 2:
    #     curr_data_csv = pd.DataFrame(curr_data)
    #     curr_data_csv.columns = ori_data.columns
    #     curr_data_csv.to_csv("data/example_preprocessed" + str(i) + "-3.csv")

    # MinMax Scaling
    curr_data = scaler.transform(curr_data)

    # if i < 2:
    #     curr_data_csv = pd.DataFrame(curr_data)
    #     curr_data_csv.columns = ori_data.columns
    #     curr_data_csv.to_csv("data/example_preprocessed" + str(i) + "-4.csv")

    # Assign to the preprocessed data (Excluding ID)
    curr_data = np.array(curr_data)
    curr_no = len(curr_data)
    if curr_no >= max_seq_len:
      processed_data[i, :, :] = curr_data[:max_seq_len, 1:]
    else:
      processed_data[i, -curr_no:, :] = (curr_data)[:, 1:] # Append to end

  return processed_data, uniq_id, ori_data.columns


def imputation(curr_data, median_vals):
  """Impute missing data using bfill, ffill and median imputation.

  Args:
    - curr_data: current pandas dataframe
    - median_vals: median values for each column

  Returns:
    - imputed_data: imputed pandas dataframe
  """

  # Backward fill
  imputed_data = curr_data.bfill(axis = 'rows')
  # Forward fill
  imputed_data = imputed_data.ffill(axis = 'rows')
  # Median fill
  imputed_data = imputed_data.fillna(median_vals)

  return imputed_data
