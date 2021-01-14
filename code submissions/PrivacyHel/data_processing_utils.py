"""
Specific implementations of data processing steps.

Used by `DataTransfomer` instances in `data_pipeline_transformers.py`.
"""

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys

def remove_outliers(data, data_bounds, padding_mask, replace_with_nans=False):
    '''Remove outliers from the data based on the given upper and lower bounds.

    Args:
        data : numpy array of shape (num_sequences x sequence len x num_features) containing the data with possible outliers
        data_bounds : numpy array of shape (num_features x 2) containing lower (1st col [:,0]) and upper (2nd col [:,1]) bounds for the data. If either is None or nan, the feature is skipped.
        padding_mask : a boolean mask of shape (num_data, max_seq_len) indicating whether (True) or not (False) a time step in a sequence is padded.
        replace_with_nans: replace values exceeding the bounds with nan instead clipping at the bounds
    Returns:
        modified numpy array with each data element clipped to the given bounds
    '''

    N,T,D = data.shape
    data = np.copy(data)

    assert D == data_bounds.shape[0], 'Number of bounds need to match number of features in the data!'
    assert data_bounds.shape[1] == 2, 'Bounds should be given as  [lower, upper]!'
    assert padding_mask.shape[0] == N, 'padding_mask shape should be (num_data, max_seq_len)'
    assert padding_mask.shape[1] == T, 'padding_mask shape should be (num_data, max_seq_len)'
    logging.debug('Starting outlier removal...')

    # clip non-padded (padding_mask value = False) values to given boundaries

    for i, bounds in enumerate(data_bounds):
        if replace_with_nans:
            outlier_mask = (data[:,:,i] < bounds[0]) | (data[:,:,i] > bounds[1])
            outlier_mask &= ~padding_mask # ignore padding
            if np.any(outlier_mask):
                logging.debug('values exceed bounds for feature {}; lower bound: {}, upper bound: {}'.format(i,*bounds))
            data[:,:,i][outlier_mask] = np.nan
        else:
            if np.sum(np.isnan(bounds)) == 0:
                data[:,:,i][~padding_mask] = np.clip(data[:,:,i][~padding_mask], a_min=bounds[0], a_max=bounds[1])


    logging.debug('Outliers removed.')
    return data


def remove_outliers_pandas(data, mod_col_names):
    '''Outlier remover using Pandas dataframes. You should probably use the Numpy version.'''

    import pandas as pd

    for i, name in enumerate(mod_col_names['feature_name']):
        logging.debug(name)
        logging.debug('lower bound: {}'.format(mod_col_names['lower_bound'].iloc[i] ))
        logging.debug('upper bound: {}'.format(mod_col_names['upper_bound'].iloc[i] ))

        if mod_col_names['lower_bound'].iloc[i] is not None and mod_col_names['upper_bound'].iloc[i] is not None:
            logging.debug('cur max:{}'.format(data[mod_col_names['column_id'].iloc[i]].max()))
            logging.debug('cur min:{}'.format(data[mod_col_names['column_id'].iloc[i]].min()))

            data[mod_col_names['column_id'].iloc[i]].clip(lower=mod_col_names['lower_bound'].iloc[i], \
                                                             upper=mod_col_names['upper_bound'].iloc[i], \
                                                             axis=0, inplace=True)
            logging.debug('new max:{}'.format(data[mod_col_names['column_id'].iloc[i]].max()))
            logging.debug('new min:{}'.format(data[mod_col_names['column_id'].iloc[i]].min()))
    return data




def Kalman_Gaussian_RW_filter(y, prior_mean, prior_std=2, obs_noise=2, latent_noise=1, first_obs_as_prior=True):
    '''
    Simple Gaussian random walk Kalman filter for 1d or 2d data. For 2d, take max of given measurements as the actual observation. This is meant as a quick solution for combining different temperature measurements.

    NOT USED

    Args:
        y : observations, possible including nans
        prior_mean :
        prior_std : prior standard deviation
        obs_noise : std for the observation model
        latent_noise : std for the latent model
        first_obs_as_prior : use first actual observations as prior for the mean
    Returns: m, p
        m : estimated mean 1d series
        p : estimated std 1d series
    '''

    logging.debug('KF got array of shape {}'.format(y.shape))

    if len(y.shape) > 1:
        multidim = True
    else:
        multidim = False
    n_x = y.shape[0]+1

    # set prior mean & var
    m = np.zeros(n_x) # set prior
    if first_obs_as_prior:
        try:
            if multidim:
                m[0] = np.nanmax(y,1)[np.isfinite(np.nanmax(y,1))][0]
            else:
                m[0] = y[np.isfinite(y)][0]
        except:
            m[0] = prior_mean
    else:
        m[0] = prior_mean
    p = np.zeros(n_x)
    p[0] = prior_std # set prior
    Q = latent_noise # latent state noise var
    R = obs_noise # observation noise var

    for i in tqdm(range(1,n_x)):
        # update prior mean
        m[i] = m[i-1]
        # update prior var
        p[i] = p[i-1] + Q
        # in case no obs just use current prior
        if multidim:
            if np.nanmax(y[i-1,:]) is not None and not np.isnan(np.nanmax(y[i-1,:])):
                    # update posterior mean
                    m[i] = m[i] + p[i]/(p[i] + R)*(np.nanmax(y[i-1,:]) - m[i])
                    # update posterior var
                    p[i] = p[i] - p[i]**2/(p[i] + R)
        else:
            if not np.isnan(y[i-1]):
                # update posterior mean
                m[i] = m[i] + p[i]/(p[i] + R)*(y[i-1] - m[i])
                # update posterior var
                p[i] = p[i] - p[i]**2/(p[i] + R)

    return m[1:], p[1:]




def competition_imputation(curr_data, median_vals):
  """
  This is the competition imputation code fitted to the pipeline.
  Impute missing data using bfill, ffill and median imputation.

  Args:
    - curr_data: list of numpy data arrays of shape T x D
    - median_vals: median values for each column

  Returns:
    - imputed_data: list of imputed numpy arrays
  """

  #N,T,D = curr_data.shape
  assert len(median_vals) == curr_data[0].shape[1] , 'Number of median vals need to match data dimensionality!'

  # currently slow&lazy implementation using pandas
  for i in range(len(curr_data)):
    pd_data = pd.DataFrame(curr_data[i])
    # Backward fill
    pd_data.bfill(axis = 'rows', inplace=True)
    # Forward fill
    pd_data.ffill(axis = 'rows', inplace=True)
    # Median fill
    pd_data.fillna(pd.Series(data=median_vals), inplace=True)
    curr_data[i] = pd_data.to_numpy()

  return curr_data


def remove_missing_features(self, padded_sequences, messages):

    '''
    Remove features that do not have any observations, i.e. those columns where
    all rows across all individuals are either padded values or NaN's.

    Args:
        padded_sequences: numpy array of shape (num_sequences x sequence len x number of CURRENT features)

        Messages:
        reads padding_mask
        reads orig_features
        reads/ emits current_features
        reads/ emits removed_features
        emits ind_removed_empty_columns

    Returns: numpy array without features that do not have any observations.

    NOTE, that mv_mask is not created.
    '''
    # remove columns of only missing values
    current_features = messages['current_features']
    removed_features = messages['removed_features']
    padding_mask = messages['padding_mask']
    orig_features = messages['orig_features']

    # check which FEATURES include only missing values and padded values and make a list
    col_remove = np.where(np.all(np.isnan(padded_sequences[~padding_mask]), axis = 0) == True)[0]

    # remove columns in the list from the array
    padded_sequences = np.delete(padded_sequences, col_remove, 2)
    if 'missing_feature_values' in messages:
        messages['missing_feature_values'] = np.delete(messages['missing_feature_values'], col_remove, 2)

    # names of the removed columns
    removed_columns = []

    # using the current features list, get the names of the removed columns
    for i in col_remove:
        removed_columns.append(current_features[i])

    # using the orig_features list,
    # get the indices of the removed columns in the original dimensions and
    # update removed_features dictionary.
    # update current_features list
    ind_removed_empty_columns = []

    for f in removed_columns:
        # get column index from the list of ORIGINAL features
        col_ind = orig_features.index(f)
        ind_removed_empty_columns.append(col_ind)
        # add name and column index of removed features to the dictionary
        # column indices of features are the values of the dictionary
        removed_features[f] = col_ind

        # update current_features to reflects the current features in the data
        current_features.remove(f)


    # add current_features list and removed_features dictionary to messages
    messages['current_features'] = current_features
    messages['removed_features'] = removed_features
    messages['ind_removed_empty_columns'] = ind_removed_empty_columns

    return padded_sequences, messages





##########################################
# Remove and return superfluous features #
##########################################

def remove_superfluous_features(padded_sequences, messages):

        '''Remove columns of superfluous features from the data based on a "features_to_remove" list.

        Args:
            padded_sequences: numpy array of shape (num_sequences x sequence len x number of CURRENT features)
            messages : pipeline messages
        Returns:
            messages and modified numpy array without columns of superfluous features
        '''

        #  verbose (boolean) are given in the message
        verbose = False
        if 'verbose' in messages:
            verbose = messages['verbose']

        # list of original features
        orig_features = messages['orig_features']

        # list of all features that are currently in the data
        current_features = messages['current_features']

        features_to_remove = ['map', 'be_abg','pc_over_peep', 'ps_over_peep', 'calcium_albc']

        # dict of removed features and their indices on ORIGINAL data
        removed_features = messages['removed_features']

        padding_mask = messages['padding_mask']
        padding_mask = np.ravel(padding_mask)

        if verbose:
            print('dimensions of the data: ', padded_sequences.shape)

        # For returning features after generation of synthetic data,
        # get intercept and coefficients for map and calcium_albc
        # from (robust) linear regression using HuberRegressor/ Linear Regression

        if verbose:
            print('\nFor returning map and calcium_albc values using regression, \
                  get intercept and coefficients of (robust) linear regression.')

        from sklearn.linear_model import HuberRegressor, LinearRegression

        # map ~ dbp + sbp

        # NaN values need to be excluded from the regression!

        # -> make 'regression subset of both dependent variables and
        # inependent variables so that rows with NaN's can be removed
        '''
        reg_data = np.zeros([padded_sequences.shape[0], padded_sequences.shape[1], 3])

        ind_dbp = current_features.index('dbp')
        ind_sbp = current_features.index('sbp')
        ind_map = current_features.index('map')

        # independent variables: dbp, sbp
        reg_data[:,:,0]= padded_sequences[:,:,ind_dbp]
        reg_data[:,:,1]= padded_sequences[:,:,ind_sbp]

        # dependent variable: map
        reg_data[:,:,2] = padded_sequences[:,:,ind_map]

        # reshape
        reg_data = reg_data.reshape(reg_data.shape[0] * reg_data.shape[1], reg_data.shape[2])

        # exclude rows with ouliers
        reg_data = reg_data[~np.isnan(reg_data).any(axis=1)]

        # fit robust regression model
        X = reg_data[:,:2]
        y = reg_data[:, 2]
        huber = HuberRegressor().fit(X, y)

        if verbose:
            print('\nHuberRegression: map ~ dbp + sbp')
            # score: coefficient of determination R^2 of the prediction
            print('huber score: ', huber.score(X, y))
            print("Huber coefficients:", huber.coef_)
            print("Huber intercept:", huber.intercept_)

        reg_coef_map = {'coef_dbp': huber.coef_[0], 'coef_sbp': huber.coef_[1],
                    'intercept': huber.intercept_}

        messages['reg_coef_map'] = reg_coef_map
        '''

        # in case dimensionality of the input data changes...
        if len(padded_sequences.shape) ==3:

            reg_data = np.zeros([padded_sequences.shape[0], padded_sequences.shape[1], 3])

            ind_dbp = current_features.index('dbp')
            ind_sbp = current_features.index('sbp')
            ind_map = current_features.index('map')

            # independent variables: dbp, sbp
            reg_data[:,:,0]= padded_sequences[:,:,ind_dbp]
            reg_data[:,:,1]= padded_sequences[:,:,ind_sbp]

            # dependent variable: map
            reg_data[:,:,2] = padded_sequences[:,:,ind_map]

            # reshape
            reg_data = reg_data.reshape(reg_data.shape[0] * reg_data.shape[1], reg_data.shape[2])


            # use padding_mask to exclude padded -1 values
            # before missing values are excluded

            # masked data array where padded values are masked out
            reg_data = reg_data[~padding_mask]


        elif len(padded_sequences.shape) ==2:
            # NOTE, masking is not used for 2D arrays

            reg_data = np.zeros([padded_sequences.shape[0], 3])

            ind_dbp = current_features.index('dbp')
            ind_sbp = current_features.index('sbp')
            ind_map = current_features.index('map')

            # independent variables: dbp, sbp
            reg_data[:,0]= padded_sequences[:,ind_dbp]
            reg_data[:,1]= padded_sequences[:,ind_sbp]

            # dependent variable: map
            reg_data[:,2] = padded_sequences[:,ind_map]

            # reshape
            #reg_data = reg_data.reshape(reg_data.shape[0] * reg_data.shape[1], reg_data.shape[2])


        # exclude rows with ouliers
        reg_data = reg_data[~np.isnan(reg_data).any(axis=1)]

        if verbose:
            print('shape of masked reg_data after excluding rows with missing values: ', reg_data.shape)


        assert np.all(np.isfinite(reg_data)), 'check that all values of reg_data are finite'

        '''
        # plot the data
        import matplotlib.backends.backend_pdf
        import matplotlib.pyplot as plt

        if verbose:
            def make_plot(plot_x, plot_y, title, x_label, y_label):
                print('making plot...')
                fig, ax1 = plt.subplots(figsize=(13, 13))
                ax1.scatter(plot_x, plot_y)
                ax1.set_title(title)
                ax1.set_ylabel(y_label)
                ax1.set_xlabel(x_label)

                return fig


            # empty list for plots
            figures = []

            # map and sbp
            fig = make_plot(reg_data[:, 1], reg_data[:, 2], 'map ~ sbp', 'sbp', 'map')

            figures.append(fig)

            fig = make_plot(reg_data[:, 0], reg_data[:, 2], 'map ~ dbp', 'dbp', 'map')

            figures.append(fig)

        '''


        # fit robust regression model
        X = reg_data[:,:2]
        y = reg_data[:, 2]


        try:
            regression = HuberRegressor().fit(X, y)
        except:
            regression = LinearRegression().fit(X, y)
            print('Regular linear regression is used')

        if verbose:
            print('\nRegression: map ~ dbp + sbp')
            # score: coefficient of determination R^2 of the prediction
            print('Regression score: ', regression.score(X, y))
            print("Regression coefficients:", regression.coef_)
            print("Regression intercept:", regression.intercept_)

        reg_coef_map = {'coef_dbp': regression.coef_[0], 'coef_sbp': regression.coef_[1],
                        'intercept': regression.intercept_}

        messages['reg_coef_map'] = reg_coef_map


        logging.debug('map ~ dbp + sbp regression score: {}'.format(regression.score(X, y)))


        # calcium_albc ~ calcium + albumin
        '''
        reg_data = np.zeros([padded_sequences.shape[0], padded_sequences.shape[1], 3])

        ind_calcium = current_features.index('calcium')
        ind_albumin = current_features.index('albumin')
        ind_calcium_albc = current_features.index('calcium_albc')


        # independent variables
            reg_data[:,:,0]= padded_sequences[:,:,ind_calcium]
            reg_data[:,:,1]= padded_sequences[:,:,ind_albumin]

            # dependent variable: map
            reg_data[:,:,2] = padded_sequences[:,:,ind_calcium_albc]

            # reshape
            reg_data = reg_data.reshape(reg_data.shape[0] * reg_data.shape[1], reg_data.shape[2])
        '''
        if len(padded_sequences.shape) ==3:
            reg_data = np.zeros([padded_sequences.shape[0], padded_sequences.shape[1], 3])

            ind_calcium = current_features.index('calcium')
            ind_albumin = current_features.index('albumin')
            ind_calcium_albc = current_features.index('calcium_albc')


            # independent variables
            reg_data[:,:,0]= padded_sequences[:,:,ind_calcium]
            reg_data[:,:,1]= padded_sequences[:,:,ind_albumin]

            # dependent variable: map
            reg_data[:,:,2] = padded_sequences[:,:,ind_calcium_albc]

            # reshape
            reg_data = reg_data.reshape(reg_data.shape[0] * reg_data.shape[1], reg_data.shape[2])

            # use padding_mask to exclude padded -1 values
            # before missing values are excluded
            reg_data = reg_data[~padding_mask]


        elif len(padded_sequences.shape) ==2:
            # NOTE, masking is not used for 2D arrays
            reg_data = np.zeros([padded_sequences.shape[0], 3])

            ind_calcium = current_features.index('calcium')
            ind_albumin = current_features.index('albumin')
            ind_calcium_albc = current_features.index('calcium_albc')


            # independent variables
            reg_data[:,0]= padded_sequences[:,ind_calcium]
            reg_data[:,1]= padded_sequences[:,ind_albumin]

            # dependent variable: map
            reg_data[:,2] = padded_sequences[:,ind_calcium_albc]

            if verbose:
                print('shape of reg_data: ', reg_data.shape)


        # exclude rows with ouliers
        reg_data = reg_data[~np.isnan(reg_data).any(axis=1)]

        if verbose:
            print('shape of masked reg_data after excluding rows with missing values: ', reg_data.shape)


        assert np.all(np.isfinite(reg_data)), 'check that all values of reg_data are finite'

        # fit robust regression model
        X = reg_data[:,:2]
        y = reg_data[:, 2]


        try:
            regression = HuberRegressor().fit(X, y)
        except:
            regression = LinearRegression().fit(X, y)
            print('Regular linear regression is used')



        if verbose:
            print('\nRegression: calcium_albc ~ calcium + albumin')
            print('Regression score: ', regression.score(X, y))
            print("Regression coefficients: ", regression.coef_)
            print("Regression intercept:", regression.intercept_)

        reg_coef_calcium_albc = {'coef_calcium': regression.coef_[0], 'coef_albumin': regression.coef_[1],
                    'intercept': regression.intercept_}

        messages['reg_coef_calcium_albc'] = reg_coef_calcium_albc

        logging.debug('calcium_albc ~ calcium + albumin regression score: {}'.format(regression.score(X, y)))

        #if verbose:
            # this should take paddings and missing values into account
            #print('Train data means (all columns):\n{}'.format(np.mean(np.mean(padded_sequences,0),0)))



        if verbose:
            print('\nget indices of superfluous features that are to be removed')


        # remove superfluous features
        col_remove = []

        # go through features to be removed and get indices of the features


        for f in features_to_remove:

            # get column index from the list of CURRENT features
            col_ind = current_features.index(f)
            col_remove.append(col_ind)
            if verbose:
                print('\nfeature_name : {}, column index in current data: {}'.format(f, col_ind))


        if verbose:
            print('\nshape of array before removing superfluous features: ',
                  padded_sequences.shape)

            print('\nfirst row of first individual: ')
            print(padded_sequences[0,0,:])

            print('\n80th row: ')
            print(padded_sequences[1,11,:])


        # slice off the columns specified in the 'col_remove' list
        padded_sequences = np.delete(padded_sequences, col_remove, 2)
        if ('missing_feature_values' in messages):
            mv_mask = messages['missing_feature_values']
            mv_mask = np.delete(mv_mask, col_remove, 2)
            messages['missing_feature_values'] = mv_mask


        if verbose:
            print('\nshape of array after removing superfluous features: ',
                  padded_sequences.shape)

            print('\nfirst row of first individual: ')
            print(padded_sequences[0,0,:])

            print('\n80th row: ')
            print(padded_sequences[1,11,:])

        if verbose:
            print('update "removed_features" dict and "current_features" list')

        # go through features to be removed and update dict and list
        for f in features_to_remove:

            # get column index from the list of ORIGINAL features
            col_ind = orig_features.index(f)

            if verbose:
                print('\nfeature_name : {}, column index in original data: {}'.format(f, col_ind))

            # add name and column index of removed features to the dictionary
            # column indices of features are the values of the dictionary
            removed_features[f] = col_ind

            # remove feature from the list of remaining features
            current_features.remove(f)



        # add removed_features dictionary to messages
        messages['removed_features'] = removed_features

        # add list of features without removed features to messages
        # -> this reflects the current features in the data
        messages['current_features'] = current_features

        if verbose:
            print('\nDictionary of superfluous features and their original \
                  column indices (removed_features) are saved to messages')
            print('\nList of remaining features (current_features) \
                  is saved to messages')



        return padded_sequences, messages






def create_array_for_returning_features(padded_sequences, messages):
        '''
        Create an array with same dimensions as the original data, and
        add columns from synthetic data to their original places and
        leave space for the removed columns so that they can be returned

        Args:
            padded_sequences: synthetic data (num_sequences x sequence len x num_features in generated data)
            messages : pipeline messages

        Returns:
            messages and modified numpy array where columns of synthetic data
            are in same locations as where the features were in the original
            data and columns for removed features are zeros -> removed
            feature can be added to their original locations

        '''

        verbose = False
        if 'verbose' in messages:
            verbose = messages['verbose']

        if verbose:
            print('\nshape of array before returning removed features: ',
                  padded_sequences.shape)

            print('\nfirst row of first individual: ')
            print(padded_sequences[0,0,:])

        # make array of zeros with original dimensions
        orig_dim = messages['original_dimensions']
        tmp_array = np.zeros(orig_dim)


        # get column indices of removed columns
        removed_features = messages['removed_features']


        # add columns from padded_sequences array to column indices that
        # do not belong to removed columns

        # indices of removed features in original data
        col_ind = list(removed_features.values())

        # indices of remaining(/current) features in original data
        remaining_col_ind = [i for i in range(orig_dim[-1]) if i not in set(col_ind)]

        # sort the indices to be ascending
        col_ind.sort()


        if verbose:
            print('\nmake tmp_array with original dimensions. Fill unchanged \
                   columns to their original locations in temp_arrary')


        # get unchanged columns till the first column for removed features
        tmp_array[:,:,:col_ind[0]] = padded_sequences[:,:,:col_ind[0]]


        # fill unchanged columns starting from previous column for removed
        # features till next column for removed features

        round = 0

        for i in range(len(col_ind)-1):
            if verbose:
                print('\ncurrent col_ind: ',col_ind[i])
            # Note that indiced DO NOT MATCH between tmp_array and
            # padded_sequences because padded_sequences doesn't have columns
            # of superfluous features -> corresponding indices of
            # padded_sequences are 1 smaller after each removed feature r

            # -> count rounds and substract the round number from indices of
            # padded_sequence because it does not contain superfluous features

            round += 1

            # if last feature
            if i == len(col_ind)-1:
                # fill unchanged columns that come after last column for removed features
                tmp_array[:,:,col_ind[i]+1:] = padded_sequences[:,:,col_ind[i]+1 -round:]

            # if not last feature
            else:

                # for columns between next column after i:th removed feature
                # and before i+1:th removed feature, add original columns
                tmp_array[:,:,col_ind[i]+1:col_ind[i+1]] = padded_sequences[:,:,col_ind[i]+1 - round :col_ind[i+1] -round]


        '''
        # check that columns of zero's, i.e. columns left for removed fetures,
        # have same column indices as the removed features have in the
        # data before their removal

        # indices of zeros in first row
        first_row = list(tmp_array[0,0,:])
        indices = [i for i, x in enumerate(first_row) if x == 0]


        if verbose:
            print('\ncolumn indices of removed superfluous in original data:')
            print(col_ind)
            print('\ncolumn indices of zeros in the first row of first individual, \
                  i.e. columns left for removed features or having zeros in synthetic data:')
            print(indices)
            print('\nfirst row of first individual after filling columns from synthetic data: ')
            print(tmp_array[0,0,:])
            print('\n12th row of second individual: ')
            print(tmp_array[1,11,:])

        '''

        padded_sequences = tmp_array

        if 'missing_feature_values' in messages:
            mv_mask = messages['missing_feature_values']
            new_mv_mask = np.zeros_like(tmp_array, dtype=np.bool)
            new_mv_mask[:,:,remaining_col_ind] = mv_mask
            messages['missing_feature_values'] = new_mv_mask

        return padded_sequences, messages





def return_superfluous_features(self, padded_sequences, messages):

        '''
        Add removed superfluous features to same locations as where they
        were in original data. Values of those features are calcultaed based
        on other features.


        Args:
            padded_sequences: output from create_array_for_returning_features
            -> synthetic data (num_sequences x sequence len x num_features in original data)

            Messages:
                reads orig_features
                reads removed_features
                reads padding_mask
                reads padding value
                reads reg_coef_map
                reads reg_coef_calcium_albc


        Returns:
            messages and modified numpy array where removed superfluous
            features are estimated and added to the data

        '''

        verbose = False
        if 'verbose' in messages:
            verbose = messages['verbose']

        # calculate removed superfluous features and add to an array
        # to their original locations

        # get features in the original data from the messages
        orig_features = messages['orig_features']

        # get column indices of removed columns in original data
        removed_features = messages['removed_features']


        padding_mask = messages['padding_mask']
        padding_value = messages['padding_value']

        # map

        # get coefficients of the regression
        reg_coef_map = messages['reg_coef_map']

        ind_dbp = orig_features.index('dbp')
        ind_sbp = orig_features.index('sbp')
        ind_map = orig_features.index('map')

        if verbose:
            print('\nCheck index of map to be same in list and dict')
            print('index from list: {}, index from dict: {}'
                  .format(ind_map, removed_features['map']))



        # map ~ dbp + sbp
        padded_sequences[:,:,ind_map][~padding_mask] = \
        reg_coef_map['coef_dbp'] * padded_sequences[:,:,ind_dbp][~padding_mask] + \
        reg_coef_map['coef_sbp'] * padded_sequences[:,:,ind_sbp][~padding_mask] + \
        reg_coef_map['intercept']

        padded_sequences[:,:,ind_map][padding_mask] = padding_value #padded_sequences[:,:,0][padding_mask]



        # be_abg
        # be_formula from literature = 0.93 * hco3 + 13.77*ph - 124.58
        ind_hco3 = orig_features.index('hco3')
        ind_ph = orig_features.index('ph')

        ind_be_abg = removed_features['be_abg']

        if verbose:
            print('\nCheck index of be_abg to be same in list and dict')
            print('index from list: {}, index from dict: {}'
                  .format(orig_features.index('be_abg'), ind_be_abg))

        padded_sequences[:,:, ind_be_abg][~padding_mask] = \
        0.93 * padded_sequences[:,:,ind_hco3][~padding_mask] + \
        13.77 * padded_sequences[:,:,ind_ph][~padding_mask] - 124.58

        padded_sequences[:,:,ind_be_abg][padding_mask] = padding_value # padded_sequences[:,:,0][padding_mask]



        # pc_over_peep
        ind_ppeak = orig_features.index('ppeak')
        ind_peep = orig_features.index('peep_set')

        ind_pc_over_peep = removed_features['pc_over_peep']

        if verbose:
            print('\nCheck index of pc_over_peep to be same in list and dict')
            print('index from list: {}, index from dict: {}'
                  .format(orig_features.index('pc_over_peep'), ind_pc_over_peep))

        padded_sequences[:,:,ind_pc_over_peep][~padding_mask] = \
        padded_sequences[:,:,ind_ppeak][~padding_mask] - padded_sequences[:,:,ind_peep][~padding_mask]

        padded_sequences[:,:,ind_pc_over_peep][padding_mask] = padding_value # padded_sequences[:,:,0][padding_mask]


        # ps_over_peep
        ind_ps_over_peep = removed_features['ps_over_peep']
        padded_sequences[:,:,ind_ps_over_peep] = padded_sequences[:,:,ind_pc_over_peep]

        # calcium_albc

        # get coefficients of the regression
        reg_coef_calcium_albc = messages['reg_coef_calcium_albc']


        ind_calcium = orig_features.index('calcium')
        ind_albumin = orig_features.index('albumin')
        ind_calcium_albc = orig_features.index('calcium_albc')

        if verbose:
            print('\nCheck index of calcium_albc to be same in list and dict')
            print('index from list: {}, index from dict: {}'
                  .format(ind_calcium_albc, removed_features['calcium_albc']))

        # calcium_albc ~ calcium + albumin
        padded_sequences[:,:,ind_calcium_albc][~padding_mask] = \
        reg_coef_calcium_albc['coef_calcium'] * padded_sequences[:,:,ind_calcium][~padding_mask] + \
        reg_coef_calcium_albc['coef_albumin'] * padded_sequences[:,:,ind_albumin][~padding_mask] + \
        reg_coef_calcium_albc['intercept']

        padded_sequences[:,:,ind_calcium_albc][padding_mask] = padding_value # padded_sequences[:,:,0][padding_mask]


        if verbose:
            print('\nshape of array after returning superfluous features: ',
                  padded_sequences.shape)
            '''
            print('\nfirst row of first individual after returning removed superfluous features: ')
            print(padded_sequences[0,0,:])

            print('\n12th row of second individual: ')
            print(padded_sequences[1,11,:])
            '''

        return padded_sequences, messages


def make_histogram(hist_data, title):
    import matplotlib.pyplot as plt

    print('making plot...')
    fig, ax1 = plt.subplots(figsize=(13, 13))
    ax1.hist(hist_data)
    mu = np.nanmean(hist_data)
    sigma = np.nanstd(hist_data)
    title = title + ', mean = {:0.2f}, std = {:0.2f}'.format(mu, sigma)
    ax1.set_title(title)

    return fig


def ln_scale(self, padded_sequences, messages):

    '''
    Scale non-negative values of rigth-skewed variables to their natural
    logatirhms before modelling

    Args: numpy array of shape (num_sequences x sequence len x num_features),
          assumes values to be scaledd between 0 and 1, and missing value to be -1
          OR padded_mask to be in the messages

    Returns:
        modified array, where non-negative values of right-skewed features are ln-sclaed

    '''
    # NOTE: if histograms are to be done, output file needs to be given
    #(currently commented out from data_pipeline.py)

    #import matplotlib.backends.backend_pdf
    #from hider.data_processing_utils import make_histogram

    verbose = messages['verbose']
    # pass data_dictionary as an argument for the LogarithimicScaler
    # data_dictionary = messages['data_dictionary']

    if verbose:
        print('data type of data_dictionary: ', type(self.data_dictionary))

    # get right-skewed features
    self.names_right_skewed = list(self.data_dictionary.loc[self.data_dictionary['right_skewed'] ==1, 'feature_name'])

    # get indices of right-skewed features in the data using the current_features list
    current_features = messages['current_features']

    for i in self.names_right_skewed:
        self.ind_right_skewed.append(current_features.index(i))


    # empty list for plots
    #figures = []

    # scale values of the columns

    # mask for indicating values for scaling and re-scaling
    self.mask = np.zeros(padded_sequences.shape, dtype = bool)

    # make mask value to True for values to be scaled an re-scaled,
    # i.e. non-negative values of right-skewed features
    for i in range(len(self.ind_right_skewed)):
        ind = self.ind_right_skewed[i]
        '''

        name = self.names_right_skewed[i]

        if verbose:
            print('\nscaling ', name)

            # make histogram before scaling
            title = name + ' before ln-scaling'
            # make plotting data 1D
            fig = make_histogram(padded_sequences[:,:,ind].flatten(), title)
            figures.append(fig)

            print('\nfirst individual before scaling')
            print(padded_sequences[0,:,ind])
            '''

        self.mask[:,:,ind] = padded_sequences[:,:,ind] >= 0



    #'padding_mask': a boolean mask of shape (num_data, max_seq_len) indicating
    # whether (True) or not (False) a time step in a sequence is padded.

    # if no padding_mask, padded values are expected to be negative and
    # thus already masked out
    if 'padding_mask' in messages:
            padding_mask = messages['padding_mask']

            if verbose:
                print('padding_mask found, dimensions: ', padding_mask.shape)

            num_data, max_seq_len = padding_mask.shape
            self.mask = self.mask & ~padding_mask.reshape((num_data, max_seq_len, 1))

            if verbose:
                print('padding_mask and "skewed_mask" combined, dimensions: ', self.mask.shape)

    if verbose:
        print('unique values before ln-scaling: ', np.unique(padded_sequences[self.mask]))


    # use mask to scale values
    padded_sequences[self.mask] = np.log(padded_sequences[self.mask] + self.epsilon)


    if verbose:
        print('\n\nLogarithmic scaling done!\n')
        print('unique values of ln-scaled values: ', np.unique(padded_sequences[self.mask]))
        '''
        for i in range(len(self.ind_right_skewed)):
            ind = self.ind_right_skewed[i]
            name = self.names_right_skewed[i]

            # make histogram after scaling
            title = name + ' after ln-scaling'
            fig = make_histogram(padded_sequences[:,:,ind].flatten(), title)
            figures.append(fig)

            print('\nfirst individual after scaling')
            print(padded_sequences[0,:,ind])


        # save figures to a pdf file
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.output_file_scaling)

        for figure in figures:
            # rasterize
            figure.set_rasterized(True)
            # dpi 200
            pdf.savefig(figure, dpi = 200)


        pdf.close()

        print('Histograms saved to ', self.output_file_scaling)
        '''

    return padded_sequences, messages



# postprocessing: re-scale

def re_scale(self, synt_data, messages):
    '''
    Args:
        synt_data: synthetic data, generated from input data where some values were ln-scaled
        numpy array of shape (num_sequences x sequence len x num_features)
    Returns:
        modified numpy array, where ln-scaled values are re-scaled

    '''
    # NOTE: if histograms are to be done, output file needs to be given
    #(currently commented out from data_pipeline.py)
    #import matplotlib.backends.backend_pdf
    #from hider.data_processing_utils import make_histogram

    verbose = messages['verbose']

    '''
    if verbose:
        # empty list for plots
        figures = []

        for i in range(len(self.ind_right_skewed)):
            ind = self.ind_right_skewed[i]
            name = self.names_right_skewed[i]

            # make histogram before re-scaling
            title = name + ' before re-scaling'
            fig = make_histogram(synt_data[:,:,ind].flatten(), title)
            figures.append(fig)

            print('\n"{}" column of first individual before re-scaling'.format(name))
            print(synt_data[0,:,ind])
            '''
    # re-scale right-skewed columns using a mask where
    # scaled values are indicated by boolean value True
    if verbose:
        print('\nunique values before re-scaling: ', np.unique(synt_data[self.mask]))

    #synt_data[self.mask] = np.exp(synt_data[self.mask])- self.epsilon
    prev_data = synt_data
    synt_data = np.copy(synt_data)
    synt_data[:,:,self.ind_right_skewed] = np.exp(synt_data[:,:,self.ind_right_skewed])- self.epsilon
    if 'padding_mask' in messages:
        padding_mask = messages['padding_mask']
        synt_data = np.where(np.expand_dims(padding_mask, -1), prev_data, synt_data)


    if verbose:
        print('\n\nRe-scaling done!\n')
        print('unique values of re-scaled values: ', np.unique(synt_data[:,:,self.ind_right_skewed]))
        '''
        for i in range(len(self.ind_right_skewed)):
            ind = self.ind_right_skewed[i]
            name = self.names_right_skewed[i]

            # make histogram after re-scaling
            title = name + ' after re-scaling'
            fig = make_histogram(synt_data[:,:,ind].flatten(), title)
            figures.append(fig)

            print('\n"{}" column of first individual after re-scaling'.format(name))
            print(synt_data[0,:,ind])

        # save figures to a pdf file
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.output_file_rescaling)

        for figure in figures:
            # rasterize
            figure.set_rasterized(True)
            # dpi 200
            pdf.savefig(figure, dpi = 200)


        pdf.close()

        print('Histograms saved to ', self.output_file_rescaling)
        '''
    return synt_data, messages

def imputation_per_feature(padded_sequences, median_vals, padding_mask):
    imputed_data = np.zeros_like(padded_sequences)
    for j, median_val in enumerate(median_vals):
        feature = padded_sequences[:,:,j]
        # backward fill
        # need to keep track of indivs that have missing values before padding
        temp = np.where(padding_mask, np.nan, feature)
        na_mask0 = np.cumprod(np.isnan(temp[:, ::-1]),axis=1)[:, ::-1]
        bfilled = pd.DataFrame(temp.flatten()).bfill().values
        bfilled = bfilled.reshape(feature.shape)
        bfilled = np.where(na_mask0, np.nan, bfilled)
        # fwd fill
        # need to keep track of completely missing sequences
        na_mask1 = np.all(np.isnan(bfilled),axis=1)
        ffilled = pd.DataFrame(bfilled.flatten()).ffill().values
        ffilled = ffilled.reshape(feature.shape)
        ffilled[na_mask1] = np.nan
        #median imputation
        imputed = np.where(np.isnan(ffilled), median_val, ffilled)
        imputed_data[:,:,j] = imputed
    return imputed_data
