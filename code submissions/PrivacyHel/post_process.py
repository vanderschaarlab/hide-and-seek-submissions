"""
This module contains some basic post-processing classes.

Our models do not deal with missing values and thus generate full-length
non-sparse outputs. The classes defined in this module aim to transform
this data into something that has padding and sparsity similar to
the original data.

"""

import numpy as np

class Padder:
    """ Applies random padding to generated sequences.

    Simply learns the padding proportion from the data (fit) and applies to synthetic data (transform).

    NOT USED
    """
    def __init__(self, padded_val = -1):
        self.padded_val = padded_val

    def fit(self, train_data):
        """
        train_data : -1 padded 3d numpy array
        """
        padded_sequences = np.all(train_data == self.padded_val, axis=-1)
        padded_lengths = np.sum(padded_sequences, axis=-1)
        lengths, counts = np.unique(padded_lengths, return_counts=True)
        self.lengths = lengths
        self.probs = counts/counts.sum()


    def fit_transform(self, train_data, syn_data, inplace=False):
        self.fit(train_data)
        num_syn, T, d = syn_data.shape
        sampled = np.random.choice(self.lengths, p=self.probs, size=len(syn_data))
        padding_mask_gen = np.zeros_like(syn_data, dtype=bool)
        for i, length in enumerate(sampled):
            padding_mask_gen[i, T-length:, :] = True

        syn_data_padded = syn_data
        if not inplace:
            syn_data_padded = syn_data.copy()
        syn_data_padded[padding_mask_gen] = self.padded_val

        return syn_data_padded, padding_mask_gen

class NAImputer:
    """ Applies random missingness to generated sequences.

    Learns the per-feature sparsity over the original data and sparsifies
    the generated data accordingly.

    Ensure that there are at least two non-padding values for each
    feature, to avoid triggering classification evaluation mode in the
    competition.

    NOT USED
    """

    def __init__(self, na_value = np.nan, padded_val=-1):
        self.na_value = na_value
        self.padded_val = padded_val

    def fit(self, train_data):
        """
        learns to impute nans time and dim wise
        """
        padding_mask = np.all(train_data == self.padded_val, axis=-1)
        if np.isnan(self.na_value):
            self.na_props = np.stack([(np.isnan(train_data[:,:,i]*~padding_mask)).sum(0)/(~padding_mask).sum(0) for i in range(train_data.shape[-1])]).T
        else:
            self.na_props = np.stack([(train_data[:,:,i]*~padding_mask == self.na_value).sum(0)/(~padding_mask).sum(0) for i in range(train_data.shape[-1])]).T

    def fit_transform(self, train_data, syn_data, inplace=False):
        num_syn, T, d = syn_data.shape
        self.fit(train_data)
        NAs = self.na_props[np.newaxis] > np.random.rand(num_syn, T, d)
        for i in np.where(np.sum(NAs,axis=(0,1))>num_syn*T-2)[0]:
            num_val_to_add=np.sum(NAs[:,:,i])-(num_syn*T-2)
            random_num=np.random.choice(range(np.sum(NAs[:,:,i])),size=num_val_to_add,replace=False)
            true_post=np.vstack(np.where(NAs[:,:,i]==1)).T
            for j in random_num:
                NAs[true_post[j,0]][true_post[j,1]][i]=0
        if inplace:
            syn_data = np.where(NAs,self.na_value,syn_data)
        else:
            syn_data_imputed = syn_data.copy()
            syn_data_imputed = np.where(NAs,self.na_value,syn_data_imputed)
        return syn_data_imputed


class Perturber:
    """ For each feature, slightly perturbs the first observation.

    This is not for privacy but merely to ensure that there are at least
    two different non-padding values so that competition evaluation does not
    switch to classification mode.
    """
    def __init__(self, noise_val=10e-10):
        self.noise_val=noise_val
    def perturb(self,syn_data,padding_mask):
        num_syn, T, d = syn_data.shape
        for i in range(d):
            if ~np.isnan(syn_data[0][0][i]) and ~padding_mask[0][0][i]:
                syn_data[0][0][i]+=self.noise_val
            else:
                for t in range(T):
                    for n in range(num_syn):
                        if ~np.isnan(syn_data[n][t][i]) and ~padding_mask[n][t][i]:
                            # print(padding_mask[n][t][i])
                            syn_data[n][t][i]+=self.noise_val
                            break
                    else:
                        continue
                    break

        return syn_data
