"""
The hider module containing the `hider(...)` function.
"""
# pylint: disable=fixme
from typing import Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Mean
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm
from tensorflow.keras import backend
from scipy.stats import bernoulli
from sklearn.utils import resample


# ----------------------------------------------- #
# RAEwGAN Recurrent Aeuto-encoder Wasserstein GAN #
# ----------------------------------------------- #

# Define models
def define_recurrent_encoder(timesteps, features, latent_dim):
   ''' Define recurrent encoder
       args: 
       dimensions timesteps, feautres.
       latent dim: hyper-parameter
       return:
       recurrent encoder model
    '''
    recurrent_encoder = tfkm.Sequential([
        tfkl.Masking(mask_value=-1.,
                     input_shape=(timesteps, features)),
        tfkl.Dropout(0.2),
        tfkl.GRU(42, return_sequences = True),
        tfkl.GRU(latent_dim)
    ])
    return recurrent_encoder

class ClipConstraint(tf.keras.constraints.Constraint):
    ''' Clip model weight to a given hypercube
    '''
    # set clip values when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    # clip model weights to hypercuber
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

def define_recurrent_decoder_critic(timesteps, latent_dim):
    ''' Define recurrent decoder
       args: 
         dimensions timesteps.
         latent dim: hyper-parameter
       return:
         recurrent decoder model
    '''
    # weight constraint
    cont = ClipConstraint(0.01)
    recurrent_decoder_critic = tfkm.Sequential([
        tfkl.RepeatVector(timesteps, input_shape = [latent_dim]),
        tfkl.GRU(10, return_sequences = True, kernel_constraint=cont),
        tfkl.LayerNormalization(),
        tfkl.GRU(5, return_sequences = True, kernel_constraint=cont),
        tfkl.LayerNormalization(),
        tfkl.TimeDistributed(tfkl.Dense(1))
    ])
    return recurrent_decoder_critic

def define_recurrent_critic(timesteps, features):
    ''' Define recurrent critic
       args: 
         dimensions timesteps, features.
       return:
         recurrent critic model
    '''
    # weight constraint
    cont = ClipConstraint(0.01)
    # define critic
    recurrent_critic = tfkm.Sequential([
         tfkl.Masking(mask_value=-1.,
                     input_shape=(timesteps, features)),
        tfkl.GRU(60, return_sequences = True, kernel_constraint=cont),
        tfkl.LayerNormalization(),
        tfkl.GRU(23, return_sequences = True, kernel_constraint=cont),
        tfkl.LayerNormalization(),
        tfkl.TimeDistributed(tfkl.Dense(1))
    ])
    return recurrent_critic

def define_recurrent_generator(timesteps, latent_dim, features):
    ''' Define recurrent generator
       args: 
         dimensions timesteps, features.
         latent dim: hyper-parameter
       return:
         recurrent generator model
    '''
    recurrent_generator = tfkm.Sequential([
        tfkl.RepeatVector(timesteps, input_shape = [latent_dim]),
        # upsamples
        tfkl.GRU(42, return_sequences = True),
        tfkl.LayerNormalization(),
        tfkl.Dropout(0.2),
        # upsample
        tfkl.GRU(87, return_sequences = True),
        tfkl.LayerNormalization(),
        tfkl.Dropout(0.2),
        tfkl.TimeDistributed(tfkl.Dense(features))
    ])
    return recurrent_generator

# Noise generator
def generate_noise(no, latent_dim):
    ''' Generates Gaussian noise with shape [no, latent_dim]
    '''
    noise = tf.random.normal(shape = [no, latent_dim])
    return noise

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    ''' Implementation of Wasserstein loss
    args:
       y_true: true values (-1 for observed and 1 for missing). 
       y_pred: predicted values.
     
    '''
    return tf.keras.backend.mean(y_true * y_pred)

# Define loss and gradients
def reconstruction_x_real_loss(model, y_true, M, dim):
    ''' Reconstruction loss for measured continuos variables. 
      We use the Huber loss, which is less sensitive to outliers in data
    args:
       y_true: true measured values. 
       model: generator.
       M: mask.
       dim: dimension of fetures.
    return:
       loss and gradients.
    '''
    with tf.GradientTape() as tape:
        y_pred = model(y_true, training = True)
        mip = tf.cast(1 + dim / 2, tf.int32) # mask initial position
        y_true_real = y_true[:,:,0:mip]
        y_pred_real = y_pred[:,:,0:mip]
        M_real = M[:,:,0:mip]
        # reconstruction real loss
        rec_loss = tf.keras.losses.huber( (1 - M_real) * y_true_real, (1 - M_real) * y_pred_real )
    return rec_loss, tape.gradient(rec_loss, model.trainable_variables)

def reconstruction_x_bin_loss(model, y_true, M, dim):
    ''' Reconstruction loss for the mask. 
      We use the a binary cross-entropy loss.
    args:
       y_true: true measured values. 
       model: generator.
       M: mask.
       dim: dimension of fetures.
    return:
       loss and gradients.
    '''
    with tf.GradientTape() as tape:
        y_pred = model(y_true, training = True)
        mip = tf.cast(1 + dim / 2, tf.int32) # mask initial position
        y_true_binary = y_true[:,:,mip:]
        y_pred_binary = y_pred[:,:,mip:]
        M_binary = M[:,:,mip:]
        # reconstruction = real + binary loss
        rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true_binary[M_binary == 0], y_pred_binary[M_binary == 0])
    return rec_loss, tape.gradient(rec_loss, model.trainable_variables)

def reconstruction_z_loss(model, z):
    ''' Reconstruction loss for latent space z. 
      We use the MSE loss.
    args:
       y_true: true measured values. 
       model: generator.
       M: mask.
       dim: dimension of fetures.
    return:
       loss and gradients.
    '''
    with tf.GradientTape() as tape:
        z_pred = model(z, training = True)
        loss = tf.keras.losses.MSE(z, z_pred)
    return loss, tape.gradient(loss, model.trainable_variables)

def critic_loss(model, xhat, xp, y1):
    '''
    Critic loss
    args:
     xhat : generated_samples
     xp : reproduced_samples
     model: discriminator/critic dx
    '''
    with tf.GradientTape() as tape:
        d_on_real = model(xp, training = True)
        d_on_fake = model(xhat, training = True)
        x_real_and_fake = tf.concat([d_on_real, d_on_fake], axis=1)
        d_loss1 = wasserstein_loss(y1, x_real_and_fake)
    return d_loss1, tape.gradient(d_loss1, model.trainable_variables)

def generator_loss(model, noise, y2):
    '''
    Train generator on critic fake samples
    args:
     noise : z
     y2 : fake labels 
     model: generator
    '''
    with tf.GradientTape() as tape:
        d_on_noise = model(noise, training = True)
        d_loss2 = wasserstein_loss(y2, d_on_noise)
    return d_loss2, tape.gradient(d_loss2, model.trainable_variables)

def train_raegan_step(dataset, n_epoch, n_critic, seq_len, latent_dim, dim):
     '''
     Train RAE GAN
    args:
       dataset: TF dataset
       n_epoch: number of iterations on the dataset 
       n_critic: number of critic iterations per generator iteration.
       latent_dim: hyper-parameter
    '''
    # create models
    rnn_generator = define_recurrent_generator(seq_len, latent_dim, dim)
    rnn_critic   = define_recurrent_critic(seq_len, dim)
    rnn_encoder  = define_recurrent_encoder(seq_len, dim, latent_dim)
    rnn_ae_critc = define_recurrent_decoder_critic(seq_len, latent_dim)
    rnn_encogen  = tfkm.Sequential([rnn_encoder, rnn_generator])
    rnn_gencoder = tfkm.Sequential([rnn_generator, rnn_encoder])
    gan_model1   = tfkm.Sequential([rnn_generator, rnn_critic])
    ae_model     = tfkm.Sequential([rnn_encoder, rnn_ae_critc])
    gan_model2   = tfkm.Sequential([rnn_generator, ae_model])
    # create optimizers
    adam_optimizer = tf.keras.optimizers.Adam()
    rmsprop_optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)

    # Keep results for plotting
    train_dx_loss_results = []
    train_dz_loss_results = []
    train_rec_loss_results = []

    for epoch in range(n_epoch):
        epoch_dx_loss_avg = Mean()
        epoch_dz_loss_avg = Mean()
        epoch_rec_loss_avg = Mean()
        for x_batch, padding_mask_batch in dataset:
            no, seq_len, dim = x_batch.shape # x.shape
            y1 = tf.cast(tf.reshape(tf.constant([[-1.]] * seq_len + [[1.]] * seq_len), [seq_len*2, 1]), tf.float32)
            y1 = tf.broadcast_to(y1, [no, seq_len*2, 1])
            y2 = tf.cast(tf.reshape(tf.constant([[-1.]] * seq_len), [seq_len, 1]), tf.float32)
            y2 = tf.broadcast_to(y2, [no, seq_len, 1])

            # sample reconstruction loss
            # loss of real value variables
            rec_real_loss1, rec_real_gradient1 = reconstruction_x_real_loss(rnn_encogen, x_batch, padding_mask_batch, dim)
            adam_optimizer.apply_gradients(zip(rec_real_gradient1, rnn_encogen.trainable_variables))
            # loss of mask reconstruction
            rec_bin_loss1, rec_bin_gradient1 = reconstruction_x_bin_loss(rnn_encogen, x_batch, padding_mask_batch, dim)
            adam_optimizer.apply_gradients(zip(rec_bin_gradient1, rnn_encogen.trainable_variables))
            # house keeping
            rec_loss1 = tf.reduce_sum(rec_real_loss1) + tf.reduce_sum(rec_bin_loss1)

            # latent vector reconstruction loss
            random_noise = generate_noise(no, latent_dim) # z
            rec_loss2, rec_gradient2 = reconstruction_z_loss(rnn_gencoder, random_noise)
            adam_optimizer.apply_gradients(zip(rec_gradient2, rnn_gencoder.trainable_variables))

            # train Dx
            for _ in range(n_critic):
                random_noise = generate_noise(no, latent_dim) # z
                generated_samples = rnn_generator(random_noise, training = True) # x_hat
                encodings = rnn_encoder(x_batch, training = True) # z_hat
                reproduced_samples = rnn_generator(encodings, training = True) # x'
                rnn_critic.trainable = True
                d_loss1, d_grads1 = critic_loss(rnn_critic, generated_samples, reproduced_samples, y1)
                rmsprop_optimizer.apply_gradients(zip(d_grads1, rnn_critic.trainable_variables))
            # update generator via Dx critic's error
            random_noise = generate_noise(no, latent_dim) # z
            rnn_critic.trainable = False
            d_loss2, d_grads2 = generator_loss(gan_model1, random_noise, y2)
            rmsprop_optimizer.apply_gradients(zip(d_grads2, gan_model1.trainable_variables))

            # train Dz
            for _ in range(n_critic):
                random_noise = generate_noise(no, latent_dim) # z
                generated_samples = rnn_generator(random_noise, training = True) # x_hat
                encodings = rnn_encoder(x_batch, training = True) # z_hat
                encoded_noise = rnn_encoder(generated_samples, training = True) # z'
                rnn_ae_critc.trainable = True
                ae_loss1, ae_grads1 = critic_loss(rnn_ae_critc, encoded_noise, encodings, y1)
                rmsprop_optimizer.apply_gradients(zip(ae_grads1, rnn_ae_critc.trainable_variables))
            # update generator via Dz critic's error
            random_noise = generate_noise(no, latent_dim) # z
            rnn_ae_critc.trainable = False
            ae_loss2, ae_grads2 = generator_loss(gan_model2, random_noise, y2)
            rmsprop_optimizer.apply_gradients(zip(ae_grads2, gan_model2.trainable_variables))
            # Track progress: Add current batch loss
            epoch_dx_loss_avg.update_state(d_loss1)
            epoch_dz_loss_avg.update_state(ae_loss1)
            epoch_rec_loss_avg.update_state(rec_loss1)

        # End epoch
        train_dx_loss_results.append(epoch_dx_loss_avg.result())
        train_dz_loss_results.append(epoch_dz_loss_avg.result())
        train_rec_loss_results.append(epoch_rec_loss_avg.result())

    return rnn_encogen, train_dx_loss_results, train_dz_loss_results, train_rec_loss_results

def rae_wgan(input_dat, seq_times, padding_mask, batch_size, n_epoch, n_critic, latent_dim):
    '''
    Call to RAE GAN
    args:
      input_dat: imputed data, inputs to RAE WGAN.
      padding_mask: padding mask of timesteps 
      batch_size: size of batch.
      n_epoch: number of iterations on the dataset 
      n_critic: number of critic iterations per generator iteration.
      latent_dim: hyper-parameter
    
    return:
      res: simulated data, post-process output from trained RAE-GAN.
      simulated_padding_mask: simulated_padding_mask, output from trained RAE-GAN.
    '''
    no, seq_len, dim = input_dat.shape
    input_dat = tf.cast(input_dat, tf.float32)
    padding_mask = tf.cast(padding_mask, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((input_dat, padding_mask))
    dataset = dataset.batch(batch_size).prefetch(1)
    dataset = dataset.shuffle(no, reshuffle_each_iteration=True)
    # train gain model
    rnn_encogen, _discriminator, _generator, _reconstruction = train_raegan_step(dataset,n_epoch, n_critic, seq_len, latent_dim, dim)

    # Bootstrap dataset
    res, simulated_padding_mask = bootstrap_results(rnn_encogen, input_dat, seq_times)

    return res, simulated_padding_mask

# ------------------------------------------ #
# GAIN Imputation network
# ------------------------------------------ #

def create_wgain(dim, latent_dim):
    ''' Define WGAIN
       args: 
         dim feautres dimension.
         latent dim: hyper-parameter
       return:
         recurrent wgain model
    '''
    # define critic weight clipping
    class ClipConstraint(keras.constraints.Constraint):
        def __init__(self, clip_value):
            self.clip_value = clip_value
        def __call__(self, weights):
            return backend.clip(weights, -self.clip_value, self.clip_value)
        def get_config(self):
            return {'clip_value': self.clip_value}
    # define generator
    generator = tfkm.Sequential([
        tfkl.Dense(24, input_shape = [latent_dim], kernel_initializer="he_normal"),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dropout(0.2),
        # upsample
        tfkl.Dense(42, input_shape = [dim], kernel_initializer="he_normal"),
        tfkl.BatchNormalization(),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dropout(0.2),
        # output
        tfkl.Dense(dim)
    ])
    # weight constraint
    cont = ClipConstraint(0.01)
    # define critic
    critic = tfkm.Sequential([
        # upsample
        tfkl.Dense(87, input_shape = [dim], kernel_constraint = cont, kernel_initializer="he_normal"),
        tfkl.BatchNormalization(),
        tfkl.LeakyReLU(alpha=0.2),
        # hidden layer
        tfkl.Dense(93, kernel_constraint = cont, kernel_initializer="he_normal"),
        tfkl.BatchNormalization(),
        tfkl.LeakyReLU(alpha=0.2),
        # scoring linear activation
        tfkl.Dense(dim - 1, activation = 'linear'),
    ])
    return tfkm.Sequential([generator, critic])

def hint_generator(X, M):
  ''' Hint generator
    args:
      X: data with raw imputed values
      M: mask
     return : hint input to generator network.
    '''
    # takes X: data with raw imputed values
    #       M: mask
    # return : hint input to generator network.
    noise =  tf.random.normal(shape = X.shape)
    hint =  M * (X + noise) + (1 - M) * X
    return hint

# Define loss and gradients
def discriminator_grad(model, inputs, mask):
    '''
    Critic loss
    args:
     inputs : noisy input dataset
     mask : mask of missing values
     model: discriminator/critic 
    '''
    with tf.GradientTape() as tape:
        output = model(inputs, training = True)
        loss_value = wasserstein_loss(y_true = mask, y_pred = output)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def gain_grad(gain, inputs):
     '''
    Generator loss and grads
    args:
     inputs : noisy input dataset
     model: gain model
    '''
    with tf.GradientTape() as tape:
        output = gain(inputs, training = True)
        y2 = - tf.ones(shape = output.shape)
        loss_value = wasserstein_loss(y_true = y2, y_pred = output)
    return loss_value, tape.gradient(loss_value, gain.trainable_variables)

def rec_grad(generator_layer, inputs, mask, alpha):
    '''
    Reconstruction loss and grads
    args:
     inputs : noisy input dataset
     generator_layer : generator
     mask: mask of missing values
     alpha: hyper-parameter
    '''
    with tf.GradientTape() as tape:
        output = generator_layer(inputs, training = True)
        y_true = inputs
        y_pred = output
        rec_loss =  alpha * tf.keras.losses.huber(y_true, y_pred )
    return rec_loss, tape.gradient(rec_loss, generator_layer.trainable_variables)

def train_wgain(dataset, gain, n_epoch, n_critic, alpha):
    '''Train wgain function
  
    Args:
      - dataset: A dataset TF2 object.
      - gain: a gain model.
      - n_epoch: number of iterations.
      - alpha: hyper-parameter
        
    Returns:
      - gain: Trained model
      - critic loss, generator loss and reconstruction loss for monitoring.
    '''

    generator, discriminator = gain.layers
    d_optimizer = keras.optimizers.RMSprop(lr=0.00005)
    g_optimizer = keras.optimizers.Adam()
    # Keep results for plotting
    train_d_loss_results = []
    train_g_loss_results = []
    train_rec_loss_results = []

    for epoch in range(n_epoch):
        epoch_d_loss_avg = Mean()
        epoch_g_loss_avg = Mean()
        epoch_rec_loss_avg = Mean()
        for x_batch, mask_batch in dataset:
            batch_size, dim = x_batch.shape
            # phase 1: train discriminator
            for _ in range(n_critic):
                hint = hint_generator(x_batch, mask_batch)
                generated_samples = generator(hint, training = True)
                discriminator.trainable = True
                d_loss, d_grads = discriminator_grad(discriminator, generated_samples, mask_batch[:,1:])
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            # phase 2 - training the generator
            hint = hint_generator(x_batch, mask_batch)
            discriminator.trainable = False
            g_loss, g_grads = gain_grad(gain, hint)
            d_optimizer.apply_gradients(zip(g_grads, gain.trainable_variables))
            hint = hint_generator(x_batch, mask_batch)
            rec_loss, rec_grads = rec_grad(generator, hint, mask_batch, alpha)
            g_optimizer.apply_gradients(zip(rec_grads, gain.trainable_variables))
            # Track progress: Add current batch loss
            epoch_d_loss_avg.update_state(d_loss)
            epoch_g_loss_avg.update_state(g_loss)
            epoch_rec_loss_avg.update_state(rec_loss)
        # End epoch
        train_d_loss_results.append(epoch_d_loss_avg.result())
        train_g_loss_results.append(epoch_g_loss_avg.result())
        train_rec_loss_results.append(epoch_rec_loss_avg.result())

    return gain, train_d_loss_results, train_g_loss_results, train_rec_loss_results

def imputation(comp_data, mask_data, padding_mask, batch_size, n_epoch, alpha, n_critic):
    '''Call GAIN method
  
    Args:
      - comp_data: original data with missing values
      - mask_data: mask of missing values on original data.
      - padding mask: mask of time-steps that are padded.
      - wgain_parameters: WGAIN network parameters:
        - batch_size: Batch size
        - n_critic: Number of iterations of the critic per generator
        - alpha: Hyperparameter
        - n_epoch: Iterations on the data
        
    Returns:
      - imputed_data: imputed data
      - padding mask: same as input padding mask
      - seq_times: T actual lenght for each observed individual n.
    '''

    input_dat, input_mask, seq_times, seq_len = preformat_data_gain(comp_data, mask_data, padding_mask)
    input_dat = np.concatenate(input_dat, axis = 0)
    input_mask = np.concatenate(input_mask, axis = 0)
    input_mask[input_mask == 0] = -1 # real data points
    seq_len, dim = input_dat.shape
    input_dat = tf.cast(input_dat, tf.float32)
    input_mask = tf.cast( input_mask, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((input_dat, input_mask))
    dataset = dataset.batch(batch_size).prefetch(1)
    dataset = dataset.shuffle(seq_len, reshuffle_each_iteration=True)
    # train gain model
    gain_model = create_wgain(dim, dim)
    gain_model, _discriminator, _generator, _reconstruction = train_wgain(dataset, gain_model, n_epoch,  n_critic, alpha)
    generator , discriminator = gain_model.layers

    # Impute dataset
    hint = hint_generator(input_dat, input_mask)
    input_mask = tf.cast(tf.equal(input_mask, 1.0), tf.float32) # real data points --> 0, fake --> 1
    imputed_data = input_mask * generator(hint) + (1 - input_mask) * input_dat
    imputed_data = post_process_data_gain(imputed_data.numpy(), input_mask.numpy(), seq_times)

    imputed_data, padding_mask = post_process_list_wgain(imputed_data, np.max(seq_times))
    return imputed_data, padding_mask, seq_times

# Interpolation module
def interpolation_fun(dat, padding_mask) :
    """Interpolation function.
    Args:
        - data: input data
        - padding_mask: original padding mask
    Returns:
        - dat: interpolated data
        - median_val: median value for each feature d.
        - all_nan: vector d indicating if a variable is completely missing.
    """
    no, seq_len, dim = dat.shape
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    # get the median value
    median_val = np.empty(dim)
    all_nan = np.zeros(dim)
    for k in range(dim):
        feature = (dat[:,:,k]).copy()
        median_val[k] = np.nanmedian(feature[padding_mask[:,:,k] != True])
        if np.isnan(median_val[k]):
            all_nan[k] = 1.0
    # attempt interpolation
    for i in range(no):
        for k in range(dim):
            observed_values = (dat[i,:,k][padding_mask[i,:,k] != True]).copy()
            nans, x = nan_helper(observed_values)
            if np.sum(nans) == len(nans): ## all nans
                observed_values = median_val[k] + np.random.normal(0, 0.1, len(observed_values))
            else:
                observed_values[nans] = np.interp(x(nans), x(~nans), observed_values[~nans])
            dat[i,:,k][padding_mask[i,:,k] != True] = observed_values
    return dat, median_val, all_nan

# ------------------------------------------ #
# Data Helpers
# ------------------------------------------ #
def MinMaxScaler(ori_data, padder_value = -1.):
    """Scale Normalizer.
    Args:
        - data: imputed data
        - ori_data: raw data
    Returns:
        - norm_data: normalized data
        - min_val: minimum values (for feature)
        - max_val: maximum values (for feature)
    """
    norm_data = ori_data.copy()
    no, seq_len, dim = ori_data.shape
    max_value = np.zeros(dim)
    min_value = np.zeros(dim)
    for k in range(dim):
        max_value[k] = np.nanmax(norm_data[:,:,k][norm_data[:,:,k] != padder_value])
        min_value[k] = np.nanmin(norm_data[:,:,k][norm_data[:,:,k] != padder_value])
        norm_data[:,:,k] = (norm_data[:,:,k] - min_value[k]) / (max_value[k] + min_value[k] + 1e-7 )
    return norm_data, max_value, min_value

def DeScaler(data, max_value, min_value, all_nan):
   """Scale Un-Normalizer.
    Args:
        - data: input data
        - max_value: vector d 
        - min_value: vector d
        - all_nan: vector d, indicator if all is nan for a feature = 1, otherwise = 0.
    Returns:
        - res: un-normalized data
    """
    no, seq_len, dim = data.shape
    dim = len(all_nan)
    res = - np.ones(shape = (no, seq_len, dim))
    # reconstruct all nan variable if there
    if np.any(all_nan == 1):
        ind = 0 # indice
        for k in range(dim):
            if all_nan[k] == 1:
                res[:,:,k] = np.nan
            else:
                res[:,:,k] = data[:,:,ind]
                ind += 1
    # unscale variables
    for k in range(dim):
        res[:,:,k] = res[:,:,k] * (max_value[k] + min_value[k] + 1e-7 ) + min_value[k]
    res[:,:,0] = np.around(res[:,:,0])
    return res

def preformat_data_gain(dat, mask, padding_mask):
    """Preformat data helpere for GAIN.

    Args:
        input data: Input data np.ndarray shape [num_examples, max_seq_len, num_features].
        seq_times : T dimension
    Returns:
        foo_imputed input data to WGAIN np.ndarray shape [num_examples, max_seq_len, num_features].
        foo_mask np.ndarray shape [num_examples, max_seq_len, num_features].
    """
    no, seq_len, dim = np.asarray(dat).shape
    # drop rectangular array in favour of ragged array approach
    foo_imputed = list()
    for i in range(no):
        foo_imputed.append(dat[i][padding_mask[i].sum(axis = 1) == 0,:])
    foo_mask = list()
    for i in range(no):
        foo_mask.append(mask[i][padding_mask[i].sum(axis = 1) == 0,:])
    get_seq_len = lambda l: l.shape[0]
    seq_times = list(map(get_seq_len, foo_imputed))
    return foo_imputed, foo_mask, seq_times, seq_len

def post_process_data_gain(generated_imputed_data, mask, seq_times):
    """Post process wgain helper.

    Args:
        input data: Input data np.ndarray shape [num_examples, max_seq_len, num_features].
        seq_times : T dimension
    Returns:
        output input data to WGAIN np.ndarray shape [num_examples, max_seq_len, num_features].
        padding_mask np.ndarray shape [num_examples, max_seq_len, num_features].
    """
    no = len(seq_times)
    foo_generated_imputed = []
    ind = 0
    for i in range(no):
        foodat = generated_imputed_data[ind:(seq_times[i] + ind)]
        foomask =  mask[ind:(seq_times[i] + ind)]
        foo_generated_imputed.append(  np.concatenate( (foodat, foomask[:,1:]), axis = 1) )
        ind = seq_times[i] + ind
    return foo_generated_imputed

def post_process_list_wgain(dat, max_seq_len):
  """Post process wgain helper.

    Args:
        input data: Input data np.ndarray shape [num_examples, max_seq_len, num_features].
        seq_times : T dimension
    Returns:
        output input data to WGAIN np.ndarray shape [num_examples, max_seq_len, num_features].
        padding_mask np.ndarray shape [num_examples, max_seq_len, num_features].
    """
    no = len(dat)
    dim = dat[0].shape[1]
    output = - np.ones(shape = (no, max_seq_len, dim))
    padding_mask = np.ones(shape = (no, max_seq_len, dim))
    for i in range(no):
        seq_len, dim = dat[i].shape
        output[i, 0:seq_len, :] = dat[i]
        padding_mask[i, 0:seq_len, :] = 0.
    return output, padding_mask

def bootstrap_results(model, input_data, seq_times):
  """Bootstrap original data and predict.

    Args:
        model: a pre-trained G(E(x)) model
        input data: Input data np.ndarray shape [num_examples, max_seq_len, num_features].
        seq_times : T dimension
    Returns:
        simulated_data np.ndarray shape [num_examples, max_seq_len, num_features].
        simulated_padding_mask np.ndarray shape [num_examples, max_seq_len, num_features].
    """
    no, seq_len, dim = input_data.shape
    boots =  resample(range(no), replace=True, n_samples=no)
    simulated_data = - np.ones(shape = (no, seq_len, dim))
    simulated_padding_mask = np.ones(shape = (no, seq_len, dim))
    for i in range(no):
        boot = boots[i]
        ori_data = input_data[boot:(1+boot)]
        reconstructed_samples = model(ori_data, training = True)
        reconstructed_samples = reconstructed_samples.numpy()
        reconstructed_samples[:, :, 0] = np.abs(reconstructed_samples[:, :, 0])
        time_order = np.argsort(reconstructed_samples[0, :, 0])
        reconstructed_samples = reconstructed_samples[:, time_order, :]
        start_len = seq_len - seq_times[boot]
        simulated_data[i, start_len:, :] = reconstructed_samples[0, :seq_times[boot], :]
        simulated_padding_mask[i, start_len:, :] = 0.
    return simulated_data, simulated_padding_mask

def post_process_rgan(obj):
  """RGAN output Post-process.

    Args:
        obj: Input data np.ndarray shape [num_examples, max_seq_len, num_features].
        the output from AE-WGAN network

    Returns:
        Return format is:
            np.ndarray shape [num_examples, max_seq_len, num_features].
        returns simulated data, with mask.
    """
    def sigmoid(x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    vsigmoid = np.vectorize(sigmoid)
    no, seq_len, dim = obj.shape
    half_dim = np.array(dim / 2).astype('int')
    simulated_data = obj[:,:,:( 1 + half_dim )]
    simulated_mask = obj[:,:,(1 + half_dim):]
    simulated_mask = vsigmoid(simulated_mask)
    simulated_mask = bernoulli.rvs(simulated_mask)
    simulated_mask = np.concatenate((np.zeros((no, seq_len, 1)).astype('int'), simulated_mask), axis = 2)
    simulated_data[(simulated_mask == 1)] = np.nan
    return simulated_data

# --------------------------------------- #

def hider(input_dict: Dict) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Solution hider function.

    Args:
        input_dict (Dict): Dictionary that contains the hider function inputs, as below:
            * "seed" (int): Random seed provided by the competition, use for reproducibility.
            * "data" (np.ndarray of float): Input data, shape [num_examples, max_seq_len, num_features].
            * "padding_mask" (np.ndarray of bool): Padding mask of bools, same shape as data.

    Returns:
        Return format is:
            np.ndarray (of float) [, np.ndarray (of bool), n_seeds]
        first argument is the hider generated data, expected shape [num_examples, max_seq_len, num_features]);
        second argument is the corresponding padding mask, same shape.
		third argument is the number of seeds in the hider evaluation step as commented in Slack for faster run-time evaluations in the platform.
        
    """

    # Get the inputs.
    seed = input_dict["seed"]  # Random seed provided by the competition, use for reproducibility.
    data = input_dict["data"]  # Input data, shape [num_examples, max_seq_len, num_features].
    padding_mask = input_dict["padding_mask"]  # Padding mask of bools, same shape as data.

    # Process data
    norm_data, max_value, min_value = MinMaxScaler(data.copy())
    mask_data = (np.isnan(norm_data)).astype("float32")
    mask_data[padding_mask] = -1.
    norm_data[padding_mask] = -1.
    no, seq_le, dim = norm_data.shape

    interp_dat, median_val, all_nan = interpolation_fun(norm_data.copy(), padding_mask)
    complete_data = interp_dat[:,:,all_nan != 1]
    mask_data = mask_data[:,:,all_nan != 1]
    padding_mask = padding_mask[:,:,all_nan != 1]

    # ------------------------------------- #
    # Set hyper-parameters for imputation network #
    # ------------------------------------- #
    batch_size = 128
    alpha = 10
    n_critic = 5
    n_epoch = 10

    imputed_data, padding_mask, seq_times = imputation(complete_data, mask_data, padding_mask, batch_size, n_epoch, alpha, n_critic)

    # ------------------------------------- #
    # Set hyper-parameters for rae_wgan network #
    # ------------------------------------- #
    n_epoch = 15
    latent_dim = 42
    n_critic = 5
    batch_size = 64

    res, simulated_padding_mask = rae_wgan(imputed_data, seq_times, padding_mask, batch_size, n_epoch, n_critic, latent_dim)

    # post process results
    simulated_padding_mask = simulated_padding_mask.astype('bool')
    res = post_process_rgan(res)
    res = DeScaler(res, max_value, min_value, all_nan)
    no, seq_len, dim = res.shape
    simulated_padding_mask = simulated_padding_mask[:,:,:dim]
    res[simulated_padding_mask] = -1.

    return res, simulated_padding_mask, 3
