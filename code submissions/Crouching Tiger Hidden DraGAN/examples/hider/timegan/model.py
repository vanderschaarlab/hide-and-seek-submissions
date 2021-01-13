# Necessary Packages
import tensorflow as tf
import numpy as np
import os
from .utils import extract_time, rnn_cell, random_generator, batch_generator, random_generator_notime, batch_generator_notime
from tqdm import tqdm
from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized, dp_optimizer
from datetime import datetime
from tensorflow.python.saved_model.simple_save import simple_save

"""Hide and seek challenge Codebase.

This file contains a lot of unnecessary code from other attempts at solutions. The solution that is actually being run
starts on line 809.
"""

def cnn(X, hidden_dim, last_activation="linear", padding="same", num_layers=3):
  final_layer = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, strides=1, dilation_rate=2**num_layers,
                                       padding=padding, activation=last_activation)
  for dr in [2**i for i in range(num_layers - 1)]:
    X = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, strides=1, dilation_rate=dr, padding=padding,
                                  activation='relu')(X)
  return final_layer(X)


def cnntimegan(ori_data, dp=False, l2_norm_clip=1., noise_multiplier=1., num_microbatches=32,
               batch_size=128, iterations=20000,
               architecture={"embedder": "rnn", "recovery": "rnn",
                             "generator": "rnn", "supervisor": "rnn",
                             "discriminator": "rnn"},
               logdir='', id='', output_while_training=0, postprocessor=None,
               vectorised=False):
  """
  NOT RUN
  Modification of TimeGAN function.
  """
  parameters = dict()
  parameters['module'] = 'gru'
  parameters['hidden_dim'] = 20
  parameters['num_layer'] = 3
  if vectorised:
    raise NotImplementedError(
      'Not currently working because G_loss_V is wrong shape (need to figure out how to vectorise means + vars)')
  if not id:
    id = 'cnntimegan_' + datetime.now().strftime('%y%m%d %H%M')
  
  if not logdir:
    logdir = 'logs/{}_loss'.format(id)
  
  if output_while_training:
    os.makedirs(os.path.join('output', id), exist_ok=True)
    os.makedirs(os.path.join('saved', id), exist_ok=True)
  
  # Initialization on the Graph
  tf.reset_default_graph()
  
  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
  
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    norm_data = norm_data * 2 - 1
    
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
  
  ## Build a RNN networks
  
  # Network Parameters
  hidden_dim = parameters['hidden_dim']
  num_layers = parameters['num_layer']
  module_name = parameters['module']
  z_dim = dim
  gamma = 1
  
  def layer_architecture(X, T, type, last_activation="relu"):
    """
    Returns either rnn, cnn or causal cnn architecture
    Args:
      X: Input layer
      T: time layer
      type: Architecture type: "rnn", "cnn" or "ccnn"
      last_activation: If type is "cnn" or "ccnn", the activation function to apply to the last layer

    Returns:

    """
    if type == "rnn":
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, X, dtype=tf.float32, sequence_length=T)
    elif type == "cnn":
      r_outputs = cnn(X, hidden_dim=hidden_dim, last_activation=last_activation, padding="same")
    elif type == "ccnn":
      r_outputs = cnn(X, hidden_dim=hidden_dim, last_activation=last_activation, padding="causal")
    else:
      raise ValueError("{} is not a valid value for type".format(type))
    return r_outputs
  
  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
  T = tf.placeholder(tf.int32, [None], name="myinput_t")
  
  def embedder(X, T):
    """Embedding network between original feature space to latent space.

    Args:
      - X: input time-series features
      - T: input time information

    Returns:
      - H: embeddings
    """
    with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
      e_outputs = layer_architecture(X, T, architecture["embedder"], last_activation="linear")
      H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
      H = tf.identity(H, "output_h")
    return H
  
  def recovery(H, T):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """
    with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
      r_outputs = layer_architecture(H, T, architecture["recovery"], last_activation="relu")
      X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=None)
      X_tilde = tf.identity(X_tilde, "output_x")
    return X_tilde
  
  def generator(Z, T):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs = layer_architecture(Z, T, architecture["generator"], last_activation="linear")
      E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
      E = tf.identity(E, "output_e")
    return E
  
  def supervisor(H, T):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
      e_outputs = layer_architecture(H, T, architecture["supervisor"], last_activation="linear")
      S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
      S = tf.identity(S, "output_s")
    return S
  
  def discriminator(H, T):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
      d_outputs = layer_architecture(H, T, architecture["discriminator"], last_activation="relu")
      Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None)
      Y_hat = tf.identity(Y_hat, "output_y")
    return Y_hat
  
  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)
  
  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
  
  # Synthetic data
  X_hat = recovery(H_hat, T)
  
  # Discriminator
  Y_fake = discriminator(H_hat, T)
  Y_real = discriminator(H, T)
  Y_fake_e = discriminator(E_hat, T)
  
  # Variables
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
  
  # Discriminator loss
  
  if vectorised:
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real, reduction=tf.losses.Reduction.NONE)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake, reduction=tf.losses.Reduction.NONE)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e,
                                                    reduction=tf.losses.Reduction.NONE)
  else:
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
  
  # Generator loss
  # # 1. Adversarial loss
  
  if vectorised:
    G_loss_U = tf.reduce_mean(
      tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake, reduction=tf.losses.Reduction.NONE),
      axis=1)
    G_loss_U_e = tf.reduce_mean(
      tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e, reduction=tf.losses.Reduction.NONE),
      axis=1)
  else:
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
  
  # # 2. Supervised loss
  if vectorised:
    G_loss_S = tf.reduce_mean(
      tf.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :], reduction=tf.losses.Reduction.NONE), [1, 2])
  else:
    G_loss_S = tf.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
  
  # # 3. Two Momments
  if vectorised:
    G_loss_V1 = tf.reduce_mean(
      tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))
  else:
    G_loss_V1 = tf.reduce_mean(
      tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    # G_loss_V1 = tf.reduce_mean(tf.sqrt(tf.abs(tfp.stats.covariance(X_hat) - tfp.stats.covariance(X))))

    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))
  
  G_loss_V = G_loss_V1 + G_loss_V2
  
  # 4. Summation
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
  
  # # Embedder network loss
  if vectorised:
    E_loss_T0 = tf.reduce_sum(
      tf.losses.mean_squared_error(X, X_tilde, reduction=tf.losses.Reduction.NONE),
      axis=[1, 2])
  else:
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  
  E_loss0 = 10 * tf.sqrt(E_loss_T0)
  E_loss = E_loss0 + 0.1 * G_loss_S
  
  # optimizer
  if dp:
    ledger = privacy_ledger.PrivacyLedger(
      population_size=no,
      selection_probability=(batch_size / no))
    opt_e = dp_optimizer.DPAdamGaussianOptimizer(
      l2_norm_clip=l2_norm_clip['e'],
      noise_multiplier=noise_multiplier['e'],
      num_microbatches=num_microbatches,
      unroll_microbatches=True,  # Unrolling doesn't work with vectorised optimizer
      ledger=ledger  # Ledger doesnt work with vectorised optimizer (might have to calculate non-parallelised
    )
    opt_g = dp_optimizer.DPAdamGaussianOptimizer(
      l2_norm_clip=l2_norm_clip['g'],
      noise_multiplier=noise_multiplier['g'],
      num_microbatches=num_microbatches,
      unroll_microbatches=True,  # Unrolling doesn't work with vectorised optimizer
      ledger=ledger  # Ledger doesnt work with vectorised optimizer (might have to calculate non-parallelised
    )
    opt_d = dp_optimizer.DPAdamGaussianOptimizer(
      l2_norm_clip=l2_norm_clip['d'],
      noise_multiplier=noise_multiplier['d'],
      num_microbatches=num_microbatches,
      unroll_microbatches=True,  # Unrolling doesn't work with vectorised optimizer
      ledger=ledger  # Ledger doesnt work with vectorised optimizer (might have to calculate non-parallelised
    )
    E0_solver = opt_e.minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = opt_e.minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = opt_d.minimize(D_loss, var_list=d_vars)
    G_solver = opt_g.minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)
  else:
    opt = tf.train.AdamOptimizer()
    E0_solver = opt.minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = opt.minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = opt.minimize(D_loss, var_list=d_vars)
    G_solver = opt.minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = opt.minimize(G_loss_S, var_list=g_vars + s_vars)
  
  # Write training loss to logs
  # Todo: remove unused summaries
  ET0_summary = tf.compat.v1.summary.scalar('E_loss_T0', tf.math.reduce_mean(E_loss_T0))
  GS_summary = tf.compat.v1.summary.scalar('G_loss_S', tf.math.reduce_mean(G_loss_S))
  GU_summary = tf.compat.v1.summary.scalar('G_loss_U', tf.math.reduce_mean(G_loss_U))
  GV_summary = tf.compat.v1.summary.scalar('G_loss_V', tf.math.reduce_mean(G_loss_V))
  G_summary = tf.compat.v1.summary.scalar('G_loss', tf.math.reduce_mean(G_loss))
  GV1_summary = tf.compat.v1.summary.scalar('G_mean_diff_loss', tf.math.reduce_mean(G_loss_V1))
  GV2_summary = tf.compat.v1.summary.scalar('G_sd_diff_loss', tf.math.reduce_mean(G_loss_V2))
  D_summary = tf.compat.v1.summary.scalar('D_loss', tf.math.reduce_mean(D_loss))
  
  # summary_writer = tf.compat.v1.summary.FileWriter(logdir)
  
  ## TimeGAN training
  from tensorflow import ConfigProto
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  
  # 1. Embedding network training
  print('Start Embedding Network Training')
  
  for itt in tqdm(range(iterations)):
    # Set mini-batch
    X_mb, T_mb = batch_generator(batch_size, ori_data, ori_time)
    # Train embedder
    _, step_e_loss, step_e_summ = sess.run([E0_solver, E_loss_T0, ET0_summary], feed_dict={X: X_mb, T: T_mb})
    # Checkpoint
    if itt % 1000 == 0:
      print('step: ' + str(itt) + '/' + str(iterations) + ', e_loss: ' + str(
        np.round(np.mean(np.sqrt(step_e_loss)), 4)))
  print('Finish Embedding Network Training')
  
  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')
  
  for itt in tqdm(range(iterations)):
    # Set mini-batch
    X_mb, T_mb = batch_generator(batch_size, ori_data, ori_time)
    # Random vector generation
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Train generator
    _, step_g_loss_s, step_gs_summary = sess.run([GS_solver, G_loss_S, GS_summary],
                                                 feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
    # Checkpoint
    if itt % 1000 == 0:
      print('step: ' + str(itt) + '/' + str(iterations) + ', s_loss: ' + str(
        np.round(np.mean(np.sqrt(step_g_loss_s)), 4)))
  print('Finish Training with Supervised Loss Only')
  
  # 3. Joint Training
  print('Start Joint Training')
  
  for itt in tqdm(range(iterations)):
    # Generator training (twice more than discriminator training)
    for kk in range(2):
      # Set mini-batch
      X_mb, _ = batch_generator(batch_size, ori_data, ori_time)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      (_, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_g_summary, step_gu_summary, step_gs_summary,
       step_gv_summary, step_gv1_summary, step_gv2_summary) = sess.run(
        [G_solver, G_loss_U, G_loss_S, G_loss_V, G_summary, GU_summary, GS_summary, GV_summary, GV1_summary,
         GV2_summary], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
      # Train embedder
      _, step_e_loss_t0, step_e_summ = sess.run([E_solver, E_loss_T0, ET0_summary],
                                                feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
    
    # Discriminator training
    # Set mini-batch
    X_mb, T_mb = batch_generator(batch_size, ori_data, ori_time)
    
    # Random vector generation
    
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    
    # Check discriminator loss before updating
    check_d_loss, step_d_summary = sess.run([D_loss, D_summary], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    
    # Train discriminator (only when the discriminator does not work well)
    if (np.mean(check_d_loss) > 0.15):
      _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    
    # Output data
    # TODO: Also save model weights so I can reload if needed
    if output_while_training and itt % output_while_training == 0:
      Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
      generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
      generated_data = list()
      
      for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)
      
      # Renormalization
      generated_data = (generated_data + 1.) / 2.
      generated_data = generated_data * max_val
      generated_data = generated_data + min_val
      
      if postprocessor is not None:
        output_data = postprocessor.postprocess(generated_data)
        output_data.to_csv(os.path.join('output', id, 'idx{}.csv'.format(itt)))
      else:
        print('Need to pass postprocessor to cnntimegan. Not implemented otherwise')
      
      # Save model
      simple_save(sess, os.path.join('saved', id, 'idx{}'.format(itt)),
                  inputs={"x": X, "t": T, "z": Z},
                  outputs={"x_hat": X_hat})
    
    # Print multiple checkpoints
    if itt % 1000 == 0:
      print('step: ' + str(itt) + '/' + str(iterations) +
            ', d_loss: ' + str(np.round(np.mean(step_d_loss), 4)) +
            ', g_loss_u: ' + str(np.round(np.mean(step_g_loss_u), 4)) +
            ', g_loss_s: ' + str(np.round(np.mean(np.sqrt(step_g_loss_s)), 4)) +
            ', g_loss_v: ' + str(np.round(np.mean(step_g_loss_v), 4)) +
            ', e_loss_t0: ' + str(np.round(np.mean(np.sqrt(step_e_loss_t0)), 4)))
  print('Finish Joint Training')
  
  ## Synthetic data generation
  Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
  
  generated_data = list()
  
  for i in range(no):
    temp = generated_data_curr[i, :ori_time[i], :]
    generated_data.append(temp)
  
  # Renormalization
  generated_data = generated_data * max_val
  generated_data = generated_data + min_val
  
  # Save model
  simple_save(sess, os.path.join('saved', id, 'idx{}'.format(iterations)),
              inputs={"x": X, "t": T, "z": Z},
              outputs={"x_hat": X_hat})
  
  return generated_data


def notimegan(ori_data, dp=False, l2_norm_clip=1., noise_multiplier=1., num_microbatches=32,
               batch_size=128, iterations=20000,
               logdir='', id='', output_while_training=0, postprocessor=None,
               vectorised=False):
  """
  NOT RUN IN FINAL SOLUTION
  Implementation of wgan

  Args:
    - ori_data: original time-series data
    - dp: Whether differentially private optimizer should be used to train model
    - l2_norm_clip: weight clipping parameter for dp optimizer
    - noise_multiplier: noise parameter for dp optimizer
    - num_microbatches: number microbatches to be used when training with dp optimizer
    - batch_size: batch size
    - iterations: Number of iterations to train for
    - logdir: directory for keeping track of training losses
    - id: string to identify different runs
    - output_while_training:int: will output data to a file everyoutput_while_training iterations
    - postprocessor: Postprocessor class for outputting data while training
    - vectorised: WNot implemented. Whether loss should be vectorised or not (required for num_microbatches > 1)

  Returns:
    - generated_data: generated time-series data
  """
  parameters = dict()
  parameters['module'] = 'gru'
  parameters['hidden_dim'] = 20
  parameters['num_layer'] = 3

  if vectorised:
    raise NotImplementedError(
      'Not currently working because G_loss_V is wrong shape (need to figure out how to vectorise means + vars)')
  if not id:
    id = 'cnntimegan_' + datetime.now().strftime('%y%m%d %H%M')
  
  if not logdir:
    logdir = 'logs/{}_loss'.format(id)
  
  if output_while_training:
    os.makedirs(os.path.join('output', id), exist_ok=True)
    os.makedirs(os.path.join('saved', id), exist_ok=True)
  
  # Initialization on the Graph
  tf.reset_default_graph()
  
  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
  
  # Extract one row from each patient - Later on should do this in batch generator so I can randomise which row
  ori_data = ori_data.reshape((-1, ori_data.shape[-1]))

  def fcn(X, final_dim, last_activation="linear", batch_norm=True):
    bn = tf.keras.layers.BatchNormalization()
    layer1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
    layer2 = tf.keras.layers.Dense(hidden_dim, activation="relu")
    layer3 = tf.keras.layers.Dense(final_dim, activation=last_activation)
    if batch_norm:
      return layer3(bn(layer2(bn(layer1(X)))))
    else:
      return layer3(layer2(layer1(X)))
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
  
  ## Build a RNN networks
  
  # Network Parameters
  hidden_dim = parameters['hidden_dim']
  num_layers = parameters['num_layer']
  # iterations   = parameters['iterations']
  # batch_size   = parameters['batch_size']
  module_name = parameters['module']
  z_dim = dim
  gamma = 1
  
  # Input place holders
  X = tf.placeholder(tf.float32, [None, dim], name="myinput_x")
  Z = tf.placeholder(tf.float32, [None, z_dim], name="myinput_z")
  
  def embedder(X):
    """Embedding network between original feature space to latent space.

    Args:
      - X: input time-series features

    Returns:
      - H: embeddings
    """
    with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
      H = fcn(X, hidden_dim, last_activation=tf.nn.sigmoid)
      H = tf.identity(H, "output_h")
    return H
  
  def recovery(H):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation

    Returns:
      - X_tilde: recovered data
    """
    with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
      X_tilde = fcn(H, dim, last_activation="tanh", batch_norm=True)
      X_tilde = tf.identity(X_tilde, "output_x")
    return X_tilde
  
  def generator(Z):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
      E = fcn(Z, hidden_dim, last_activation=tf.nn.sigmoid)
      E = tf.identity(E, "output_e")
    return E
  
  def supervisor(H):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
      S = fcn(H, hidden_dim, last_activation=tf.nn.sigmoid)
      S = tf.identity(S, "output_s")
    return S
  
  def discriminator(H):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
      Y_hat = fcn(H, 1,batch_norm=False)
      Y_hat = tf.identity(Y_hat, "output_y")
    return Y_hat
  
  # Embedder & Recovery

  X_hat = recovery(Z)
  # Discriminator
  Y_fake = discriminator(X_hat)
  Y_real = discriminator(X)
  
  # Variables
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
  
  # Discriminator loss
  # Wasserstein modification
  D_loss = tf.reduce_mean(Y_fake) - tf.reduce_mean(Y_real)
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = -tf.reduce_mean(Y_fake)
  G_loss = G_loss_U
  
  # optimizer
  if dp:
    ledger = privacy_ledger.PrivacyLedger(
      population_size=no,
      selection_probability=(batch_size / no))
    # Wasserstein modification
    opt_d = dp_optimizer.DPAdamGaussianOptimizer(
      l2_norm_clip=l2_norm_clip['d'],
      noise_multiplier=noise_multiplier['d'],
      num_microbatches=num_microbatches,
      learning_rate=1e-4,
      beta1=0.5, beta2=.9,
      ledger=ledger  # Ledger doesnt work with vectorised optimizer (might have to calculate non-parallelised
    )
    # Instead of clipping apply Gradient penalty
    alpha = tf.random_uniform(
        shape=[batch_size,1],
        minval=0.,
        maxval=1.
    )
    differences = X_hat - X
    interpolates = X + (alpha*differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])
    print(gradients)
    gradients = gradients[0]
    print(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    LAMBDA = 10
    D_loss += LAMBDA*gradient_penalty

    D_solver = opt_d.minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=.9).minimize(G_loss, var_list=r_vars)

  else:
    print('Not implemented')
    exit()
    opt = tf.train.AdamOptimizer()
    D_solver = opt.minimize(D_loss, var_list=d_vars)
  
  # Write training loss to logs
  G_summary = tf.compat.v1.summary.scalar('G_loss', tf.math.reduce_mean(G_loss))
  D_summary = tf.compat.v1.summary.scalar('D_loss', tf.math.reduce_mean(D_loss))
  
  ## TimeGAN training
  from tensorflow import ConfigProto
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  
  # 1. Joint Training
  print('Start Joint Training')
  
  for itt in tqdm(range(iterations)):
    # Generator training (twice more than discriminator training)
    if itt > 0:
      # Set mini-batch
      X_mb = batch_generator_notime(ori_data, batch_size)
      # Random vector generation
      Z_mb = random_generator_notime(batch_size, z_dim)
      # Train generator
      (_, step_g_loss, step_g_summary) = sess.run(
        [G_solver, G_loss, G_summary], feed_dict={Z: Z_mb #, X: X_mb
                                                     })
    
    # Discriminator training
    # Set mini-batch
    for kk in range(5):
      X_mb = batch_generator_notime(ori_data, batch_size)
      
      # Random vector generation
      Z_mb = random_generator_notime(batch_size, z_dim)
      
      # Train discriminator (only when the discriminator does not work well)
      _, step_d_loss, step_d_summary = sess.run([D_solver, D_loss, D_summary], feed_dict={X: X_mb, Z: Z_mb})
    
    # Output data
    # TODO: Also save model weights so I can reload if needed
    if output_while_training and itt % output_while_training == 0:
      Z_mb = random_generator_notime(no, z_dim)
      generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data})
      generated_data = list()

      for i in range(no):
        temp = generated_data_curr[i, :]
        generated_data.append(temp)

      # Renormalization
      generated_data = generated_data * max_val
      generated_data = generated_data + min_val

      if postprocessor is not None:
        output_data = postprocessor.postprocess(generated_data)
        output_data.to_csv(os.path.join('output', id, 'idx{}.csv'.format(itt)))
      else:
        print('Need to pass postprocessor to cnntimegan. Not implemented otherwise')

      # Save model
      simple_save(sess, os.path.join('saved', id, 'idx{}'.format(itt)),
                  inputs={"x": X, "z": Z},
                  outputs={"x_hat": X_hat})
    
    # Print multiple checkpoints
    if (itt % 1000 == 0) & (itt > 0):
      print('step: ' + str(itt) + '/' + str(iterations) +
            ', d_loss: ' + str(np.round(np.mean(step_d_loss), 4)) +
            ', g_loss: ' + str(np.round(np.mean(step_g_loss), 4))
            )
  print('Finish Joint Training')
  
  ## Synthetic data generation
  Z_mb = random_generator_notime(no, z_dim)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data})
  
  generated_data = list()
  
  for i in range(no):
    temp = generated_data_curr[i, :]
    generated_data.append(temp)
  
  # Renormalization
  print(np.array(generated_data).shape)
  generated_data = np.array(generated_data) * max_val
  generated_data = generated_data + min_val
  
  # Turn into sequence
  no, p = generated_data.shape
  temp_sequence = np.zeros([no, seq_len, p])
  
  for i in range(no):
    temp_sequence[i, :, :] = np.stack([generated_data[i, :]] * seq_len)
    temp_sequence[i, :, 0] = np.arange(seq_len) + 1
  
  generated_data = temp_sequence
  
  # Save model
  simple_save(sess, os.path.join('saved', id, 'idx{}'.format(iterations)),
              inputs={"x": X, "z": Z},
              outputs={"x_hat": X_hat})
  
  return generated_data

"""Hide and seek challenge model Codebase.

This model does not treat the data as a sequence. Instead a single timepoint from each patient visit is selected and this
simplified dataset is generated using a Wasserstein GAN with a gradient penalty applied to enforce a Lipshitz constaint.

Reference for method: Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
"Improved Training of Wasserstein GANs"
Neural Information Processing Systems (NeurIPS), 2017.
Paper link: https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
Codebase:
https://github.com/igul222/improved_wgan_training
Author: Ishaan Gulrajani

Modifications to above codebase for challenge made by: Flynn Gewirtz-O'Reilly
-----------------------------

timegan.py

Note: Use original data as training set to generate synthetic data (time-series)
"""
# Files from tflib
def print_model_settings(locals_):
  """
  Print list of model settings
  Args:
    locals_:
  """
  print("Uppercase local vars:")
  all_vars = [(k,v) for (k,v) in locals_.items() if (k.isupper() and k!='T' and k!='SETTINGS' and k!='ALL_SETTINGS')]
  all_vars = sorted(all_vars, key=lambda x: x[0])
  for var_name, var_value in all_vars:
      print ("\t{}: {}".format(var_name, var_value))


_params = {}
_param_aliases = {}


def param(name, *args, **kwargs):
  """
  A wrapper for `tf.Variable` which enables parameter sharing in models.

  Creates and returns theano shared variables similarly to `tf.Variable`,
  except if you try to create a param with the same name as a
  previously-created one, `param(...)` will just return the old one instead of
  making a new one.
  This constructor also adds a `param` attribute to the shared variables it
  creates, so that you can easily search a graph for all params.
  """
  
  if name not in _params:
    kwargs['name'] = name
    param = tf.Variable(*args, **kwargs)
    param.param = True
    _params[name] = param
  result = _params[name]
  i = 0
  while result in _param_aliases:
    i += 1
    result = _param_aliases[result]
  return result

# Whether Layer Normalization should be applied
_default_weightnorm = False

def enable_default_weightnorm():
  """Not used"""
  global _default_weightnorm
  _default_weightnorm = True

def disable_default_weightnorm():
  """Not used"""
  global _default_weightnorm
  _default_weightnorm = False

# _weights_stdev is unused parameter for overriding weight initialisations
_weights_stdev = None

def set_weights_stdev(weights_stdev):
  global _weights_stdev
  _weights_stdev = weights_stdev

def unset_weights_stdev():
  global _weights_stdev
  _weights_stdev = None


def Linear(
    name,
    input_dim,
    output_dim,
    inputs,
    biases=True,
    initialization=None,
    weightnorm=None,
    gain=1.
):
  """
  Initialise layer of neural network with linear activation function.
  Args:
    name: Layer name
    input_dim: Number of input dimensions
    output_dim: Number of output dimensions
    inputs: Preceeding Layer
    biases: Whether bias should be added to weight vector.
    initialization: Type of weight initialization (default: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`,
                    `("uniform", range)`)
    weightnorm: Whether layer normalization should be applied (Reference: Ba, Kiros & Hinton (2016))
    gain: Gain parameter for scaling the normalized activation function (See Layer Normalization by Ba, Kiros & Hinton
          (2016))

  Returns: Single NN layer with linear output
  """
  with tf.name_scope(name) as scope:
    # Initialize weights
    def uniform(stdev, size):
      if _weights_stdev is not None:
        stdev = _weights_stdev
      return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
      ).astype('float32')
    
    if initialization == 'lecun':  # and input_dim != output_dim):
      # disabling orth. init for now because it's too slow
      weight_values = uniform(
        np.sqrt(1. / input_dim),
        (input_dim, output_dim)
      )
    
    elif initialization == 'glorot' or (initialization == None):
      
      weight_values = uniform(
        np.sqrt(2. / (input_dim + output_dim)),
        (input_dim, output_dim)
      )
    
    elif initialization == 'he':
      
      weight_values = uniform(
        np.sqrt(2. / input_dim),
        (input_dim, output_dim)
      )
    
    elif initialization == 'glorot_he':
      
      weight_values = uniform(
        np.sqrt(4. / (input_dim + output_dim)),
        (input_dim, output_dim)
      )
    
    elif initialization == 'orthogonal' or \
        (initialization == None and input_dim == output_dim):
      
      # From lasagne
      def sample(shape):
        if len(shape) < 2:
          raise RuntimeError("Only shapes of length 2 or more are "
                             "supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        # TODO: why normal and not uniform?
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q.astype('float32')
      
      weight_values = sample((input_dim, output_dim))
    
    elif initialization[0] == 'uniform':
      
      weight_values = np.random.uniform(
        low=-initialization[1],
        high=initialization[1],
        size=(input_dim, output_dim)
      ).astype('float32')
    
    else:
      
      raise Exception('Invalid initialization!')
    
    weight_values *= gain
    
    weight = param(
      name + '.W',
      weight_values
    )
    
    if weightnorm == None:
      weightnorm = _default_weightnorm
    if weightnorm:
      norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
      target_norms = param(
        name + '.g',
        norm_values
      )
      
      with tf.name_scope('weightnorm') as scope:
        norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
        weight = weight * (target_norms / norms)
    
    if inputs.get_shape().ndims == 2:
      result = tf.matmul(inputs, weight)
    else:
      reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
      result = tf.matmul(reshaped_inputs, weight)
      result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))
    
    if biases:
      result = tf.nn.bias_add(
        result,
        param(
          name + '.b',
          np.zeros((output_dim,), dtype='float32')
        )
      )
    
    return result

def params_with_name(name):
  """ Print parameters that contain name """
  return [p for n, p in _params.items() if name in n]


def delete_all_params():
  """ Unused function """
  _params.clear()


def alias_params(replace_dict):
  """ Unused function"""
  for old, new in replace_dict.items():
    # print "aliasing {} to {}".format(old,new)
    _param_aliases[old] = new


def delete_param_aliases():
  """ Unused function"""
  _param_aliases.clear()
  
def wgangp(ori_data, dp=False, l2_norm_clip=1., noise_multiplier=1., num_microbatches=32,
               batch_size=128, iterations=20000,
               logdir='', id='', output_while_training=0, postprocessor=None,
               vectorised=False, codalab=True):
  """
  Train Wasserstein to generate data from preprocessed sequence of data.
  Args:
    ori_data: Amsterdam data
    dp: Whether differentially private optimizer should be used to train the model (not used)
    l2_norm_clip: Weight clipping parameter for differentially private training
    noise_multiplier: Noise parameter for differentially private training
    num_microbatches: Microbatch parameter for differentially private training (not implemented)
    batch_size: Number of samples per batch
    iterations: Number of iterations to train over
    logdir: directory for keeping track of training losses
    id: string to identify different runs
    codalab: If true will not run filewriter lines as they break the docker containers

    Unused parameters (should be removed)
    output_while_training: will output data to a file every output_while_training iterations
    postprocessor: Postprocessor class for outputting unscaled data while training
    vectorised: Whether loss should be vectorised or not (required for num_microbatches > 1)

  Returns: Synthetic 'sequence' of data.

  """
  import os, sys
  sys.path.append(os.getcwd())
  
  import random
  import time
  
  from sklearn.preprocessing import MinMaxScaler, StandardScaler
  # Whether Lipschitz constraint should be enforced with weight clipping ('wgan') or gradient penality ('wgan-gp')
  MODE = 'wgan-gp'
  DATASET = '8gaussians'  # Unused variable, should be removed
  DIM = 1024  # Number of neurons per hidden layer
  FIXED_GENERATOR = False  # If true generator would just add noise to data
  LAMBDA = 10  # Larger values of lambda increase the gradiant penalty that is applied per iteration
  CRITIC_ITERS = 100  # How many critic iterations per generator iteration
  
  # For each run of model set a model and create an output directory for the logs
  if not id:
    id = 'wgangp' + datetime.now().strftime('%y%m%d %H%M')
  if not logdir:
    logdir = 'logs/{}_loss'.format(id)
  if not codalab:
    summary_writer = tf.compat.v1.summary.FileWriter(logdir)

  # drop time variable from data
  ori_data = np.asarray(ori_data)[:, :, 1:]
  # Get dimensions of input data
  no, seq_len, dim = np.asarray(ori_data).shape


  # Normalization
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    Use Min-max scaling to scale data between -1 and 1
    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    # Generalised minmax function to work for 2d and 3d (assumes that last axis is variable axis)
    min_val = data
    max_val = data
    for i in range(len(data.shape) - 1):
      min_val = np.min(min_val, axis=0)
      max_val = np.max(max_val, axis=0)

    data = data - min_val
    norm_data = data / (max_val + 1e-7)
    norm_data = norm_data * 2. - 1. # -1, 1 scaling
    return norm_data, min_val, max_val


  # Remove sequential aspect of data by extracting last row from visit
  ori_data = ori_data[:, -1, :]
  
  ori_data, min_val, max_val = MinMaxScaler(ori_data)

  print_model_settings(locals().copy())
  
  def ReLULayer(name, n_in, n_out, inputs):
    """Apply ReLU activation to output of Linear layer"""
    output = Linear(
      name + '.Linear',
      n_in,
      n_out,
      inputs,
      initialization='he'
    )
    output = tf.nn.relu(output)
    return output
  
  def tanhLayer(name, n_in, n_out, inputs):
    """Apply tanh activation to output of Linear layer"""
    output = Linear(
      name + '.Linear',
      n_in,
      n_out,
      inputs,
      initialization='he'
    )
    output = tf.nn.tanh(output)
    return output
  
  def Generator(n_samples, real_data):
    """ Define architecture of Generator"""
    if FIXED_GENERATOR:
      # Not used, but would add noise to data instead of generating with neural network
      return real_data + (1. * tf.random_normal(tf.shape(real_data)))
    else:
      noise = tf.random_normal([n_samples, dim])
      output = ReLULayer('Generator.1', dim, DIM, noise)
      output = ReLULayer('Generator.2', DIM, DIM, output)
      output = ReLULayer('Generator.3', DIM, DIM, output)
      output = tanhLayer('Generator.4', DIM, dim, output)
      return output
  
  def Discriminator(inputs):
    """ Define architecture of Discriminator"""
    output = ReLULayer('Discriminator.1', dim, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])
  
  real_data = tf.placeholder(tf.float32, shape=[None, dim])
  fake_data = Generator(batch_size, real_data)
  
  # Apply discriminator to real and synthetic data
  disc_real = Discriminator(real_data)
  disc_fake = Discriminator(fake_data)
  
  # WGAN loss
  disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
  gen_cost = -tf.reduce_mean(disc_fake)
  
  # WGAN gradient penalty
  if MODE == 'wgan-gp':
    alpha = tf.random_uniform(
      shape=[batch_size, 1],
      minval=0.,
      maxval=1.
    )
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = Discriminator(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    
    disc_cost += LAMBDA * gradient_penalty
  
  disc_params = params_with_name('Discriminator')
  gen_params = params_with_name('Generator')
  
  if dp:
    # Ledger is used to keep track of the number of times the optimizer "sees" the the data
    ledger = privacy_ledger.PrivacyLedger(
      population_size=no,
      selection_probability=(batch_size / no))

  # Train discriminator and generator
  if MODE == 'wgan-gp':
    if dp:
      disc_train_op = dp_optimizer.DPAdamGaussianOptimizer(
        l2_norm_clip=l2_norm_clip['d'],
        noise_multiplier=noise_multiplier['d'],
        num_microbatches=num_microbatches,
        learning_rate=1e-4,
        beta1=0.5, beta2=.9,
        ledger=ledger  # Ledger doesnt work with vectorised optimizer (might have to calculate non-parallelised
      ).minimize(
        disc_cost,
        var_list=disc_params
      )
    else:
      disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
      ).minimize(
        disc_cost,
        var_list=disc_params
      )
    if len(gen_params) > 0:
      gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
      ).minimize(
        gen_cost,
        var_list=gen_params
      )
    else:
      gen_train_op = tf.no_op()
  
  else:
    if dp:
      disc_train_op = dp_optimizer.DPRMSPropGaussianOptimizer(
        l2_norm_clip=l2_norm_clip['d'],
        noise_multiplier=noise_multiplier['d'],
        num_microbatches=num_microbatches,
        learning_rate=5e-5,
        ledger=ledger  # Ledger doesnt work with vectorised optimizer (might have to calculate non-parallelised
      ).minimize(
        disc_cost,
        var_list=disc_params
      )
    else:
      disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost,
        var_list=disc_params
      )
    if len(gen_params) > 0:
      gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        gen_cost,
        var_list=gen_params
      )
    else:
      gen_train_op = tf.no_op()
    
    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
      clip_bounds = [-.01, .01]
      clip_ops.append(
        tf.assign(
          var,
          tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
        )
      )
    clip_disc_weights = tf.group(*clip_ops)
  
  print
  "Generator params:"
  for var in params_with_name('Generator'):
    print
    "\t{}\t{}".format(var.name, var.get_shape())
  print
  "Discriminator params:"
  for var in params_with_name('Discriminator'):
    print
    "\t{}\t{}".format(var.name, var.get_shape())
  
  frame_index = [0]
  
  # File writers to save D and G loss
  G_summary = tf.compat.v1.summary.scalar('G_loss', tf.math.reduce_mean(gen_cost))
  D_summary = tf.compat.v1.summary.scalar('D_loss', tf.math.reduce_mean(disc_cost))
  
 
  # Train loop
  with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for iteration in tqdm(range(iterations)):
      # Train generator
      if iteration > 0:
        _, step_g_summ = session.run([gen_train_op, G_summary])
      # Train critic
      for i in range(CRITIC_ITERS):
        X_mb = batch_generator_notime(ori_data, batch_size)
        _disc_cost, _, step_d_summ = session.run(
          [disc_cost, disc_train_op, D_summary],
          feed_dict={real_data: X_mb}
        )
        if MODE == 'wgan':
          _ = session.run([clip_disc_weights])
          
      if (iteration > 0) and (not codalab):
        summary_writer.add_summary(step_g_summ, iteration)
        summary_writer.add_summary(step_d_summ, iteration)

    # Generate data
    generated_data = session.run(Generator(no, real_data))
    print(generated_data[:5, :5]) # For checking output

    # Reverse minmax scaling
    generated_data = .5 * (np.array(generated_data) + 1 ) * max_val + min_val
    """Seekers expect the output to be in the form of a sequence, so the single timepoint for each visit is repeated
    seq_len times so that it is the correct shape and the time column is added back in (as a sequence from 0 to seq_len)
    """
    temp_sequence = np.zeros([no, seq_len, dim+1])
    for i in range(no):
      temp_sequence[i, :, 1:] = np.stack([generated_data[i, :]] * seq_len)
      temp_sequence[i, :, 0] = np.arange(seq_len) + 1

    generated_data = temp_sequence
    return generated_data
