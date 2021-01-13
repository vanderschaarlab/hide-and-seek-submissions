# """
# Nothing in this file is run as part of my submitted solution
# """
#
# # Necessary Packages
# import tensorflow as tf
# import tensorflow_probability as tfp
# import numpy as np
# import os
# from .utils import extract_time, rnn_cell, random_generator, batch_generator
# from tqdm import tqdm
# from tensorflow_privacy.privacy.analysis import privacy_ledger
# from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized, dp_optimizer
# from datetime import datetime
# from tensorflow.python.saved_model.simple_save import simple_save
#
#
# def cnn(X, hidden_dim, last_activation="linear", padding="same"):
#   layer1 = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, strides=1, dilation_rate=1, padding=padding,
#                                   activation="relu")
#   layer2 = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, strides=1, dilation_rate=2, padding=padding,
#                                   activation="relu")
#   layer3 = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=3, strides=1, dilation_rate=4, padding=padding,
#                                   activation=last_activation)
#
#   return layer3(layer2(layer1(X)))
#
#
# def fcblock(X, hidden_dim, last_activation="linear"):
#   layer1 = tf.keras.layers.Dense(units=hidden_dim, activation="relu")
#   layer2 = tf.keras.layers.Dense(units=hidden_dim, activation="relu")
#   layer3 = tf.keras.layers.Dense(units=hidden_dim, activation=last_activation)
#
#   return layer3(layer2(layer1(X)))
#
#
# def statictimegan(ori_data, ori_mask,
#                   batch_size=128, iterations=20000,
#                   architecture={"embedder": "rnn", "recovery": "rnn",
#                                 "generator": "rnn", "supervisor": "rnn",
#                                 "discriminator": "rnn"},
#                   logdir='', id='', output_while_training=0, postprocessor=None,
#                   codalab=True):
#   """TimeGAN function.
#
#   Use original data as training set to generater synthetic data (time-series)
#
#   Args:
#     - ori_data: original time-series data: dict with shape=(n, t, p)
#     - ori_mask: binary var with shape(n, p). Represents whether ori_data[n, :, p] contains any observed values.
#     - parameters: TimeGAN network parameters
#     - TODO: Move 3 DP parameters below into a single dictionary
#     - l2_norm_clip:
#     - noise_multiplier
#     - num_microbatches:
#     - TODO: pass parameters as dictionary
#     - architecture: dictionary defining whether different parts of mel should be rnn, cnn or causal cnn
#     - logdir: directory for keeping track of training losses
#     - id: string to identify different runs
#     - output_while_training:int: will output data to a file everyoutput_while_training iterations
#     - postprocesser: Postprocessor class for outputting data while training
#     - codalab: Won't run filewriter lines since they break docker containers
#
#   Returns:
#     - generated_x: generated time-series data
#   """
#   parameters = dict()
#   parameters['module'] = 'gru'
#   parameters['hidden_dim'] = 20
#   parameters['num_layer'] = 3
#   # parameters['iterations'] = 20000
#   # parameters['batch_size'] = 128
#   static_type = 'binary'  # will change loss and activation functions to suit. Value is binary since Im using as missingness mask
#
#   if not id:
#     id = 'cnntimegan_' + datetime.now().strftime('%y%m%d %H%M')
#
#   if not logdir:
#     logdir = 'logs/{}_loss'.format(id)
#
#   if output_while_training:
#     os.makedirs(os.path.join('output', id), exist_ok=True)
#     os.makedirs(os.path.join('saved', id), exist_ok=True)
#
#   # if static_data is not None:
#   #   static_values = np.unique(static_data)
#   #   if len(static_values) == 2:
#   #     static_type = 'binary'
#   #     # TODO: Could allow for categorical variables
#   #     # TODO: Could allow for seperate choices of different static vars
#   #   else:
#   #     static_type = 'continuous'
#   # else:
#   #   static_type = False
#   #
#   # Initialization on the Graph
#   tf.reset_default_graph()
#
#   # Basic Parameters
#   no, seq_len, dim = np.asarray(ori_data).shape
#
#   # Maximum sequence length and each sequence length
#   ori_time, max_seq_len = extract_time(ori_data)
#   ori_mask = ori_mask[:,1:] # Remove time dimension from mask (since it's always observed)
#   input_data = [ori_data, ori_mask, ori_time]
#   assert all(len(x) == no for x in input_data), 'all *args must have same length'
#
#   def MinMaxScaler(data):
#     """Min-Max Normalizer.
#
#     Args:
#       - data: raw data
#
#     Returns:
#       - norm_data: normalized data
#       - min_val: minimum values (for renormalization)
#       - max_val: maximum values (for renormalization)
#     """
#     min_val = np.min(np.min(data, axis=0), axis=0)
#     data = data - min_val
#
#     max_val = np.max(np.max(data, axis=0), axis=0)
#     norm_data = data / (max_val + 1e-7)
#
#     return norm_data, min_val, max_val
#
#   # Normalization
#   ori_data, min_val, max_val = MinMaxScaler(ori_data)
#
#   ## Build a RNN networks
#
#   # Network Parameters
#   hidden_dim = parameters['hidden_dim']
#   # TODO: Allow for seperate number of hidden dimensions for static variables
#   num_layers = parameters['num_layer']
#   # iterations   = parameters['iterations']
#   # batch_size   = parameters['batch_size']
#   module_name = parameters['module']
#   z_dim = dim
#   s_dim = dim - 1
#   gamma = 1
#
#   def layer_architecture(X, T, type, last_activation="relu"):
#     """
#     Returns either rnn, cnn or causal cnn architecture
#     Args:
#       X: Input layer
#       T: time layer
#       type: Architecture type: "rnn", "cnn" or "ccnn"
#       last_activation: If type is "cnn" or "ccnn", the activation function to apply to the last layer
#
#     Returns:
#
#     """
#     if type == "rnn":
#       r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
#       r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, X, dtype=tf.float32, sequence_length=T)
#     elif type == "cnn":
#       r_outputs = cnn(X, hidden_dim=hidden_dim, last_activation=last_activation, padding="same")
#     elif type == "ccnn":
#       r_outputs = cnn(X, hidden_dim=hidden_dim, last_activation=last_activation, padding="causal")
#     else:
#       raise ValueError("{} is not a valid value for type".format(type))
#     return r_outputs
#
#   # Input place holders
#   X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
#   S = tf.placeholder(tf.float32, [None, s_dim], name="myinput_s")
#   Zt = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_zt")
#   Zs = tf.placeholder(tf.float32, [None, z_dim], name="myinput_zs")
#   T = tf.placeholder(tf.int32, [None], name="myinput_t")
#
#   def embedder(S, X, T):
#     """Embedding network between original feature space to latent space.
#
#     Args:
#       - S: input static features
#       - X: input time-series features
#       - T: input time information
#
#     Returns:
#       - H: embeddings
#     """
#     with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
#       # e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
#       # e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
#       # e_outputs = cnn(X, hidden_dim=hidden_dim, last_activation="linear")
#       if static_type:
#         es_outputs = fcblock(S, hidden_dim=hidden_dim, last_activation="linear")
#         Hs = tf.keras.layers.Dense(hidden_dim, activation="sigmoid", name="output_ht")(es_outputs)
#         Hs_repeat = tf.keras.layers.RepeatVector(seq_len)(Hs)
#         X = tf.keras.layers.Concatenate(axis=2)([X, Hs_repeat])  # Concat static to temporal features
#       else:
#         Hs = None
#
#       et_outputs = layer_architecture(X, T, architecture["embedder"], last_activation="linear")
#       Ht = tf.contrib.layers.fully_connected(et_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
#       Ht = tf.identity(Ht, "output_ht")
#
#     return Hs, Ht
#
#   def recovery(Hs, Ht, T):
#     """Recovery network from latent space to original space.
#
#     Args:
#       - Hs: latent static representation
#       - Ht: latent temporal representation
#       - T: input time information
#
#     Returns:
#       - S_tilde: recovered static data
#       - X_tilde: recovered temporal data
#
#     """
#     with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
#       # r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
#       # r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
#       # r_outputs = cnn(H, hidden_dim=hidden_dim, last_activation="relu")
#       if static_type:
#         # TODO: CUrrently using sigmoid activation since S will be binary missingness
#         if static_type == 'binary':
#           rs_activation = "sigmoid"
#         else:
#           rs_activation = "linear"
#         rs_outputs = fcblock(Hs, hidden_dim=hidden_dim)
#         S_tilde = tf.keras.layers.Dense(s_dim, rs_activation, name="output_s")(rs_outputs)
#         # Paper calls for recovery to implemented seperately, but I think it makes for S representing missingness in X to feed into recovery for X
#         Ht = tf.keras.layers.Concatenate(axis=2)(
#           [Ht, tf.keras.layers.RepeatVector(seq_len)(S_tilde)])  # Concat static to temporal features
#       else:
#         S_tilde = None
#
#       rt_outputs = layer_architecture(Ht, T, architecture["recovery"], last_activation="relu")
#       X_tilde = tf.contrib.layers.fully_connected(rt_outputs, dim, activation_fn=None)
#       X_tilde = tf.identity(X_tilde, "output_x")
#
#     return S_tilde, X_tilde
#
#   def generator(Zs, Zt, T):
#     """Generator function: Generate time-series data in latent space.
#
#     Args:
#       - Zs: random static variables
#       - Zt: random temporal variables
#       - T: input time information
#
#     Returns:
#       - Es: generated static embedding
#       - Et: generated temporal embedding
#
#     """
#     with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
#       # e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
#       # e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
#       # e_outputs = cnn(Z, hidden_dim=hidden_dim, last_activation="linear", padding='causal')
#       if static_type:
#         es_outputs = fcblock(Zs, hidden_dim=hidden_dim)
#         Es = tf.keras.layers.Dense(hidden_dim, activation="sigmoid", name="output_es")(es_outputs)
#         Zt = tf.keras.layers.Concatenate(axis=2)(
#           [Zt, tf.keras.layers.RepeatVector(seq_len)(Es)])  # Concat static to temporal features
#       else:
#         Es = None
#       et_outputs = layer_architecture(Zt, T, architecture["generator"], last_activation="linear")
#       Et = tf.contrib.layers.fully_connected(et_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
#       Et = tf.identity(Et, "output_et")
#     return Es, Et
#
#   def supervisor(Hs, Ht, T):
#     """Generate next sequence using the previous sequence.
#
#     Args:
#       - H: latent representation
#       - T: input time information
#
#     Returns:
#       - S: generated sequence based on the latent representations generated by the generator
#     """
#     with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
#       # e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
#       # e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
#       # e_outputs = cnn(H, hidden_dim=hidden_dim, last_activation="linear", padding='causal')
#       if static_type:
#         es_outputs = fcblock(Hs, hidden_dim=hidden_dim, last_activation="sigmoid")
#         Ss = tf.keras.layers.Dense(hidden_dim, name="output_supervisor_s")(es_outputs)
#         Ht = tf.keras.layers.Concatenate(axis=2)(
#           [Ht, tf.keras.layers.RepeatVector(seq_len)(Ss)])  # Concat static to temporal features
#       else:
#         Ss = None
#       et_outputs = layer_architecture(Ht, T, architecture["supervisor"], last_activation="linear")
#       St = tf.contrib.layers.fully_connected(et_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
#       St = tf.identity(St, "output_supervisor_t")
#     return Ss, St
#
#   def discriminator(Hs, Ht, T):
#     """Discriminate the original and synthetic time-series data.
#
#     Args:
#       - Hs: latent static representation
#       - Ht: latent temporal representation
#       - T: input time information
#
#     Returns:
#       - Y_hat: classification results between original and synthetic time-series
#     """
#     with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
#       # d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
#       # d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
#       # d_outputs = cnn(H, hidden_dim=hidden_dim, last_activation="relu")
#       if static_type:
#         Ht = tf.keras.layers.Concatenate(axis=2)(
#           [Ht, tf.keras.layers.RepeatVector(seq_len)(Hs)])  # Concat static to temporal features
#       d_outputs = layer_architecture(Ht, T, architecture["discriminator"], last_activation="relu")
#       Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None)
#       Y_hat = tf.identity(Y_hat, "output_y")
#     return Y_hat
#
#   # Embedder & Recovery
#   Hs, Ht = embedder(S, X, T)
#   S_tilde, X_tilde = recovery(Hs, Ht, T)
#
#   # Generator
#   Es_hat, Et_hat = generator(Zs, Zt, T)
#   Hs_hat, Ht_hat = supervisor(Es_hat, Et_hat, T)
#   Hs_hat_supervise, Ht_hat_supervise = supervisor(Hs, Ht, T)
#
#   # Synthetic data
#   S_hat, X_hat = recovery(Hs_hat, Ht_hat, T)
#
#   # Discriminator
#   Y_fake = discriminator(Hs_hat, Ht_hat, T)
#   Y_real = discriminator(Hs, Ht, T)
#   Y_fake_e = discriminator(Es_hat, Et_hat, T)
#
#   # Variables
#   e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
#   r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
#   g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
#   s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
#   d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
#
#   # Discriminator loss
#   D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
#   D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
#   D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
#   D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
#
#   # Generator loss
#   # # 1. Adversarial loss
#   G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
#   G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
#
#   # # 2. Supervised loss
#   G_loss_S = (tf.losses.mean_squared_error(Ht[:, 1:, :], Ht_hat_supervise[:, :-1, :]) +
#               tf.losses.mean_squared_error(Hs, Hs_hat_supervise)) # Model doesnt train static vars without this part
#
#   # # 3. Two Momments
#   # G_loss_V1 = tf.reduce_mean(
#   #   tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
#   G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))
#   # Calculate covariance difference to try and enforce similarities between variables (instead of var matrix)
#   G_loss_V1 = tf.reduce_mean(tf.sqrt(tf.abs(tfp.stats.covariance(X_hat) - tfp.stats.covariance(X))))
#   # G_loss_V1 = tf.reduce_mean(
#   #   tf.abs(tf.sqrt(tfp.stats.covariance(X_hat) + 1e-6) - tf.sqrt(tfp.stats.covariance(X) + 1e-6)))
#   G_loss_V = G_loss_V1 + G_loss_V2
#
#   # 4. Summation
#   G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
#
#   # # Embedder network loss
#   E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
#   if static_type == 'binary':
#     E_loss_S0 = tf.losses.sigmoid_cross_entropy(S, S_tilde)
#   elif static_type == 'continuous':
#     E_loss_S0 = tf.losses.mean_squared_error(S, S_tilde)
#
#   E_loss0 = 10 * tf.sqrt(E_loss_T0 + E_loss_S0)
#   E_loss = E_loss0 + 0.1 * G_loss_S
#   # optimizer
#   opt = tf.train.AdamOptimizer()
#   E0_solver = opt.minimize(E_loss0, var_list=e_vars + r_vars)
#   E_solver = opt.minimize(E_loss, var_list=e_vars + r_vars)
#   D_solver = opt.minimize(D_loss, var_list=d_vars)
#   G_solver = opt.minimize(G_loss, var_list=g_vars + s_vars)
#   # I think I dont need to train GS_Solver with DP since, its loss between embedder output, which is already trained with DP
#   # But the rest all either see, X or Y (whether sample is real or not )
#   GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)
#
#   # Write training loss to logs
#   # Todo: remove unused summaries
#   ET0_summary = tf.compat.v1.summary.scalar('E_loss_T0', tf.math.reduce_mean(E_loss_T0))
#   GS_summary = tf.compat.v1.summary.scalar('G_loss_S', tf.math.reduce_mean(G_loss_S))
#   GU_summary = tf.compat.v1.summary.scalar('G_loss_U', tf.math.reduce_mean(G_loss_U))
#   GV_summary = tf.compat.v1.summary.scalar('G_loss_V', tf.math.reduce_mean(G_loss_V))
#   G_summary = tf.compat.v1.summary.scalar('G_loss', tf.math.reduce_mean(G_loss))
#   GV1_summary = tf.compat.v1.summary.scalar('G_covar_diff_loss', tf.math.reduce_mean(G_loss_V1))
#   GV2_summary = tf.compat.v1.summary.scalar('G_mean_diff_loss', tf.math.reduce_mean(G_loss_V2))
#   D_summary = tf.compat.v1.summary.scalar('D_loss', tf.math.reduce_mean(D_loss))
#   static_mse_summ = tf.compat.v1.summary.scalar('E_loss_TS0', tf.math.reduce_mean(E_loss_S0))
#
#   if not codalab:
#     summary_writer = tf.compat.v1.summary.FileWriter(logdir)
#
#   ## TimeGAN training
#   from tensorflow import ConfigProto
#   config = ConfigProto()
#   config.gpu_options.allow_growth = True
#   sess = tf.Session(config=config)
#   sess.run(tf.global_variables_initializer())
#
#   # 1. Embedding network training
#   print('Start Embedding Network Training')
#
#   for itt in tqdm(range(iterations)):
#     # Set mini-batch
#     X_mb, S_mb, T_mb = batch_generator(batch_size, *input_data)
#     # Train embedder
#     _, step_e_loss, step_e_summ, step_e_static_summ = sess.run([E0_solver, E_loss_T0, ET0_summary, static_mse_summ],
#                                                                feed_dict={S: S_mb, X: X_mb, T: T_mb})
#     # Checkpoint
#     if itt % 1000 == 0:
#       print(
#         'step: ' + str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.mean(np.sqrt(step_e_loss)), 4)))
#     # Write loss to logs
#     if not codalab:
#       summary_writer.add_summary(step_e_summ, itt)
#   print('Finish Embedding Network Training')
#
#   # 2. Training only with supervised loss
#   print('Start Training with Supervised Loss Only')
#
#   for itt in tqdm(range(iterations)):
#     # Set mini-batch
#     X_mb, S_mb, T_mb = batch_generator(batch_size, *input_data)
#     # Random vector generation
#     Zt_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
#     Zs_mb = list(np.random.uniform(0.0, 1, [z_dim]) for i in range(batch_size))
#     # Train generator
#     _, step_g_loss_s, step_gs_summary = sess.run([GS_solver, G_loss_S, GS_summary],
#                                                  feed_dict={Zt: Zt_mb, Zs: Zs_mb, S: S_mb, X: X_mb, T: T_mb})
#     # Checkpoint
#     if itt % 1000 == 0:
#       print(
#         'step: ' + str(itt) + '/' + str(iterations) + ', s_loss: ' + str(np.round(np.mean(np.sqrt(step_g_loss_s)), 4)))
#     # Write loss to logs
#     if not codalab:
#       summary_writer.add_summary(step_gs_summary, itt + iterations)
#   print('Finish Training with Supervised Loss Only')
#
#   # 3. Joint Training
#   print('Start Joint Training')
#
#   for itt in tqdm(range(iterations)):
#     # Generator training (twice more than discriminator training)
#     for kk in range(2):
#       # Set mini-batch
#       X_mb, S_mb, T_mb = batch_generator(batch_size, *input_data)
#       # Random vector generation
#       Zt_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
#       Zs_mb = list(np.random.uniform(0.0, 1, [z_dim]) for i in range(batch_size))
#       # Train generator
#       (_, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_g_summary, step_gu_summary, step_gs_summary,
#        step_gv_summary, step_gv1_summary, step_gv2_summary,
#        step_e_static_summ) = sess.run(
#         [G_solver, G_loss_U, G_loss_S, G_loss_V, G_summary, GU_summary, GS_summary, GV_summary, GV1_summary,
#          GV2_summary, static_mse_summ], feed_dict={Zs: Zs_mb, Zt: Zt_mb, S: S_mb, X: X_mb, T: T_mb})
#       # Train embedder
#       _, step_e_loss_t0, step_e_summ = sess.run([E_solver, E_loss_T0, ET0_summary],
#                                                 feed_dict={Zs: Zs_mb, Zt: Zt_mb, S: S_mb, X: X_mb, T: T_mb})
#       # Write loss to logs
#       if not codalab:
#         summary_writer.add_summary(step_g_summary, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_gu_summary, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_gs_summary, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_gv_summary, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_gv1_summary, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_gv2_summary, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_e_summ, 2 * (iterations + itt) + kk)
#         summary_writer.add_summary(step_e_static_summ, 2 * (iterations + itt) + kk)
#
#     # Discriminator training
#     # Set mini-batch
#
#     X_mb, S_mb, T_mb = batch_generator(batch_size, *input_data)
#
#     # Random vector generation
#     Zt_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
#     Zs_mb = list(np.random.uniform(0.0, 1, [z_dim]) for i in range(batch_size))
#     # Check discriminator loss before updating
#     check_d_loss, step_d_summary = sess.run([D_loss, D_summary],
#                                             feed_dict={S: S_mb, X: X_mb, T: T_mb, Zs: Zs_mb, Zt: Zt_mb})
#
#     # Train discriminator (only when the discriminator does not work well)
#     if (np.mean(check_d_loss) > 0.15):
#       _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={S: S_mb, X: X_mb, T: T_mb, Zs: Zs_mb, Zt: Zt_mb})
#
#     # Write loss to logs
#     if not codalab:
#       summary_writer.add_summary(step_d_summary, 2 * (iterations + itt) + 1)
#
#     # Output data
#     # TODO: Also save model weights so I can reload if needed
#     if output_while_training and itt % output_while_training == 0:
#       Zt_mb = random_generator(no, z_dim, ori_time, max_seq_len)
#       Zs_mb = [np.random.uniform(0.0, 1, [z_dim]) for i in range(no)]
#       generated_x_curr = sess.run(X_hat, feed_dict={Zs: Zs_mb, Zt: Zt_mb, S: ori_mask, X: ori_data, T: ori_time})
#       generated_x = list()
#
#       for i in range(no):
#         temp = generated_x_curr[i, :ori_time[i], :]
#         generated_x.append(temp)
#
#       # Renormalization
#       generated_x = generated_x * max_val
#       generated_x = generated_x + min_val
#
#       if postprocessor is not None:
#         output_data = postprocessor.postprocess(generated_x)
#         output_data.to_csv(os.path.join('output', id, 'idx{}.csv'.format(itt)))
#       else:
#         print('Need to pass postprocessor to cnntimegan. Not implemented otherwise')
#
#       # Save model
#       if not codalab:
#         simple_save(sess, os.path.join('saved', id, 'idx{}'.format(iterations)),
#                     inputs={"s": S, "x": X, "t": T, "zs": Zs, "zt": Zt},
#                     outputs={"s_hat": S_hat, "x_hat": X_hat})
#
#     # Print multiple checkpoints
#     if itt % 1000 == 0:
#       print('step: ' + str(itt) + '/' + str(iterations) +
#             ', d_loss: ' + str(np.round(np.mean(step_d_loss), 4)) +
#             ', g_loss_u: ' + str(np.round(np.mean(step_g_loss_u), 4)) +
#             ', g_loss_s: ' + str(np.round(np.mean(np.sqrt(step_g_loss_s)), 4)) +
#             ', g_loss_v: ' + str(np.round(np.mean(step_g_loss_v), 4)) +
#             ', e_loss_t0: ' + str(np.round(np.mean(np.sqrt(step_e_loss_t0)), 4)))
#   print('Finish Joint Training')
#
#   ## Synthetic data generation
#   Zt_mb = random_generator(no, z_dim, ori_time, max_seq_len)
#   Zs_mb = [np.random.uniform(0.0, 1, [z_dim]) for i in range(no)]
#   generated_x_curr, generated_s = sess.run([X_hat, S_hat],
#                                            feed_dict={Zs: Zs_mb, Zt: Zt_mb, S: ori_mask, X: ori_data, T: ori_time})
#   generated_x = list()
#   for i in range(no):
#     temp = generated_x_curr[i, :ori_time[i], :]
#     generated_x.append(temp)
#
#   # Renormalization
#   generated_x = generated_x * max_val
#   generated_x = generated_x + min_val
#
#   # Save model
#   if not codalab:
#     simple_save(sess, os.path.join('saved', id, 'idx{}'.format(iterations)),
#                 inputs={"s": S, "x": X, "t": T, "zs": Zs, "zt": Zt},
#                 outputs={"s_hat": S_hat, "x_hat": X_hat})
#
#   # Using mask probability to randomly apply missingness to generated data
#   random_s = np.random.uniform(0., 1., generated_s.shape)
#   mask_s = np.where(generated_s > random_s, 1., np.nan)
#   # mask_s[:, 0] = 1. # Time variable is never missing. (Really need to figure out why model ever predicts it as missing)
#   generated_x[:,:,1:] = np.einsum('ntp,np->ntp', generated_x[:,:,1:], mask_s)
#   return generated_x, generated_s