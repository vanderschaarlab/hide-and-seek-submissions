import tensorflow as tf
import argparse, json
import numpy as np
from .modules.embedder import Embedder
from .modules.recovery import Recovery
from .modules.generator import Generator
from .modules.supervisor import Supervisor
from .modules.discriminator import Discriminator
from .modules.model_utils import extract_time, random_generator, batch_generator, MinMaxScaler, save_dict_to_json
import logging, os , datetime, time
#logging.disable(logging.WARNING) 


class TimeGAN(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.embedder = Embedder(args)
        self.recovery = Recovery(args)
        self.generator = Generator(args)
        self.supervisor = Supervisor(args)
        self.discriminator = Discriminator(args)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.use_dpsgd = args.use_dpsgd

    # E0_solver
    def recovery_forward(self, X, optimizer):
        # initial_hidden = self.embedder.initialize_hidden_state()
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = self.mse(X, X_tilde)
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)
        
        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss0, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return E_loss_T0

    # GS_solver
    def supervisor_forward(self, X, Z, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            #H_hat = self.generator(Z, training=True)
            H_hat_supervise = self.supervisor(H, training=True)
            G_loss_S = self.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        
        var_list = self.supervisor.trainable_weights
        grads = tape.gradient(G_loss_S, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return G_loss_S

    # D_solver + G_solver
    def adversarial_forward(self, X, Z, optimizer=None, gamma=0.01, train_G=False, train_D=False):
        with tf.GradientTape() as tape:
            H = self.embedder(X)
            E_hat = self.generator(Z, training=True)

            # Supervisor & Recovery
            H_hat_supervise = self.supervisor(H, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            X_hat = self.recovery(H_hat)

            # Discriminator
            Y_fake = self.discriminator(H_hat, training=True)
            Y_real = self.discriminator(H, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)

            if train_G:
                # Generator loss
                G_loss_U = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake), Y_fake, from_logits=True)
                )
                G_loss_U_e = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True)
                )
                G_loss_S = tf.math.reduce_mean(
                    tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                )
                # Difference in "variance" between X_hat and X
                G_loss_V1 = tf.math.reduce_mean(tf.math.abs(tf.math.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6)
                                                - tf.math.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
                # Difference in "mean" between X_hat and X
                G_loss_V2 = tf.math.reduce_mean(tf.math.abs((tf.nn.moments(X_hat, [0])[0])
                                                - (tf.nn.moments(X, [0])[0])))
                G_loss_V = G_loss_V1 + G_loss_V2
                #G_loss_V = tf.math.add(G_loss_V1, G_loss_V2)
                ## Sum of all G_losses
                G_loss = 100 * G_loss_U + gamma * G_loss_U_e + 100 * tf.math.sqrt(G_loss_S) + 100 * G_loss_V

            elif not train_G:
                # Discriminator loss
                D_loss_real = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(Y_real), Y_real, from_logits=True)
                )
                D_loss_fake = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.zeros_like(Y_fake), Y_fake, from_logits=True)
                )
                D_loss_fake_e = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=True)
                )
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        if train_G:
            GS_var_list = self.generator.trainable_weights + self.supervisor.trainable_weights
            GS_grads = tape.gradient(G_loss, GS_var_list)
            optimizer.apply_gradients(zip(GS_grads, GS_var_list))
            
            return G_loss_U, G_loss_S, G_loss_V

        elif train_D:
            D_var_list = self.discriminator.trainable_weights
            D_grads = tape.gradient(D_loss, D_var_list)
            optimizer.apply_gradients(zip(D_grads, D_var_list))

            return D_loss

        elif not train_D:
            # Checking if D_loss > 0.15
            return D_loss

    # E_solver
    def embedding_forward_joint(self, X, optimizer, eta):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = tf.math.reduce_mean(
                tf.keras.losses.mean_squared_error(X, X_tilde)
            )
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)

            H_hat_supervise = self.supervisor(H)
            G_loss_S = tf.math.reduce_mean(
                self.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            )
            E_loss = E_loss0 + eta * G_loss_S
        
        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return E_loss

    # Inference
    def generate(self, Z, ori_data_num, ori_time, max_val, min_val):
        """
        Args:
            Z: input random noises
            ori_data_num: the first dimension of ori_data.shape
            ori_time: timesteps of original data
            max_val: the maximum value of MinMaxScaler(ori_data)
            min_val: the minimum value of MinMaxScaler(ori_data)
        Return:
            generated_data: synthetic time-series data
        """
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        generated_data_curr = self.recovery(H_hat)
        generated_data_curr = generated_data_curr.numpy()
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, :ori_time[i], :]
            generated_data.append(temp)
        
        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val
        
        return generated_data

    # Inference by autoencoder
    def ae_generate(self, ori_data, ori_data_num, ori_time, max_val, min_val):
        H = self.embedder(ori_data)
        generated_data_curr = self.recovery(H)
        generated_data_curr = generated_data_curr.numpy()
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, :ori_time[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

        return generated_data


class MyScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
    
    def __call__(self, step):
        if step % 100 == 0:
            self.initial_learning_rate = self.initial_learning_rate * 0.9
            print("Learning rate is adjusted to {}".format(
                self.initial_learning_rate
            ))
            return self.initial_learning_rate
        else:
            return self.initial_learning_rate


class VaswaniScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        super().__init__()
        self.d_model = d_model*2
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# NOTE: ES Updated: ====================================================================================================
class Args(object):
    def __init__(self, exp_name, data_name, max_seq_len, train_rate, feature_prediction_no, seed, hider_model, noise_size, module_name, epsilon, optimizer, batch_size, z_dim, hidden_dim, num_layers, embedding_iterations, supervised_iterations, joint_iterations, feature_dim, eta, learning_rate, gen_type, use_dpsgd, l2_norm_clip, noise_multiplier, dp_lr):
        self.exp_name = exp_name
        self.data_name = data_name
        self.max_seq_len = max_seq_len
        self.train_rate = train_rate
        self.feature_prediction_no = feature_prediction_no
        self.seed = seed
        self.hider_model = hider_model
        self.noise_size = noise_size
        self.module_name = module_name
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_iterations = embedding_iterations
        self.supervised_iterations = supervised_iterations
        self.joint_iterations = joint_iterations
        self.feature_dim = feature_dim
        self.eta = eta
        self.learning_rate = learning_rate
        self.gen_type = gen_type
        self.use_dpsgd = use_dpsgd
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.dp_lr = dp_lr
# ======================================================================================================================


def train_timegan(ori_data, mode='train'):
    # NOTE: ES Updated: ================================================================================================
    params_dict = {
        "exp_name": "timegan",
        "data_name": "amsterdam",
        "max_seq_len": 100,
        "train_rate": 0.5,
        "feature_prediction_no": 5,
        "seed": 0,
        "hider_model": "timegan",
        "noise_size": 0.1,
        "module_name": "gru",
        "epsilon": 1e-08,
        "optimizer": "adam",
        "batch_size": 256,
        "z_dim": -1,
        "hidden_dim": 10,
        "num_layers": 3,
        "embedding_iterations": 2000,
        "supervised_iterations": 0,
        "joint_iterations": 0,
        "feature_dim": 71,
        "eta": 0.1,
        "learning_rate": 0.001,
        "gen_type": "autoencoder",
        "use_dpsgd": False,
        "l2_norm_clip": 1.0,
        "noise_multiplier": 1.1,
        "dp_lr": 0.15
    }
    args = Args(**params_dict)
    # ==================================================================================================================

    no, seq_len, dim = np.asarray(ori_data).shape

    if args.z_dim == -1:  # choose z_dim for the dimension of noises
        args.z_dim = dim

    ori_time, max_seq_len = extract_time(ori_data)
    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    if mode == 'train':
        model = TimeGAN(args)

        # Set up optimizers
        if args.optimizer == 'adam':
            #learning_rate = MyScheduler(args.learning_rate)
            learning_rate = VaswaniScheduler(d_model=dim)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=args.epsilon)
        if args.use_dpsgd:
            from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
            from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
            from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

            D_optimizer = DPGradientDescentGaussianOptimizer(
                l2_norm_clip=1.0,
                noise_multiplier=0.1,
                num_microbatches=args.batch_size,
                learning_rate=0.15
            )
        else:
            D_optimizer = optimizer

        print('Set up Tensorboard')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('/opt/hide-and-seek/hiders/yingjialin/' + 'tensorboard', current_time + '-' + args.exp_name)  # NOTE: ES Updated.
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # save args to tensorboard folder
        save_dict_to_json(args.__dict__, os.path.join(train_log_dir, 'params.json'))

        # 1. Embedding network training
        print('Start Embedding Network Training')
        start = time.time()
        for itt in range(args.embedding_iterations):
        #for itt in range(1):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size, use_tf_data=False)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            step_e_loss = model.recovery_forward(X_mb, optimizer)
            if itt % 100 == 0:
                print('step: '+ str(itt) + '/' + str(args.embedding_iterations) +
                      ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)))
                # Write to Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('Embedding_loss', np.round(np.sqrt(step_e_loss),4), step=itt)

        print('Finish Embedding Network Training')
        end = time.time()
        print('Train embedding time elapsed: {} sec'.format(end-start))

        # 2. Training only with supervised loss
        if args.gen_type == 'autoencoder':
            print('Skip Training with Supervised Loss')
        else:
            print('Start Training with Supervised Loss Only')
            start = time.time()
            for itt in range(args.supervised_iterations):
            #for itt in range(1):
                X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
                Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)

                X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
                Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

                step_g_loss_s = model.supervisor_forward(X_mb, Z_mb, optimizer)
                if itt % 100 == 0:
                    print('step: '+ str(itt)  + '/' + str(args.supervised_iterations) +', s_loss: '
                                + str(np.round(np.sqrt(step_g_loss_s),4)))
                    # Write to Tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Supervised_loss', np.round(np.sqrt(step_g_loss_s),4), step=itt)
        
            print('Finish Training with Supervised Loss Only')
            end = time.time()
            print('Train Supervisor time elapsed: {} sec'.format(end-start))

        # 3. Joint Training
        if args.gen_type == 'autoencoder':
            print('Skip Joint Training')
        else:
            print('Start Joint Training')
            start = time.time()
            for itt in range(args.joint_iterations):
            #for itt in range(1):
                # Generator training (two times as discriminator training)
                for g_more in range(2):
                    X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
                    Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)
                    
                    X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
                    Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

                    step_g_loss_u, step_g_loss_s, step_g_loss_v = model.adversarial_forward(X_mb, Z_mb,
                                                                                    optimizer,
                                                                                    train_G=True,
                                                                                    train_D=False)
                    step_e_loss_t0 = model.embedding_forward_joint(X_mb, optimizer, args.eta)

                # Discriminator training
                X_mb, T_mb = batch_generator(ori_data, ori_time, args.batch_size)
                Z_mb = random_generator(args.batch_size, args.z_dim, T_mb, args.max_seq_len)

                X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
                Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

                check_d_loss = model.adversarial_forward(X_mb, Z_mb, train_G=False, train_D=False)
                if (check_d_loss > 0.15): 
                    step_d_loss = model.adversarial_forward(X_mb, Z_mb, D_optimizer, train_G=False, train_D=True)
                else:
                    step_d_loss = check_d_loss

                if itt % 100 == 0:
                    print('step: '+ str(itt) + '/' + str(args.joint_iterations) + 
                        ', d_loss: ' + str(np.round(step_d_loss, 4)) + 
                        ', g_loss_u: ' + str(np.round(step_g_loss_u, 4)) + 
                        ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) + 
                        ', g_loss_v: ' + str(np.round(step_g_loss_v, 4)) + 
                        ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0), 4)))
                    # Write to Tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Joint/Discriminator',
                                        np.round(step_d_loss, 4), step=itt)
                        tf.summary.scalar('Joint/Generator',
                                        np.round(step_g_loss_u, 4), step=itt)
                        tf.summary.scalar('Joint/Supervisor',
                                        np.round(step_g_loss_s, 4), step=itt)
                        tf.summary.scalar('Joint/Moments',
                                        np.round(step_g_loss_v, 4), step=itt)
                        tf.summary.scalar('Joint/Embedding',
                                        np.round(step_e_loss_t0, 4), step=itt)        
            print('Finish Joint Training')
            end = time.time()
            print('Train jointly time elapsed: {} sec'.format(end-start))

    
        ## Synthetic data generation
        if args.gen_type == 'autoencoder':
            input_ori = tf.convert_to_tensor(ori_data, dtype=tf.float32)
            generated_data = model.ae_generate(input_ori, no, ori_time, max_val, min_val)
        else:    
            Z_mb = random_generator(no, args.z_dim, ori_time, args.max_seq_len)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
            generated_data = model.generate(Z_mb, no, ori_time, max_val, min_val)
        
        return generated_data
