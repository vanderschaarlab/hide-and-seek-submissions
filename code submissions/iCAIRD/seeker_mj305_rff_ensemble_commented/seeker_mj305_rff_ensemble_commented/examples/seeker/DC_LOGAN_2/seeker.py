import os
import scipy.misc
import numpy as np

from .model_notAdjusted import DCGAN
from .utils_seeker import pp, assess, visualize, to_json, show_all_variables

import tensorflow as tf

def dclogan_seeker(generated_data, enlarged_data):
    if not os.path.isdir('./data/hide_and_seek_'):
        os.makedirs('./data/hide_and_seek_')

    full_stack_data = generated_data#.transpose((0, 2, 1))
    train_size = np.shape(full_stack_data)[0]
    for i in range(train_size):
        data = full_stack_data[i, :, :].reshape((1, 10, 71, 1))
        record = np.concatenate((data, data[:, :, -1:, :]), axis=2)
        np.save('./data/hide_and_seek_/record_{:04d}.npy'.format(i), record)
        
    
#    flags = tf.app.flags
#    flags.DEFINE_integer("epoch", 3, "Epoch to train [25]")
#    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
#    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#    flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
#    flags.DEFINE_integer("batch_size", 300, "The size of batch images [64]")
#    flags.DEFINE_integer("input_height", 10, "The size of image to use (will be center cropped). [108]")
#    flags.DEFINE_integer("input_width", 72, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
#    flags.DEFINE_integer("output_height", 10, "The size of the output images to produce [64]")
#    flags.DEFINE_integer("output_width", 72, "The size of the output images to produce. If None, same value as output_height [None]")
#    flags.DEFINE_string("dataset", "hide_and_seek_", "The name of dataset [celebA, mnist, lsun]")
#    flags.DEFINE_string("input_fname_pattern", "*.npy", "Glob pattern of filename of input images [*]")
#    flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
#    flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
#    flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
#    flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
#    flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
#    FLAGS = flags.FLAGS
#    
#    pp.pprint(flags.FLAGS.__flags)

#    if FLAGS.input_width is None:
#      FLAGS.input_width = FLAGS.input_height
#    if FLAGS.output_width is None:
#      FLAGS.output_width = FLAGS.output_height

#    if not os.path.exists('./checkpoint'):
#      os.makedirs('./checkpoint')
#    if not os.path.exists(FLAGS.sample_dir):
#      os.makedirs(FLAGS.sample_dir)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
      dcgan = DCGAN(
          sess,
          input_width=72,
          input_height=10,
          output_width=72,
          output_height=10,
          batch_size=300,
          sample_num=300,
          dataset_name="hide_and_seek_",
          input_fname_pattern="*.npy",
          crop=False,
          checkpoint_dir="checkpoint",
          sample_dir="sample")

      show_all_variables()
#      config1 = tf.ConfigProto()
#      config1.gpu_options.allow_growth=True
#      config1.batch_size=300;
#      config1.train_size=300
#      config1.dataset_name="hide_and_seek_";
#      config1.learning_rate = 0.0002
#      config1.beta1 = 0.5
#      if FLAGS.train:
      dcgan.train(batch_size=300, train_size=train_size, dataset="hide_and_seek_", learning_rate = 0.0002, beta1 = 0.5, epoch = 4, checkpoint_dir = "checkpoint")
#      else:
#        if not dcgan.load(FLAGS.checkpoint_dir)[0]:
#          raise Exception("[!] Train a model first, then run test mode")
      
      # Below is codes for visualization
      OPTION = 1
      binary_label = assess(sess, dcgan, 300, OPTION, enlarged_data, train_size)

    return binary_label

#enlarged_data = np.load('enl_data.npy')
#generated_data = np.load('gen_data.npy')

#full_label_binary_test = dclogan_seeker(generated_data, enlarged_data)