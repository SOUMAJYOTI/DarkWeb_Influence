#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import pickle
from random import randint
from cae_test import autoencoder
import sklearn.metrics


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 10, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.3, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Data Preparation
# ==================================================
# Load data
InpMatrix, OutMatrix = pickle.load(open('../../../darkweb_data/5_15/unlabeled/Input_Output_Matrices.pickle', 'rb'))
InpMatrix = np.reshape(InpMatrix, (InpMatrix.shape[0], InpMatrix.shape[1], InpMatrix.shape[2], 1))
OutMatrix = np.reshape(OutMatrix, (OutMatrix.shape[0], OutMatrix.shape[1], OutMatrix.shape[2], 1))
vocab_dict = pickle.load(open('../../../darkweb_data/5_15/data/vocab_dict_revised.pickle', 'rb'))
# FLAG_SCOPE = True
conv_layer = pickle.load(open('../../../darkweb_data/5_15/conv_layer.pickle', 'rb'))
conv_out = conv_layer[0] # for now consider the first layer

filter_weights= pickle.load(open('../../../darkweb_data/5_15/filter_weights_10.pickle', 'rb'))

num_folds = 5
num_cols = 10
offset_col = 2
f1_col = [0. for _ in range(num_cols)]

for idx_fold in range(num_folds):
    for col in range(offset_col, num_cols+offset_col):
        print('Fold: {}, Col: {}'.format(idx_fold, col))
        with tf.Graph().as_default():
            output_dir = '../../../darkweb_data/5_15/data/w2v/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            X_t = pickle.load(open(output_dir + 'X_train_l.pickle', 'rb'))
            Y_t = pickle.load(open(output_dir + 'Y_train_l.pickle', 'rb'))

            X_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
            Y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))

            print("Vocabulary Size: {:d}".format(len(vocab_dict)))
            print("Train/Dev split: {:d}/{:d}".format(len(Y_t), len(Y_test)))

            # Training
            # ==================================================
            # with sess.as_default():
            # tf.variable_scope('Train', reuse=FLAG_SCOPE)
            # Initialize all variables

            # Restore the pretrained layers
            sess = tf.Session(config=config)
            tf.cast(conv_out, tf.float32)
            # conv_out = tf.slice(conv_out, [0,0,0,0], [32,8,1,10]).eval(session=sess)
            # conv_out = np.ones((32, 8, 1, 10))
            # print(tf.shape(conv_out))

            cnn = TextCNN(
                sequence_length=X_t.shape[1],
                num_classes=Y_t.shape[1],
                vocab_size=len(vocab_dict),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """

                Y_act = []
                for idx_y in range(y_batch.shape[0]):
                    if np.array_equal(y_batch[idx_y], [1, 0]):
                        Y_act.append(0)
                    else:
                        Y_act.append(1)

                Y_act = np.array(Y_act)
                # print(Y_act)
                feed_dict = {
                  cnn.embedded_chars: x_batch,
                  cnn.input_y: y_batch,
                  cnn.custom_units: conv_out,
                  cnn.custom_W: filter_weights,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, pred = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                # if step % 100 == 0:
                #     print("step {:g}, loss {:g}, acc {:g}, pred {} ".format(step, loss, accuracy, pred))
                # print("loss {:g}, acc {:g}, pred: {} ".format(loss, accuracy, pred))

                # train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.embedded_chars: x_batch,
                  cnn.input_y: y_batch,
                  cnn.custom_units: conv_out,
                  cnn.custom_W: filter_weights,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, pred = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)

                # print(y_batch)
                pred = np.array(pred)
                Y_act = []
                for idx_y in range(y_batch.shape[0]):
                    if np.array_equal(y_batch[idx_y], [1, 0]):
                        Y_act.append(0)
                    else:
                        Y_act.append(1)

                Y_act = np.array(Y_act)

                # print('Y_act: ', Y_act)
                # print('Pred: ', pred)
                time_str = datetime.datetime.now().isoformat()
                # print("loss {:g}, acc {:g}".format(loss, accuracy))
                f1_col[col-offset_col] += sklearn.metrics.f1_score(Y_act, pred)
                print("F1: ", sklearn.metrics.f1_score(Y_act, pred))


            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(X_t, Y_t)), FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)
            # Training loop. For each batch...
            cnt_batch = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)

                train_step(x_batch, y_batch)
                cnt_batch += 1
                current_step = tf.train.global_step(sess, global_step)
                # if current_step % FLAGS.evaluate_every == 0:
                #     print("\nEvaluation:")
                #     dev_step(X_test, Y_test)
                    # dev_step(x_dev, y_dev, writer=dev_summary_writer)
            dev_step(X_test, Y_test)

            sess.close()

f1_col = np.array(f1_col)
for col in range(f1_col.shape[0]):
    f1_col[col] /= num_folds

print(f1_col)