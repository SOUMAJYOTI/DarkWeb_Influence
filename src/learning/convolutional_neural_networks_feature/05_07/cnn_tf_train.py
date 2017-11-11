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

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "data/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "data/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 5, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.2, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.3, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
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
conv_out = conv_layer[0]


for idx_fold in range(5):
    for col in range(2, 3):
        with tf.Graph().as_default():
            output_dir = '../../../darkweb_data/5_15/data/indices/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            X_t = pickle.load(open(output_dir + 'X_train_l.pickle', 'rb'))
            Y_t = pickle.load(open(output_dir + 'Y_train_l.pickle', 'rb'))

            X_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
            Y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))
            # x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

            # # Create the final train and validation sets
            # rand_idx = [randint(0, X_t.shape[0] - 1) for p in range(int(0.2 * X_t.shape[0]))]
            # train_ind = list(set(list(range(X_t.shape[0]))) - set(rand_idx))
            # x_train = X_t[train_ind, :]
            # x_dev = X_t[rand_idx, :]
            # y_train = Y_t[train_ind]
            # y_dev = Y_t[rand_idx]

            # # Build vocabulary
            # max_document_length = max([len(x.split(" ")) for x in x_text])
            # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            # x = np.array(list(vocab_processor.fit_transform(x_text)))
            #
            # # Randomly shuffle data
            # np.random.seed(10)
            # shuffle_indices = np.random.permutation(np.arange(len(y)))
            # x_shuffled = x[shuffle_indices]
            # y_shuffled = y[shuffle_indices]
            #
            # # Split train/test set
            # # TODO: This is very crude, should use cross-validation
            # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
            # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
            # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
            print("Vocabulary Size: {:d}".format(len(vocab_dict)))
            print("Train/Dev split: {:d}/{:d}".format(len(Y_t), len(X_t)))

            # Training
            # ==================================================
            # with sess.as_default():
            # tf.variable_scope('Train', reuse=FLAG_SCOPE)
            # Initialize all variables

            # Restore the pretrained layers
            sess = tf.Session(config=config)
            # saver = tf.train.Saver()
            # saver = tf.train.import_meta_graph('tmp/model.meta')
            # saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
            # conv_out = sess.run(ae['conv'], feed_dict={ae['x']: InpMatrix[:32], ae['y']: OutMatrix[:32]})
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
                # print(y_batch)
                # exit()
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  # cnn.custom_units: conv_out[:x_batch.shape[0]],
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, pred = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                # print("step {:g}, loss {:g}, acc {:g}, pred {} ".format(step, loss, accuracy, pred))
                # print("loss {:g}, acc {:g}, pred: {} ".format(loss, accuracy, pred))

                # train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  # cnn.custom_units: conv_out[:x_batch.shape[0]],
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, pred = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)

                pred = np.array(pred)
                # print(pred)
                Y_act = []
                for idx_y in range(y_batch.shape[0]):
                    if np.array_equal(y_batch[idx_y], [1, 0]):
                        Y_act.append(0)
                    else:
                        Y_act.append(1)

                Y_act = np.array(Y_act)
                # print(Y_act.shape)
                # print(Y_act[Y_act == 1].shape)
                # for idx_y in range(pred.shape)
                time_str = datetime.datetime.now().isoformat()
                print("loss {:g}, acc {:g}".format(loss, accuracy))
                print("F1: ", sklearn.metrics.f1_score(Y_act, pred))


            # print(x_train[64])
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(X_t, Y_t)), FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)
            # Training loop. For each batch...
            cnt_batch = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                # print(x_batch[0])
                # print(x_batch[0][0], x_batch[1][0])
                # print(y_batch[0])
                # print(cnt_batch, x_batch[0][0], x_batch[1][0])
                train_step(x_batch, y_batch)
                cnt_batch += 1
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(X_test, Y_test)
                    # dev_step(x_dev, y_dev, writer=dev_summary_writer)

            sess.close()
