import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt
import pickle


# Use some sort of unsupervised deconvolution feature for the final neural network
# Instead of the convolution feature, use the second decoding model for the convolved layer

# Much of the code has been used from Parag Mittal
# %%
def autoencoder(input_shape=[None, 30, 50, 1],
                n_filters=[1, 20],
                filter_sizes=[3, 3],
                corruption=False):
    """Build a deep autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    y_out = tf.placeholder(
        tf.float32, input_shape, name='x')

    # %%

    x_tensor = x
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    # if corruption:
    #     current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    conv_layer = []
    # n_input --> number of layers
    # n_output --> number of layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3] # number of channels of the input ---> 3 for image, 1 for text
        shapes.append(current_input.get_shape().as_list())
        print(shapes)
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                50,
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        # print(W.get_shape())

        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        c_l = tf.nn.conv2d(
                current_input, W, strides=[1, 1, 1, 1], padding='VALID', name='encoder')
        conv_layer.append(c_l)
        output = lrelu(tf.add(conv_layer[layer_i], b))
        # print(c_l.get_shape())

        n_input = c_l.get_shape().as_list()[3]
        shapes.append(output.get_shape().as_list())
        W_2 = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                1,
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W_2)
        c_l = tf.nn.conv2d(
            output, W_2, strides=[1, 1, 1, 1], padding='VALID', name='encoder')
        conv_layer.append(c_l)
        # print(W_2.get_shape())
        output = lrelu(tf.add(conv_layer[layer_i+1], b))

        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 1, 1, 1], padding='VALID', name='decoder'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - y_out))

    # %%
    return {'x': x, 'z': z, 'y': y_out, 'conv': conv_layer, 'encoder': encoder, 'cost': cost}


# %%
def test_convVAE():
    InpMatrix, OutMatrix = pickle.load(open('../../../darkweb_data/5_15/unlabeled/Input_Output_Matrices.pickle', 'rb'))
    InpMatrix = np.reshape(InpMatrix, (InpMatrix.shape[0], InpMatrix.shape[1], InpMatrix.shape[2], 1))
    OutMatrix = np.reshape(OutMatrix, (OutMatrix.shape[0], OutMatrix.shape[1], OutMatrix.shape[2], 1))

    print(InpMatrix.shape)
    ae = autoencoder()

    # # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    #
    sess.run(tf.global_variables_initializer())

    batch_size = 32
    # Create the batches of the data
    num_train_data = InpMatrix.shape[0]
    # num_train_data = 1000

    # Fit all training data
    n_epochs = 30
    for epoch_i in range(n_epochs):
        # Shuffle the training data over the epochs
        shuffle_indices = np.random.permutation(np.arange(num_train_data))
        InpMatrix = InpMatrix[shuffle_indices]
        for idx in range(num_train_data // batch_size):
            batch_xs_in = InpMatrix[idx * batch_size: (idx + 1) * batch_size]
            batch_xs_out = OutMatrix[idx * batch_size: (idx + 1) * batch_size]
            _, conv_layer, encoder = sess.run([optimizer, ae['conv'], ae['encoder']], feed_dict={ae['x']: batch_xs_in, ae['y']: batch_xs_out})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs_in, ae['y']: batch_xs_out}))
    # saver.save(sess, "tmp/model")

    print(np.array(encoder[0]).shape)
    print(np.array(encoder[1]).shape)
    pickle.dump(encoder, open('../../../darkweb_data/5_15/filter_weights_20_stacked.pickle', 'wb'))
    # pickle.dump(encoder, open('../../../darkweb_data/5_15/conv_layer.pickle', 'wb'))

    # Restore the pretrained layers
    # saver = tf.train.import_meta_graph('tmp/model.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
    # conv_out = sess.run(ae['conv'], feed_dict={ae['x']: InpMatrix, ae['y']: OutMatrix})
    # print(tf.shape(conv_out))


if __name__ == '__main__':
    test_convVAE()