import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt
import pickle

# Much of the code has been used from Parag Mittal
# %%
def autoencoder(input_shape=[None, 15, 50, 1],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
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
    # n_input --> number of layers
    # n_output --> number of layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                50, #filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
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
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - y_out))

    # %%
    return {'x': x, 'z': z, 'y': y_out, 'cost': cost}


# %%
def test_convVAE():
    InpMatrix, OutMatrix = pickle.load(open('../../../darkweb_data/5_10/Input_Output_Matrices.pickle', 'rb'))
    InpMatrix = np.reshape(InpMatrix, (InpMatrix.shape[0], InpMatrix.shape[1], InpMatrix.shape[2], 1))
    OutMatrix = np.reshape(OutMatrix, (OutMatrix.shape[0], OutMatrix.shape[1], OutMatrix.shape[2], 1))

    ae = autoencoder()

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    # Create the batches of the data
    num_train_data = InpMatrix.shape[0]
    # num_train_data = 1000

    # Fit all training data
    n_epochs = 100
    for epoch_i in range(n_epochs):
        for idx in range(num_train_data // batch_size):
            batch_xs_in = InpMatrix[idx * batch_size: (idx + 1) * batch_size]
            batch_xs_out = OutMatrix[idx * batch_size: (idx + 1) * batch_size]
            sess.run(optimizer, feed_dict={ae['x']: batch_xs_in, ae['y']: batch_xs_out})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs_in, ae['y']: batch_xs_out}))


if __name__ == '__main__':
    test_convVAE()