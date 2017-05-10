import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
import pickle

np.random.seed(0)

# ---------------------- Parameters section -------------------

# Model Hyperparameters
filter_sizes = (2, 3)
num_filters = 20
dropout_prob = (0.1, 0.1)
hidden_dims = 50

# Training parameters
batch_size = 32
num_epochs = 100

# ---------------------- Parameters end -----------------------

""" Load the training and the validation data"""
idx_part = 0
idx_fold = 0
col_p1 = 3
# output_dir = ''
output_dir = 'data/fold_' + str(idx_fold) + '/col/' + str(col_p1-3) + '/'
x_train = pickle.load(open(output_dir + 'X_train.pickle', 'rb'))
y_train = pickle.load(open(output_dir + 'Y_train.pickle', 'rb'))

x_train = x_train[:int(0.8*x_train.shape[0]),:,:]
x_val = x_train[int(0.8*x_train.shape[0]):,:,:]
y_train = y_train[:int(0.8*y_train.shape[0])]
y_val = y_train[int(0.8*y_train.shape[0]):]
# x_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
# y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))

#
x_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))

# print(y_test[y_test == 1].shape[0]/ y_test.shape[0])
# print(y_val[y_val == 1].shape[0]/ y_val.shape[0])
# exit()

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print(x_train[2])
exit()
sequence_length = x_train.shape[1]
embedding_dim = x_train.shape[2]

# Build model
input_shape = (sequence_length, embedding_dim)
model_input = Input(shape=input_shape)

# Static model do not have embedding layer
z = Dropout(dropout_prob[0])(model_input)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
sgd = SGD(lr=0.01, nesterov=False)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["acc"])


# print(x_val.shape, y_val.shape)
# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)