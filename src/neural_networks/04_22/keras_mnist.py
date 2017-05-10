'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pickle

"""Load the training and the validation data"""
idx_part = 0
idx_fold = 1
col_p1 = 10
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


batch_size = 32
num_classes = 2
epochs = 10

# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]

# print(img_rows, img_cols)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# input_shape = (img_rows, img_cols, 1)

# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(x_train[0])
exit()
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])