from __future__ import print_function
import keras
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Reshape
from keras import backend as K
import pickle
from random import randint
import random
import numpy as np
from keras import backend as K
import sklearn.metrics


# Get the interim embeddings from the convolutional layer of the network
# Use the SGD optimizer

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch, 0])
    return activations


def main():
    batch_size = 16
    num_classes = 2
    epochs = 20

    """Load the training and the validation data"""
    # idx_part = 0
    # idx_fold = 0
    # col_p1 = 4

    f1_values = [0. for _ in range(9)]
    f1_random = [0. for _ in range(9)]

    for idx_fold in range(2):
        print("Fold: ", idx_fold)
        for col_p1 in range(3, 12):
            print("Col: ", col_p1)
            # output_dir = ''
            output_dir = 'data/05_02/fold_' + str(idx_fold) + '/col/' + str(col_p1 - 3) + '/'
            # Initial training set
            x_t = pickle.load(open(output_dir + 'X_train_l.pickle', 'rb'))
            y_t = pickle.load(open(output_dir + 'Y_train_l.pickle', 'rb'))

            y_temp = []
            for idx_y in range(y_t.shape[0]):
                if y_t[idx_y] == -1:
                    y_temp.append([1, 0])
                else:
                    y_temp.append([0, 1])

            y_t = np.array(y_temp)

            # Initial test set
            x_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
            y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))
            # Balance the test set so that no test set ends up all the minority
            X_test_pos = x_test[y_test == 1]
            X_test_neg = x_test[y_test == -1]
            # if X_test_pos.shape[0] < X_test_neg.shape[0]:
            #     X_test_neg = X_test_neg[:X_test_pos.shape[0]]
            # else:
            #     X_test_pos = X_test_pos[:X_test_neg.shape[0]]
            #
            x_test = np.concatenate((X_test_neg, X_test_pos), axis=0)
            y_test = np.array([0] * X_test_neg.shape[0] + [1] * X_test_pos.shape[0])

            print(y_test[y_test == 1].shape[0] / y_test.shape[0])

            # Create the final train and validation sets
            rand_idx = [randint(0, x_t.shape[0] - 1) for p in range(int(0.2 * x_t.shape[0]))]
            train_ind = list(set(list(range(x_t.shape[0]))) - set(rand_idx))
            x_train = x_t[train_ind, :, :]
            x_val = x_t[rand_idx, :, :]
            y_train = y_t[train_ind]
            y_val = y_t[rand_idx]

            # x_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
            # y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))

            img_rows, img_cols = x_train.shape[1], x_train.shape[2]

            # # the data, shuffled and split between train and test sets
            # (x_train, y_train), (x_test, y_test) = mnist.load_data()

            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
                x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
                x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1)

            # print(y_train[y_train == 1].shape[0] / y_train.shape[0])
            x_train = x_train.astype('float32')
            x_val = x_val.astype('float32')
            x_test = x_test.astype('float32')

            print('x_train shape:', x_train.shape)
            print('x_val shape:', x_val.shape)

            model = Sequential()
            model.add(Embedding(batch_size, 50, input_length=30))
            model.add(Reshape(input_shape, input_shape=(input_shape[0]*input_shape[1], )))

            model.add(Conv2D(10, kernel_size=(3, img_cols),
                             activation='relu',
                             input_shape=input_shape, strides=1))

            # model.add(Conv2D(20, (3, 1), activation='relu', strides=1))
            # model.add(Conv2D(40, (3, 1), activation='relu', strides=1))
            model.add(MaxPooling2D(pool_size=(2, 1)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            # set embedding weights - TODO: check this
            # embedding_weights = np.vstack((x_train, x_val))
            # embedding_layer = model.get_layer("embedding")
            # embedding_layer.set_weights(embedding_weights)

            sgd = SGD(lr=0.01, nesterov=False)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=sgd,
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=(x_val, y_val))
            # print(y_val)

            Y_random = []
            for idx_r in range(x_test.shape[0]):
                Y_random.append(random.sample([0, 1], 1)[0])

            Y_random = np.array(Y_random).astype(int)

            # score = model.evaluate(x_val, y_val, verbose=2)
            val = model.predict_classes(x_test, batch_size=10, verbose=0)
            f1_values[col_p1 - 3] += sklearn.metrics.f1_score(y_test, val)
            f1_random[col_p1 - 3] += sklearn.metrics.f1_score(y_test, Y_random)

            print("\n", sklearn.metrics.f1_score(y_test, val), "\n")
            print("\n", sklearn.metrics.f1_score(y_test, Y_random), "\n")
            # exit()
            # print('Test loss:', score[0])
            # print('Test accuracy:', score[1])

    for idx in range(9):
        f1_values[idx] /= 2.
        f1_random[idx] /= 2.

    print(f1_values, f1_random)

if __name__ == "__main__":
    main()