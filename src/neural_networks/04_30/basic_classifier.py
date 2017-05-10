import numpy as np
from sklearn.svm import LinearSVC
import pickle
import sklearn.metrics as skm
from sklearn import linear_model
import random

f1_values = [0. for _ in range(9)]
f1_random = [0. for _ in range(9)]

for idx_fold in range(1,2):
    print("Fold: ", idx_fold)
    for col_p1 in range(3, 12):
        print("Col: ", col_p1)
        # output_dir = ''
        output_dir = 'data/05_02/fold_' + str(idx_fold) + '/col/' + str(col_p1-3) + '/'
        x_train = pickle.load(open(output_dir + 'X_train_l+u.pickle', 'rb'))
        y_train = pickle.load(open(output_dir + 'Y_train_l+u.pickle', 'rb'))
        x_test = pickle.load(open(output_dir + 'X_test.pickle', 'rb'))
        y_test = pickle.load(open(output_dir + 'Y_test.pickle', 'rb'))

        x_train = np.mean(x_train, axis=1)
        x_test = np.mean(x_test, axis=1)

        X_test_pos = x_test[y_test == 1]
        X_test_neg = x_test[y_test == -1]
        if X_test_pos.shape[0] < X_test_neg.shape[0]:
            X_test_neg = X_test_neg[:X_test_pos.shape[0]]
        else:
            X_test_pos = X_test_pos[:X_test_neg.shape[0]]

        x_test = np.concatenate((X_test_neg, X_test_pos), axis=0)
        y_test = np.array([-1] * X_test_pos.shape[0] + [1] * X_test_neg.shape[0])

        rand_idx = [random.randint(0, x_train.shape[0] - 1) for p in range(int(0.2 * x_train.shape[0]))]
        train_ind = list(set(list(range(x_train.shape[0]))) - set(rand_idx))
        x_inp = x_train[train_ind, :]
        x_val = x_train[rand_idx, :]
        # print(y_train.shape[0], x_train.shape[0])
        y_inp = y_train[train_ind]
        y_val = y_train[rand_idx]
        #
        svm_model = linear_model.LogisticRegression()#LinearSVC(penalty='l1')
        svm_model.fit(x_train, y_train)
        #
        predict = svm_model.predict(x_test)

        Y_random = []
        for idx_r in range(x_test.shape[0]):
            Y_random.append(random.sample([-1, 1], 1))

        Y_random = np.array(Y_random)

        # print(predict)
        print(skm.f1_score(y_test, predict), skm.f1_score(y_test, Y_random))

