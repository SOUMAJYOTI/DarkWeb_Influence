import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
import sklearn
import random
from sklearn.model_selection import LeaveOneOut
import operator
import warnings
import os
from sklearn.svm import SVC
from sklearn import svm
from sklearn.cluster import AffinityPropagation
import scipy.stats as scst
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import cvxopt

eta_1 = 1
eta_2 = 5


def linear_kernel(F_1, F_2):
    return  np.dot(F_1.T, F_2)


def rbf_kernel(F_1, F_2, sigma_2=1.0):
    return np.exp(-0.5*np.linalg.norm(F_1 - F_2)**2 / sigma_2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def labelCorrMatrix(Y):
    corr = np.zeros((Y.shape[1], Y.shape[1]))
    for l1 in range(Y.shape[1]):
        for l2 in range(Y.shape[1]):
            if l1 == l2:
                continue
            for idx in range(Y.shape[0]):
                if Y[idx, l1] == Y[idx, l2] and Y[idx, l1] == 1.:
                    corr[l1, l2] += 1

            corr[l1, l2] /= Y.shape[0]
            # 0.02 is the threshold for the probability
            if corr[l1, l2] < 0.03:
                corr[l1, l2] = 0.

    return corr


# get the matrices required for cvxopt computation
# This function is for each label
def get_matrices(X, Y, curLabel, relLables, C=1000.1):
    # relLabels - related labels to the current label in consideration
    # eta_1 and eta_2 control the weights of inter-label
    # and individual leable contributions to feature model $f$
    # eta_1 = 1.
    # eta_2 = 0.5

    # the svm objective function is:
    # \frac{1}{2} \alpha^T K \alpha + \alpha 1^T
    # sb. to Y \alpha = 0
    # and  \alpha < \lambda

    # kernel matrix computations - matrix K
    # K_ij = 4(eta_1^2 + eta_2^2*\sum_{t \in N_l} y_t^i y_t^j) \phi(I^i) \phi(I^j)
    K = np.zeros((X.shape[0], X.shape[0]))
    for idx_1 in range(X.shape[0]):
        for idx_2 in range(X.shape[0]):
            sum_inter_label = 0
            for idx_r in range(relLables.shape[0]):
                sum_inter_label += (Y[idx_1, relLables[idx_r]]*Y[idx_2, relLables[idx_r]])
            # K[idx_1, idx_2] = 1.*(eta_1**2 + (eta_2**2 * sum_inter_label)) * \
            #                   (linear_kernel(X[idx_1], X[idx_2]) )
            K[idx_1, idx_2] = linear_kernel(X[idx_1, :], X[idx_2, :])
            # print(K[idx_1, idx_2])

    P = cvxopt.matrix(np.outer(Y[:, curLabel], Y[:, curLabel]) * K)

    q = cvxopt.matrix(-np.ones((X.shape[0], 1)))

    A = Y[:, curLabel]
    A.shape = (1, A.shape[0])
    A = cvxopt.matrix(A)

    b = cvxopt.matrix([0.])

    G_1 = np.diag(np.ones(X.shape[0]) * -1) #-np.identity(X.shape[0]) # first constraint: all \alpha >= 0
    G_2 = np.identity(X.shape[0]) # second constraint: all \alpha <= \lambda
    G = cvxopt.matrix(np.vstack((G_1, G_2)))

    h_1 = np.zeros((X.shape[0], 1))
    h_2 = np.ones((X.shape[0], 1))*(C)
    h = cvxopt.matrix(np.vstack((h_1, h_2)))

    return K, P, q, A, G, h, b


# parameters for each label l \in L
def getParams(a, X, Y, relLabels, curLabel):
    # eta_1 = 1.
    # eta_2 = 0.5

    sum_u = np.zeros((1, X.shape[1]))
    a.shape = (a.shape[0], 1)
    for idx in range(a.shape[0]):
        # print(a[idx, 0], Y[idx, 0])
        sum_u += a[idx, 0]*Y[idx, curLabel] * X[idx, :]
    sum_u = 2*eta_1*sum_u

    sum_w = np.zeros((relLabels.shape[0], X.shape[1]))
    for t in range(relLabels.shape[0]):
        temp = 0
        for idx in range(a.shape[0]):
            temp += (a[idx, 0] * Y[idx, curLabel] * Y[idx, relLabels[t]] * X[idx, :])
        sum_w[t] = 2*eta_2*temp

    return sum_u, sum_w


def returnModelVal(X, Y, y_fix, u, w, b, relLabels):
    """
    :param X: Feature vector for instance i
    :param Y: Initial test labels for instance i
    :param y_fix: either -1. or 1.
    :param u: parameter for SVM
    :param w: parameter for SVM
    :param b: parameter for SVM
    :param relLabels: related labels for label l
    :return: f value
    """

    # eta_1 = 1.
    # eta_2 = 0.5
    # X.shape = (X.shape[0], 1)

    p_1 = eta_1*y_fix*(np.dot(u, np.transpose(X)) + b)

    p_2 = 0
    for t in range(relLabels.shape[0]):
        p_2 += (y_fix * Y[relLabels[t]] * np.dot(w[t], X))
    p_2 = p_2*eta_2

    return p_1 + p_2


def main():
    forumsData = pd.read_csv('../../../darkweb_data/05/5_15/Forum_40_labels.csv', encoding="ISO-8859-1")

    cnt_fold = 0
    Y_labels = np.array(forumsData.ix[:, 2:12])
    Y_labels_new = np.zeros(Y_labels.shape)
    for idx in range(Y_labels.shape[0]):
        Y_labels[idx][np.isnan(Y_labels[idx])] = 0
        if np.count_nonzero(Y_labels[idx]) < 2:
            continue
        for idx_1 in range(Y_labels.shape[1]):
            if Y_labels[idx, idx_1] == 0.:
                Y_labels_new[idx, idx_1] = -1.
            else:
                Y_labels_new[idx, idx_1] = 1.

    Y_labels_new = Y_labels_new[~np.all(Y_labels_new == 0., axis=1)]

    corr = labelCorrMatrix(np.array(Y_labels_new)) # CORRELATION MATRIX

    for idx_fold in range(0, 5):
        print('\n Fold: ', idx_fold)
        cnt_fold += 1
        X_test = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) +
                                  '/' + 'X_test.pickle', 'rb'))
        Y_test_all = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) +
                                  '/' + 'Y_test_all.pickle', 'rb'))
        Y_test_initial = []
        X_test_new = []
        Y_test_new = []

        """ Initial labels from prediction """
        Y_initial = np.zeros(Y_test_all.shape)
        for col in range(2, 12):
            input_dir = '../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            X_train = pickle.load(open(input_dir + 'X_train_l.pickle', 'rb'))
            Y_train = pickle.load(open(input_dir + 'Y_train_l.pickle', 'rb'))

            clf = svm.LinearSVC(penalty='l2')
            clf.fit(X_train, Y_train)
            Y_initial[:, col-2] = clf.predict(X_test)

            # Y_test = Y_test_all[:, col - 2]

            """ Sample test data """
            # X_test_pos = []
            # X_test_neg = []
            # Y_test_initial_pos = []
            # Y_test_initial_neg = []
            # for idx in range(X_test.shape[0]):
            #     if Y_test[idx] == 1.:
            #         X_test_pos.append(X_test[idx])
            #         Y_test_initial_pos.append(Y_initial[idx])
            #     else:
            #         X_test_neg.append(X_test[idx])
            #         Y_test_initial_neg.append(Y_initial[idx])
            #
            # X_test_pos = np.array(X_test_pos)
            # X_test_neg = np.array(X_test_neg)
            #
            # if X_test_pos.shape[0] < X_test_neg.shape[0]:
            #     X_test_neg = X_test_neg[:X_test_pos.shape[0]]
            #     Y_test_initial_neg = Y_test_initial_neg[:X_test_pos.shape[0]]
            # else:
            #     X_test_pos = X_test_pos[:X_test_neg.shape[0]]
            #     Y_test_initial_pos = Y_test_initial_pos[:X_test_neg.shape[0]]
            #
            # Y_test_initial.append(np.concatenate((Y_test_initial_neg, Y_test_initial_pos), axis=0))
            # X_test_new.append(np.concatenate((X_test_neg, X_test_pos), axis=0))
            # Y_test_new.append(np.array([-1.] * X_test_neg.shape[0] + [1.] * X_test_pos.shape[0]))

        Y_curr = Y_initial
        Y_random = []
        for idx_r in range(X_test.shape[0]):
            Y_random.append(random.sample([-1., 1.], 1))

        Y_random = np.array(Y_random)
        for col in range(2, 4):
            print(sklearn.metrics.f1_score(Y_test_all[:, col - 2], Y_curr[:, col - 2]),
                  sklearn.metrics.f1_score(Y_test_all[:, col - 2], Y_random))

        u_l = []
        b_l = []
        w_l = []
        # Iterated Conditional Modes
        # for iter_predict in range(1):
            # print('Iter: ', iter_predict)
        print("Training: ")
        for col in range(2, 12):
            print("Col: ", col-2)
            # print('Iter: {}, Column: {}'.format(iter_predict, col))
            input_dir = '../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) + '/col_' + str(
                col) + '/'
            X_train = pickle.load(open(input_dir + 'X_train_l.pickle', 'rb'))
            Y_train_all = pickle.load(open(input_dir + 'Y_train_all.pickle', 'rb'))

            # Y_test = Y_test_all[:, col - 2]

            # X_test = X_test_new[col-2]
            # Y_test = Y_test_new[col-2]
            # Y_prev = Y_test_initial[col-2]

            # X_train = Y
            rel_labels = []
            # TODO : CHECK THIS IF CORRECT !!!!!!!
            for l in range(corr.shape[0]):
                if col-2 != l: #corr[col - 2, l] > 0. and ((col-2) != l):
                    rel_labels.append(l)

            rel_labels = np.array(rel_labels)

            for idx_ind1 in range(Y_train_all.shape[0]):
                for idx_ind2 in range(Y_train_all.shape[1]):
                    if Y_train_all[idx_ind1, idx_ind2] == 0.:
                        Y_train_all[idx_ind1, idx_ind2] = -1.

            K, P, q, A, G, h, b_opt = get_matrices(X_train, Y_train_all, col - 2, rel_labels)

            """ CVXOPT SOLUTION """
            opt = cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h, A, b_opt)
            a = np.ravel(solution['x'])
            # print(a.shape)
            # print(a)
            sv = a > 1e-1
            a = a[sv]
            # print(a.shape)
            # exit()
            """ Get the parameters """
            Y_params = Y_train_all[sv]
            X_params = X_train[sv]

            u, w_t = getParams(a, X_params, Y_params, rel_labels, col-2)
            u_l.append(u)
            w_l.append(w_t)
            # get weights
            w_svm = np.dot(np.transpose(a) * Y_params[:, col-2], X_params)
            # get bias - ANY INSTANCE
            b_svm = Y_params[0, col-2] - np.dot(w_svm, np.transpose(X_params[0]))
            b_l.append(b_svm)

        train_params = {'u': u_l, 'w': w_l, 'b': b_l}
        pickle.dump(train_params, open('../../../darkweb_data/05/5_19/data_test/v3/fold_'
                                       + str(idx_fold) + '/train_params.pickle', 'wb'))

        exit()
        Y_curr = Y_initial
        print("Testing: ")
        for iter_predict in range(5):
            print('Iter: ', iter_predict)
            for col in range(2, 12):
                rel_labels = []
                # TODO : CHECK THIS IF CORRECT !!!!!!!
                for l in range(corr.shape[0]):
                    if True: #corr[col - 2, l] > 0. and ((col - 2) != l):
                        rel_labels.append(l)

                rel_labels = np.array(rel_labels)
                for idx_inst_test in range(X_test.shape[0]):
                    # print(Y_initial[idx_inst_test,:])
                    model_val_pos = returnModelVal(X_test[idx_inst_test], Y_initial[idx_inst_test,:], 1.0, u_l[col-2]
                                                   , w_l[col-2], b_l[col-2], rel_labels)
                    model_val_neg = returnModelVal(X_test[idx_inst_test], Y_initial[idx_inst_test,:], -1.0, u_l[col-2]
                                                   , w_l[col-2], b_l[col-2], rel_labels)
                    if model_val_pos > model_val_neg:
                        Y_curr[idx_inst_test, col-2] = 1.
                    else:
                        Y_curr[idx_inst_test, col-2] = -1.

            Y_random = np.array(Y_random)
            for col in range(2, 12):
                print(sklearn.metrics.f1_score(Y_test_all[:, col-2], Y_curr[:, col-2]),
                      sklearn.metrics.f1_score(Y_test_all[:, col - 2], Y_random))
            # print(Y_initial)
            Y_initial = Y_curr
            # print(Y_initial)



    # print("random: ", np.array(random_f1) / len(train_fold))

if __name__ == "__main__":
    main()