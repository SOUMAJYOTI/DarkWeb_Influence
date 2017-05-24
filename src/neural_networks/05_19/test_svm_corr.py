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


    # corr = d.corr()

    # sns.set(style="white")
    #
    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    #
    # labels = ['Apprehension', 'Community_Supportive', 'Authenticating', 'Info_seeking', 'Providing_information',
    #           'Prudent_Statement', 'Disgressive', 'Personal_Involvement', 'Sarcastic', 'Sensemaking',
    #           'Directive_Action']
    #
    # l1 = labels
    # l1.reverse()
    # x = range(len(l1))
    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))
    #
    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(50, 100, as_cmap=True)
    #
    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, cmap=cmap,
    #             square=True,
    #             linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    #
    # plt.yticks(x, l1, rotation=360)
    # labels = ['Apprehension', 'Community_Supportive', 'Authenticating', 'Info_seeking', 'Providing_information',
    #           'Prudent_Statement', 'Disgressive', 'Personal_Involvement', 'Sarcastic', 'Sensemaking',
    #           'Directive_Action']
    # plt.xticks(x, labels, rotation=50)
    # plt.subplots_adjust(bottom=0.20)
    # plt.show()

    return corr


def my_kernel(F_1, F_2, relLabels, y_1, y_2):
    eta_1 = 1.
    eta_2 = 0.2

    sum_inter_label = 0
    for idx_r in range(relLabels.shape[0]):
        sum_inter_label += y_1[relLabels[idx_r]] * y_2[relLabels[idx_r]]
    K = 1. * (eta_1 ** 2 + (eta_2 ** 2 * sum_inter_label)) * rbf_kernel(F_1, F_2)

    return K


# get the matrices required for cvxopt computation
# This function is for each label
def get_matrices(X, Y, curLabel, relLables, C=1000.1):
    # relLabels - related labels to the current label in consideration
    # eta_1 and eta_2 control the weights of inter-label
    # and individual leable contributions to feature model $f$
    eta_1 = 1.
    eta_2 = 0.2

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
            K[idx_1, idx_2] = 4.*(eta_1**2 + (eta_2**2 * sum_inter_label)) * \
                              (rbf_kernel(X[idx_1], X[idx_2]) )
            # K[idx_1, idx_2] = linear_kernel(X[idx_1, :], X[idx_2, :])
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


# solve the SVM for each label
def project(X, a, w, b, sv, sv_y):
    if w is not None:
        return np.dot(X, w) + b
    else:
        Y_predict = np.zeros((len(X), 1))
        print(Y_predict.shape)
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(a, sv_y, sv):
                s += a * sv_y * linear_kernel(X[i], sv)
            print(s)
            Y_predict[i, 0] = s
        return Y_predict + b


def predict(X):
    return np.sign(project(X))

# parameters for each label l \in L
def getParams(a, X, Y, relLabels, curLabel):
    eta_1 = 1.
    eta_2 = 0.2

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

    eta_1 = 1.
    eta_2 = 0.2
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
        cnt_fold += 1
        X_test = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v1/fold_' + str(idx_fold) +
                                  '/' + 'X_test.pickle', 'rb'))
        Y_test_all = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v1/fold_' + str(idx_fold) +
                                  '/' + 'Y_test_all.pickle', 'rb'))

        """ Initial labels from prediction """
        Y_initial = np.zeros(Y_test_all.shape)
        for col in range(2, 12):
            input_dir = '../../../darkweb_data/05/5_19/data_test/v1/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            X_train = pickle.load(open(input_dir + 'X_train_l.pickle', 'rb'))
            Y_train = pickle.load(open(input_dir + 'Y_train_l.pickle', 'rb'))
            Y_train_all = pickle.load(open(input_dir + 'Y_train_all.pickle', 'rb'))

            clf = svm.LinearSVC(penalty='l2')
            clf.fit(X_train, Y_train)
            Y_initial[:, col-2] = clf.predict(X_test)

        for col in range(2, 12):
            Y_test = Y_test_all[:, col-2]

            """ Sample test data """
            X_test_pos = []
            X_test_neg = []
            Y_test_initial_pos = []
            Y_test_initial_neg = []
            for idx in range(X_test.shape[0]):
                if Y_test[idx] == 1.:
                    X_test_pos.append(X_test[idx])
                    Y_test_initial_pos.append(Y_initial[idx])
                else:
                    X_test_neg.append(X_test[idx])
                    Y_test_initial_neg.append(Y_initial[idx])

            X_test_pos = np.array(X_test_pos)
            X_test_neg = np.array(X_test_neg)

            if X_test_pos.shape[0] < X_test_neg.shape[0]:
                X_test_neg = X_test_neg[:X_test_pos.shape[0]]
                Y_test_initial_neg = Y_test_initial_neg[:X_test_pos.shape[0]]
            else:
                X_test_pos = X_test_pos[:X_test_neg.shape[0]]
                Y_test_initial_pos = Y_test_initial_pos[:X_test_neg.shape[0]]

            Y_test_initial = np.concatenate((Y_test_initial_neg, Y_test_initial_pos), axis=0)
            X_test_final = np.concatenate((X_test_neg, X_test_pos), axis=0)
            Y_test_final = np.array([-1.] * X_test_neg.shape[0] + [1.] * X_test_pos.shape[0])

            rel_labels = []
            # TODO : CHECK THIS IF CORRECT !!!!!!!
            for l in range(corr.shape[0]):
                if corr[col - 2, l] > 0. and ((col-2) != l):
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

            sv = a > 1e-5
            ind = np.arange(len(a))[sv]

            a = a[sv]

            Y_params = Y_train_all[sv]
            X_params = X_train[sv]
            sv_x = X_train[sv]
            sv_y = Y_train_all[sv]
            # print("%d support vectors out of %d points" % (len(a), X_train.shape[0]))

            u, w_t = getParams(a, X_params, Y_params, rel_labels, col-2)

            # get weights
            w_svm = np.dot(np.transpose(a) * Y_params[:, col-2], X_params)

            # get bias - ANY INSTANCE
            b_svm = Y_params[0, col-2] - np.dot(w_svm, np.transpose(X_params[0]))

            for idx_inst in range(X_test.shape[0]):
                model_val_pos = returnModelVal(X_test[idx_inst], Y_test_initial[idx_inst,:], 1.0, u, w_t, b_svm, rel_labels)
            exit()
            y_predict = np.zeros(len(X_test))

            for i in range(len(X_test)):
                s = 0
                for a_1, sv_y_1, sv_1 in zip(a, sv_y, sv_x):
                    # print(self.kernel(X[i], sv))
                    # s += a_1 * sv_y_1 * int_svm.rbf_kernel(X_test_3[i], sv_1)
                    s += a_1 * sv_y_1 * int_svm.my_kernel(X_test_3[i], sv_1, rel_labels, Y_predict_2_out[i],
                                                          Y_params[i])
                y_predict[i] = s

            # Intercept
            b_int = 0
            for n_p3 in range(len(a)):
                b_int += sv_y[n_p3]
                b_int -= np.sum(a * sv_y * K[ind[n_p3], sv])
            b_int /= len(a)

            Y_predict_3_out = np.sign(y_predict + b_int)

            Y_predict_final[:, col_p3 - 3] = Y_predict_3_out



    # print("random: ", np.array(random_f1) / len(train_fold))

if __name__ == "__main__":
    main()