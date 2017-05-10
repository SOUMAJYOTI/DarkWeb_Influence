import pandas as pd
import numpy as np
import pickle
import cvxopt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import scipy.stats as scst
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import gensim
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import KFold
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
import random
import matplotlib.pyplot as plt
import seaborn as sns
import SVM_cvxopt as svopt
import eval


def linear_kernel(F_1, F_2):
    return np.dot(F_1.T, F_2)


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
            if corr[l1, l2] < 0.02:
                corr[l1, l2] = 0.

    return corr


def my_kernel(F_1, F_2, relLables):
    # relLabels - related labels to the current label in consideration
    # eta_1 and eta_2 control the weights of inter-label
    # and individual leable contributions to feature model $f$
    eta_1 = 1
    eta_2 = 1.

    # kernel matrix computations - matrix K
    # K_ij = 4(eta_1^2 + eta_2^2*\sum_{t \in N_l} y_t^i y_t^j) \phi(I^i) \phi(I^j)
    sum_pos = 0
    for idx_r in range(relLables.shape[0]):
        sum_pos += (1 * relLables[idx_r])

    sum_neg = 0
    for idx_r in range(relLables.shape[0]):
        sum_neg += (-1 * relLables[idx_r])

    if sum_pos > sum_neg:
        sum_inter_label = sum_pos
    else:
        sum_inter_label = sum_neg

    K = 1. * (eta_1 ** 2 + (eta_2 ** 2 * sum_inter_label)) * rbf_kernel(F_1, F_2)

    return K


# get the matrices required for cvxopt computation
# This function is for each label
def get_matrices(X, Y, curLabel, relLables, C=1000.1):
    # relLabels - related labels to the current label in consideration
    # eta_1 and eta_2 control the weights of inter-label
    # and individual leable contributions to feature model $f$
    eta_1 = 1
    eta_2 = 1.


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
            K[idx_1, idx_2] = 1.*(eta_1**2 + (eta_2**2 * sum_inter_label))*rbf_kernel(X[idx_1], X[idx_2])
            # K[idx_1, idx_2] = linear_kernel(X[idx_1, :], X[idx_2, :])

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


if __name__ == "__main__":
    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)
    Y_labels = np.array(forumsData.ix[:, 3:14])
    for idx in range(Y_labels.shape[0]):
        for idx_1 in range(Y_labels.shape[1]):
            if Y_labels[idx, idx_1] == 0.:
                Y_labels[idx, idx_1] = -1.
    Y_test_all = np.array([])

    corr = labelCorrMatrix(np.array(Y_labels)) # get the correlation

    # clf = svm.LinearSVC(penalty='l2')
    # clf = linear_model.LogisticRegression()

    # FINAL PREDICTED LABELS
    Y_predict_labels = np.zeros(Y_labels.shape)

    dir_fold = '../../../darkweb_data/4_4/folds_data/d200/'
    for col in range(3, 12):
        avg_precision_custom = 0
        avg_recall_custom = 0
        avg_f1_custom = 0

        avg_precision_cvxopt = 0
        avg_recall_cvxopt = 0
        avg_f1_cvxopt = 0

        avg_precision_random = 0

        perc_pos = 0

        cnt_fold = 0
        for idx_t in range(10):
            data_fold = pickle.load(open(dir_fold + 'label_' +
                                         str(col) + '_fold_' + str(idx_t) + '.pickle', 'rb'))
            X_train = data_fold[0]
            Y_train = data_fold[1]
            Y_labels_train = data_fold[2]
            X_test = data_fold[3]
            Y_test = data_fold[4]

            clf = svopt.SVM_cvxopt(kernel=rbf_kernel, C=1000.1)
            clf.fit(X_train, Y_train)


            # clf = svm.SVC(kernel='linear', C=1000.1)
            # clf.fit(X_train, Y_train)

            Y_predict = clf.predict(X_test)

            # RANDOM OUTPUT LABELS
            Y_random = []
            for idx_r in range(X_test.shape[0]):
                Y_random.append(random.sample(range(2), 1))

            Y_random = np.array(Y_random)
            # print(Y_predict, Y_test)
            # avg_precision += sklearn.metrics.f1_score(Y_test, Y_predict)

            # correct = np.sum(Y_predict == Y_test)
            # print("%d out of %d predictions correct" % (correct, len(Y_predict)))

            avg_precision_cvxopt += sklearn.metrics.precision_score(Y_test, Y_predict)
            avg_precision_random += sklearn.metrics.precision_score(Y_test, Y_random)
            # print(sklearn.metrics.precision_score(Y_test, Y_predict))

            rel_labels = []

            #TODO : CHECK THIS IF CORRECT !!!!!!!
            for l in range(corr.shape[0]):
                if corr[col-3, l] > 0.:
                    rel_labels.append(l)

            rel_labels = np.array(rel_labels)
            # print(rel_labels)

            K, P, q, A, G, h, b_opt = get_matrices(X_train, Y_labels_train, col-3, rel_labels)

            opt = cvxopt.solvers.options['show_progress'] = False

            solution = cvxopt.solvers.qp(P, q, G, h, A, b_opt)
            a = np.ravel(solution['x'])

            sv = a > 1e-5
            ind = np.arange(len(a))[sv]
            # print(X.shape)
            a = a[sv]
            Y_params = Y_labels[sv]
            X_params = X_train[sv]
            sv_x = X_train[sv]
            sv_y = Y_train[sv]
            # print("%d support vectors out of %d points" % (len(a), X_train.shape[0]))

            # TODO: FIX THE KERNEL TO INCLUDE RELATED LABELS FOR TEST DATA
            y_pred = np.zeros(len(X_test))
            for i in range(len(X_test)):
                s = 0
                for a_1, sv_y_1, sv_1 in zip(a, sv_y, sv_x):
                    # print(self.kernel(X[i], sv))
                    print(X_test[i].shape, sv_1.shape)
                    s += a_1 * sv_y_1 * K[X_test[i], sv_1]
                y_pred[i] = s

            # Intercept
            b_int = 0
            for n in range(len(a)):
                b_int += sv_y[n]
                b_int -= np.sum(a * sv_y * rbf_kernel(ind[n], sv))
            b_int /= len(a)

            Y_predict = np.sign(y_pred + b_int)

            correct = np.sum(Y_predict == Y_test)
            # print("%d out of %d predictions correct" % (correct, len(Y_predict)))

            # print(Y_predict)
            # print(Y_test)
            avg_precision_custom += sklearn.metrics.f1_score(Y_test, Y_predict)
            # print(sklearn.metrics.precision_score(Y_test, Y_predict))

            cnt_fold += 1

        print(avg_precision_cvxopt/cnt_fold, avg_precision_custom/cnt_fold, avg_precision_random/cnt_fold)
