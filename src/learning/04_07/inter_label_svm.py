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
            if corr[l1, l2] < 0.02:
                corr[l1, l2] = 0.


    # corr = d.corr()

    # sns.set(style="white")

    # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # labels = ['Apprehension', 'Community_Supportive', 'Authenticating', 'Info_seeking', 'Providing_information',
    #           'Prudent_Statement', 'Disgressive', 'Personal_Involvement', 'Sarcastic', 'Sensemaking',
    #           'Directive_Action']
    #
    # l1 = labels
    # plt.xticks(x, labels, rotation=50, ha='center')
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
    #
    # plt.yticks(x, l1, rotation=360)
    # plt.subplots_adjust(bottom=0.20)
    # plt.show()
    # print(corr)
    # exit()
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


def getParams(a, Y, X, relLabels, curLabel):
    eta_1 = 1.
    eta_2 = 0.

    sum_u = np.zeros((1, X.shape[1]))
    a.shape = (a.shape[0], 1)
    for idx in range(a.shape[0]):
        # print(a[idx, 0], Y[idx, 0])
        sum_u += a[idx, 0]*Y[idx, curLabel] * X[idx, :]

    sum_u = 2*eta_1*sum_u

    sum_w = np.zeros((relLabels.shape[0], X.shape[1]))
    for t in range(relLabels.shape[0]):
        for idx in range(a.shape[0]):
            sum_w[t] += (a[idx, 0] * Y[idx, curLabel] * Y[idx, relLabels[t]] * X[idx, :])

        sum_w[t] = 2*eta_2*sum_w[t]

    return sum_u, sum_w


def returnModelVal(u, w, b, y, X, relLabels):
    eta_1 = 1.
    eta_2 = 0.
    X.shape = (X.shape[0], 1)
    # print(u.shape, X.shape)
    p_1 = eta_1*y*(np.dot(u, X) + b)

    p_2 = 0
    for t in range(relLabels.shape[0]):
        p_2 += (y * relLabels[t] * np.dot(w[t, :],X))

    p_2 = p_2*eta_2
    #
    # print(p_1, p_2)

    return p_1 + p_2


def getFolds(Y):
    train_folds = []
    test_folds = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # X_train, X_test = train_index, test_index
        # # Y_train, Y_test = train_index, test_index
        train_folds.append(train_index)
        test_folds.append(test_index)

    return train_folds, test_folds


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

    # prepare the folds for CV test
    train_fold, test_fold = getFolds(Y_labels)


    # SVM classifier
    # clf = svm.LinearSVC(penalty='l2')
    clf = linear_model.LogisticRegression()
    dir_feat = '../../../darkweb_data/3_25/features_d200/'
    for col in range(3, 12):
        X_inst, Y_inst = pickle.load(open(dir_feat + 'feat_label_' + str(col) + '.pickle', 'rb'))
        for idx_y in range(len(Y_inst)):
            if Y_inst[idx_y] == 0:
                Y_inst[idx_y] = -1.
            else:
                Y_inst[idx_y] = 1.

        X_inst = np.array(X_inst)
        X_inst = np.squeeze(X_inst, axis=(1, ))
        Y_inst = np.array(Y_inst)

        # print(Y_inst.shape)

        Y_predict = []
        avg_precision_custom = 0
        avg_recall_custom = 0
        avg_f1_custom = 0

        avg_precision_cvxopt = 0
        avg_recall_cvxopt = 0
        avg_f1_cvxopt = 0

        perc_pos = 0

        cnt_fold = 0
        # print(X_inst[0])
        for idx_t in range(len(train_fold)):
            # X_train = X_inst[:380, :]
            # Y_train = Y_inst[:380]
            # # print(Y_train)
            # X_test = X_inst[380:, :]
            # Y_test = Y_inst[380:]
            # Y_labels_train = Y_labels[:380]

            X_train = X_inst[train_fold[idx_t]]
            Y_train = Y_inst[train_fold[idx_t]]
            Y_labels_train = Y_labels[train_fold[idx_t]]
            X_test = X_inst[test_fold[idx_t]]
            Y_test = Y_inst[test_fold[idx_t]]

            clf = svopt.SVM_cvxopt(kernel=rbf_kernel, C=1000.1)
            clf.fit(X_train, Y_train)


            # clf = svm.SVC(kernel='linear', C=1000.1)
            # clf.fit(X_train, Y_train)

            Y_predict = clf.predict(X_test)

            # print((len(Y_test[Y_test == 1.]) + len(Y_train[Y_train == 1.]) )/ (len(Y_test) + len(Y_train)))
            # perc_pos += len(Y_test[Y_test == 1.]) / len(Y_test)
            # clf.fit(X_train, Y_train)

            # print(clf.coef_[0])
            # Y_predict = clf.predict(X_test)
            # print(Y_test, Y_predict)
            #
            # Y_random = []
            # for idx_r in range(X_test.shape[0]):
            #     Y_random.append(random.sample(range(2), 1))
            #
            # Y_random = np.array(Y_random)
            # # print(Y_predict, Y_test)
            # avg_precision += sklearn.metrics.f1_score(Y_test, Y_predict)

            correct = np.sum(Y_predict == Y_test)
            # print("%d out of %d predictions correct" % (correct, len(Y_predict)))

            avg_precision_cvxopt += sklearn.metrics.f1_score(Y_test, Y_predict)
            print(sklearn.metrics.precision_score(Y_test, Y_predict))

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

            y_predict = np.zeros(len(X_test))
            for i in range(len(X_test)):
                s = 0
                for a_1, sv_y_1, sv_1 in zip(a, sv_y, sv_x):
                    # print(self.kernel(X[i], sv))
                    s += a_1 * sv_y_1 * rbf_kernel(X_test[i], sv_1)
                y_predict[i] = s

            # Intercept
            b_int = 0
            for n in range(len(a)):
                b_int += sv_y[n]
                b_int -= np.sum(a * sv_y * K[ind[n], sv])
            b_int /= len(a)

            Y_predict = np.sign(y_predict + b_int)

            correct = np.sum(Y_predict == Y_test)
            # print("%d out of %d predictions correct" % (correct, len(Y_predict)))

            # print(Y_predict)
            # print(Y_test)
            avg_precision_custom += sklearn.metrics.f1_score(Y_test, Y_predict)
            print(sklearn.metrics.precision_score(Y_test, Y_predict))

            u, w = getParams(a, Y_params, X_params, rel_labels, col - 3)


            Y_predict = []
            for idx_test in range(X_test.shape[0]):
                model_val_y_pos = returnModelVal(u, w, b_int, 1., X_test[idx_test, :], rel_labels)
                model_val_y_neg = returnModelVal(u, w, b_int, -1., X_test[idx_test, :], rel_labels)
                # print(model_val_y_1[0][0], model_val_y_neg1[0][0])
                if model_val_y_pos[0][0] > model_val_y_neg[0][0]:
                    Y_predict.append(1.)
                else:
                    Y_predict.append(-1.)

            cnt_fold += 1

        print(avg_precision_cvxopt/cnt_fold, avg_precision_custom/cnt_fold)
