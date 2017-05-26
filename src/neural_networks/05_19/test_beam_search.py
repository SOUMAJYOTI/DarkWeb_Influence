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
eta_2 = 100
MAX_ITER = 20


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
            if corr[l1, l2] < 0.05:
                corr[l1, l2] = 0.

    return corr


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


def beam_search(X, u, w, b, relLabels):
    """
    For each instance search for the best possible set of labels

    :param X: Feature for instance
    :param scores: Initial Label scores
    :param hash_table: Label set for each score
    :param u:
    :param w:
    :param b:
    :param relLabels:
    :return: Best possible set of labels
    """
    candidate_paths = [[] for _ in range(5)] # contains the candidate label sets
    candidate_vals =[[] for _ in range(5)] # contains the label values (-1/1) for each candidate set
    candidate_scores = [0. for _ in range(5)]
    min_score = -1000

    iter = 0
    start = 0
    while True:
        # print("Iter: ", iter)
        intermediate_paths = {}
        # intermediate_paths_val = []
        interim_scores = []
        hash_table = {}

        cnt_paths = 0
        for cp in range(10):
            labels_curr = candidate_paths[cp]
            labels_val_curr = candidate_vals[cp]
            Y = -np.ones((10, 1))
            for lv in range(len(labels_val_curr)):
                Y[labels_curr[lv]] = labels_val_curr[lv]

            for l in range(10):
                candidate_interim = labels_curr[:]
                candidate_vals_interim = labels_val_curr[:]
                if l in labels_curr:
                    continue

                temp_relLabels = []
                for lc in range(len(labels_curr)):
                    temp_relLabels.extend(relLabels[labels_curr[lc]])

                # temp_relLabels = np.array(list(set(temp_relLabels)))
                temp_relLabels = np.array(list(set(relLabels[l]).intersection(set(labels_curr))))
                model_pos = returnModelVal(X, Y, 1.0, u[l], w[l], b[l], temp_relLabels)
                candidate_interim.append(l)

                # print(model_pos)
                if model_pos < 0:
                    # print('hello')
                    candidate_vals_interim.append(-1)
                    interim_scores.append(-model_pos)
                else:
                    candidate_vals_interim.append(1)
                    interim_scores.append(model_pos)

                hash_table[cnt_paths] = candidate_interim
                intermediate_paths[cnt_paths] = candidate_vals_interim
                cnt_paths += 1
            # For the first iteration, just iterate once - all labels in one iteration
            if start == 0:
                start = 1
                break

        # print(interim_scores)
        temp_paths = intermediate_paths
        interim_zip = zip(intermediate_paths, interim_scores)
        sorted_scores = sorted(interim_zip, key=lambda x: x[1], reverse=True)[:10]
        intermediate_paths, scores = zip(*sorted_scores)

        temp_cand = []
        temp_val = []
        for i in range(len(intermediate_paths)):
            temp_cand.append(hash_table[intermediate_paths[i]])
            temp_val.append(temp_paths[intermediate_paths[i]])

        candidate_paths = temp_cand
        candidate_vals = temp_val
        # print(candidate_paths)
        # print(candidate_vals)
        # print(scores)
        # candidate_scores = scores

        # Exit condition from loop
        # if max(interim_scores) < min_score:
        #     break
        #
        # min_score = min(interim_scores)

        if iter > 2:
            break
        iter += 1

    candidate_dict = {}
    for i in range(10):
        for c in range(len(candidate_paths[i])):
            if candidate_paths[i][c] not in candidate_dict:
                candidate_dict[candidate_paths[i][c]] = candidate_vals[i][c]
            elif candidate_dict[candidate_paths[i][c]] != 2:
                if candidate_dict[candidate_paths[i][c]] != candidate_vals[i][c]:
                    candidate_dict[candidate_paths[i][c]] = 2.

    # print(candidate_dict)
    return candidate_dict

    # return candidate_paths[0], candidate_vals[0]
    # Y_final = -np.ones((10, 1))
    # intersect = candidate_paths[0]
    # for cp in range(1, len(candidate_paths)):
    #     intersect = list(set(candidate_paths[cp]).intersection(set(intersect)))
    #
    # for idx in range(len(intersect)):
    #     Y_final[intersect[idx]] = 1.

    # return Y_final.transpose()
    # print(intersect)
    # exit()
        # print(candidate_paths[cp])
        # print(candidate_vals[cp])



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

    rel_labels = [[] for _ in range(10)]
    for col in range(2, 12):
        for l in range(corr.shape[0]):
            if corr[col - 2, l] > 0. and ((col-2) != l):
                rel_labels[col-2].append(l)

        # print(col-2, rel_labels[col-2])

    # rel_labels = np.array(rel_labels)
    for idx_fold in range(0, 1):
        print('\n Fold: ', idx_fold)
        cnt_fold += 1
        X_test = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) +
                                  '/' + 'X_test.pickle', 'rb'))
        Y_test_all = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) +
                                  '/' + 'Y_test_all.pickle', 'rb'))

        train_params = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_'
                                        + str(idx_fold) + '/train_params.pickle', 'rb'))

        """ Initial labels from prediction """
        Y_initial = np.zeros(Y_test_all.shape)
        for col in range(2, 12):
            input_dir = '../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            X_train = pickle.load(open(input_dir + 'X_train_l.pickle', 'rb'))
            Y_train = pickle.load(open(input_dir + 'Y_train_l.pickle', 'rb'))

            clf = svm.LinearSVC(penalty='l2')
            clf.fit(X_train, Y_train)
            Y_initial[:, col - 2] = clf.predict(X_test)

        print(sklearn.metrics.f1_score(Y_test_all[:, 6], Y_initial[:, 6]))
        # print(Y_initial[0])
        """ Prune labels using beam search algorithm """
        # Y_curr = np.zeros(Y_initial.shape)
        print("Testing: ")
        # rel_labels = [list(range(10)) for _ in range(10)]
        Y_pred = np.copy(Y_initial)
        for idx_inst_test in range(1): #X_test.shape[0]):
            # print("Actual: ", Y_test_all[idx_inst_test])
            # print("Predicted: ", Y_initial[idx_inst_test])
            val_dict = beam_search(X_test[idx_inst_test], train_params['u'], train_params['w'],
                        train_params['b'], rel_labels)

            for c in val_dict:
                if val_dict[c] != 2.:
                    Y_pred[idx_inst_test, c] = val_dict[c]

            # print("Before beam", Y_initial[idx_inst_test])
            # print("After beam: ", Y_pred[idx_inst_test])
            # print(val)
            # Y_pred[idx_inst_test,:] = val



            # Y_random = np.array(Y_random)
            # for col in range(2, 4):
            #     print(sklearn.metrics.f1_score(Y_test_all[:, col-2], Y_curr[:, col-2]),
            #           sklearn.metrics.f1_score(Y_test_all[:, col - 2], Y_random))
            # # print(Y_initial)
            # print(Y_initial)

        # print(Y_test_all[:, 0])
        # print(Y_pred[:, 0])
        Y_random = []
        for idx_r in range(X_test.shape[0]):
            Y_random.append(random.sample([-1., 1.], 1))
        print(sklearn.metrics.f1_score(Y_test_all[:, 6], Y_pred[:, 6]),
              sklearn.metrics.f1_score(Y_test_all[:, 6], Y_random))

            #           sklearn.metrics.f1_score(Y_test_all[:, col - 2], Y_random))

    # print("random: ", np.array(random_f1) / len(train_fold))

if __name__ == "__main__":
    main()