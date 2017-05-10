import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
import words_sentences_svm as sent_svm
import sklearn
import random
import matplotlib.pyplot as plt
import inter_label_svm as int_svm
import SVM_cvxopt as svopt
import eval as eval_met
from imblearn.over_sampling import SMOTE
import itertools
import cvxopt


def min_label_coherence(label_vectors, y):
    sum_min = 10000.
    for l in range(len(label_vectors)):
        sum_temp = 0.
        label_cur = label_vectors[l]
        for cur in range(len(label_cur)):
            sum_temp += np.absolute((label_cur[cur] - y[cur]))

        if sum_temp < sum_min:
            sum_min = sum_temp
            label_output = label_cur

    return label_output


def getFolds(Y):
    train_folds = []
    test_folds = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(Y):
        train_folds.append(train_index)
        test_folds.append(test_index)

    return train_folds, test_folds


if __name__ == "__main__":

    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)
    cnt_fold = 0

    """ EVALUATION STRUCTURES  """
    avg_label_precision = [0. for _ in range(9)]
    avg_label_recall = [0. for _ in range(9)]
    avg_label_F1 = [0. for _ in range(9)]

    avg_label_precision_3 = [0. for _ in range(9)]
    avg_label_recall_3 = [0. for _ in range(9)]
    avg_label_F1_3 = [0. for _ in range(9)]

    avg_haming_loss = 0.
    avg_accuracy = 0.
    avg_exact_match = 0.
    avg_f1_measure = 0.
    avg_macro_f1 = 0.
    avg_micro_f1 = 0.

    """" PHASE 1 DATA """
    """  Corpora words have dimension 100"""
    w2v_feat = pickle.load(open('../../../darkweb_data/3_25/word2vec_train_model_d100.pickle', 'rb'))
    docs = pickle.load(open('../../../darkweb_data/4_8/docs_corpora_label_tfidf.pickle', 'rb'))

    Y_labels = np.array(forumsData.ix[:, 3:12])
    for idx in range(Y_labels.shape[0]):
        for idx_1 in range(Y_labels.shape[1]):
            if Y_labels[idx, idx_1] == 0.:
                Y_labels[idx, idx_1] = -1.

    corr = int_svm.labelCorrMatrix(np.array(Y_labels)) # CORRELATION MATRIX

    """ prepare the folds for CV test """
    train_fold, test_fold = getFolds(Y_labels)

    """ SVM Classifier used """
    clf = sklearn.svm.SVC(kernel='rbf', C=1000.1)

    for idx_fold in range(2): #range(len(train_fold)):
        print('Fold: ', idx_fold, '\n')

        Y_true = Y_labels[test_fold[idx_fold]]  # TRUE LABELS
        Y_predict_final = -np.ones(Y_true.shape)

        print("Phase 1....")
        map_test_indices = {}  # This is to keep track of the test indices
        for idx_indicator in range(len(test_fold[idx_fold])):
            map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator

        cnt_fold += 1
        Y_predict_1_out = -np.ones(Y_true.shape)

        for col_p1 in range(3, 12):
            # print("Col: ", col-3)
            """ PHASE 1 FEATURES"""
            X_inst_1, Y_inst_1, row_indices, pos_perc, neg_perc = \
                sent_svm.get_X_Y_data(forumsData, docs, w2v_feat, col_p1)

            X_inst_1 = np.array(X_inst_1)
            Y_inst_1 = np.array(Y_inst_1)

            row_indices = np.array(row_indices)

            train_indices = []
            test_indices = []
            train_row_indices = []
            test_row_indices = []
            for inst_index in range(len(train_fold[idx_fold])):
                train_indices.extend(list(np.where(row_indices == train_fold[idx_fold][inst_index])[0]))
                train_row_indices.extend(
                    list(row_indices[np.where(row_indices == train_fold[idx_fold][inst_index])[0]]))

            for inst_index in range(len(test_fold[idx_fold])):
                test_indices.extend(list(np.where(row_indices == test_fold[idx_fold][inst_index])[0]))
                test_row_indices.extend(list(row_indices[np.where(row_indices == test_fold[idx_fold][inst_index])[0]]))

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            """ SET THE INSTANCES FOR THIS COLUMN"""
            X_train_1 = X_inst_1[train_indices]
            Y_train_1 = Y_inst_1[train_indices]
            X_test_1 = X_inst_1[test_indices]
            Y_test_1 = Y_inst_1[test_indices] # TODO: NOT USED !!!

            """ PHASE 1 RESULTS"""
            clf.fit(X_train_1, Y_train_1)

            # Y_random = []
            # for idx_r in range(X_test.shape[0]):
            #     Y_random.append(random.sample(range(2), 1))
            #
            # Y_random = np.array(Y_random)

            Y_predict_1 = clf.predict(X_test_1)

            """If at least one sentence in a doc is positive, the doc is +ve for that label"""

            for y_p in range(Y_predict_1.shape[0]):
                if Y_predict_1[y_p] == 1.:
                    Y_predict_1_out[map_test_indices[test_row_indices[y_p]], col_p1-3] = 1.


        """ PHASE 2 : SENTENCE AND DOC LEVEL COHERENCE"""
        print("Phase 2....\n")
        Y_labels_train = Y_labels[train_fold[idx_fold]]
        # Find the number of unique labels vectors
        # label_vectors = []
        # for idx_labels in range(Y_labels_train.shape[0]):
        #     if list(Y_labels_train[idx_labels, :]) not in label_vectors:
        #         label_vectors.append(list(Y_labels_train[idx_labels, :]))

        # print("Number of unique label vectors: ", len(label_vectors))

        label_vectors = list(map(list, itertools.product([-1., 1.], repeat=9)))

        Y_predict_2_out = -np.ones(Y_predict_1_out.shape)
        for idx_p2 in range(Y_predict_1_out.shape[0]):
            Y_predict_2_out[idx_p2] = min_label_coherence(label_vectors, Y_predict_1_out[idx_p2])

        # print(Y_predict_2_out.shape)
        # for l in range(Y_predict_2_out.shape[1]):
        #     avg_label_precision[l] += sklearn.metrics.precision_score(Y_true[:, l], Y_predict_2_out[:, l])
        #     avg_label_recall[l] += sklearn.metrics.recall_score(Y_true[:, l], Y_predict_2_out[:, l])
        #     avg_label_F1[l] += sklearn.metrics.f1_score(Y_true[:, l], Y_predict_2_out[:, l])

        # for l in range(len(avg_label_precision)):
        #     avg_label_precision[l] /= 10.
        #     avg_label_recall[l] /= 10.
        #     avg_label_F1[l] /= 10.
        #
        # print(avg_label_precision)
        # print(avg_label_recall)
        # print(avg_label_F1)


        # """ PHASE 3: CONSIDER INTER LABEL CORRELATION"""
        print("Phase 3..... \n")
        # Y_predict_3_out = -np.ones(Y_predict_2_out.shape)
        # clf = sklearn.svm.LinearSVC(penalty='l2')
        dir_feat = '../../../darkweb_data/3_25/features_d200/'
        for col_p3 in range(3, 12):
            X_inst_3, Y_inst_3 = pickle.load(open(dir_feat + 'feat_label_' + str(col_p3) + '.pickle', 'rb'))
            for idx_y in range(len(Y_inst_3)):
                if Y_inst_3[idx_y] == 0:
                    Y_inst_3[idx_y] = -1.
                else:
                    Y_inst_3[idx_y] = 1.

            X_inst_3 = np.array(X_inst_3)
            X_inst_3 = np.squeeze(X_inst_3, axis=(1,))
            Y_inst_3 = np.array(Y_inst_3)

            X_train_3 = X_inst_3[train_fold[idx_fold]]
            Y_train_3 = Y_inst_3[train_fold[idx_fold]]
            Y_labels_train = Y_labels[train_fold[idx_fold]]
            X_test_3 = X_inst_3[test_fold[idx_fold]]
            Y_test_3 = Y_inst_3[test_fold[idx_fold]]

            # clf_p3 = svopt.SVM_cvxopt(kernel=int_svm.rbf_kernel, C=1000.1)

            # SMOTE SAMPLING
            # sm = SMOTE(random_state=42)
            # X_res, Y_res = sm.fit_sample(X_train_3, Y_train_3)
            # clf_p3.fit(X_train_3, Y_train_3)

            # clf_p3 = sklearn.linear_model.LogisticRegression()
            # clf_p3.fit(X_train, Y_train)
            #
            # Y_predict_3 = clf_p3.predict(X_test_3)
            # avg_label_precision_3[col_p3 - 3] += sklearn.metrics.precision_score(Y_test_3, Y_predict_3)
            # avg_label_recall_3[col_p3 - 3] += sklearn.metrics.recall_score(Y_test_3, Y_predict_3)
            # avg_label_F1_3[col_p3 - 3] += sklearn.metrics.f1_score(Y_test_3, Y_predict_3)
            # print(Y_predict_3)
            # Y_predict_3_out[:, col_p3-3] = Y_predict_3

            # print(eval_met.hamming_loss(Y_true, Y_predict_3_out))

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
            # avg_precision += sklearn.metrics.precision_score(Y_test_3, Y_predict_3)

            # print(sklearn.metrics.precision_score(Y_test_3, Y_predict_3))

            rel_labels = []

            # TODO : CHECK THIS IF CORRECT !!!!!!!
            for l in range(corr.shape[0]):
                if corr[col_p3 - 3, l] > 0.:
                    rel_labels.append(l)

            rel_labels = np.array(rel_labels)
            # print(rel_labels)

            K, P, q, A, G, h, b_opt = int_svm.get_matrices(X_train_3, Y_labels_train, col_p3 - 3, rel_labels)

            """ CVXOPT SOLUTION """
            opt = cvxopt.solvers.options['show_progress'] = False

            solution = cvxopt.solvers.qp(P, q, G, h, A, b_opt)
            a = np.ravel(solution['x'])

            sv = a > 1e-5
            ind = np.arange(len(a))[sv]

            a = a[sv]
            Y_params = Y_labels[sv]
            X_params = X_train_3[sv]
            sv_x = X_train_3[sv]
            sv_y = Y_train_3[sv]
            # print("%d support vectors out of %d points" % (len(a), X_train.shape[0]))

            y_predict = np.zeros(len(X_test_3))

            for i in range(len(X_test_3)):
                s = 0
                for a_1, sv_y_1, sv_1 in zip(a, sv_y, sv_x):
                    # print(self.kernel(X[i], sv))
                    # s += a_1 * sv_y_1 * int_svm.rbf_kernel(X_test_3[i], sv_1)
                    s += a_1 * sv_y_1 * int_svm.my_kernel(X_test_3[i], sv_1, rel_labels, Y_predict_2_out[i], Y_params[i])
                y_predict[i] = s

            # Intercept
            b_int = 0
            for n_p3 in range(len(a)):
                b_int += sv_y[n_p3]
                b_int -= np.sum(a * sv_y * K[ind[n_p3], sv])
            b_int /= len(a)

            Y_predict_3_out = np.sign(y_predict + b_int)

            Y_predict_final[:, col_p3-3] = Y_predict_3_out

            # avg_label_precision[col_p3-3] += sklearn.metrics.precision_score(Y_test_3, Y_predict_3_out)
            # avg_label_recall[col_p3-3] += sklearn.metrics.recall_score(Y_test_3, Y_predict_3_out)
            # avg_label_F1[col_p3-3] += sklearn.metrics.f1_score(Y_test_3, Y_predict_3_out)

            # print(sklearn.metrics.precision_score(Y_test_3, Y_predict_3_out))

        avg_haming_loss += eval_met.hamming_loss(Y_true, Y_predict_final)
        avg_accuracy += eval_met.accuracy(Y_true, Y_predict_final)
        avg_exact_match += eval_met.accuracy(Y_true, Y_predict_final)
        avg_f1_measure += eval_met.F1_measure(Y_true, Y_predict_final)
        avg_macro_f1 += eval_met.macro_f1(Y_true, Y_predict_final)
        avg_micro_f1 += eval_met.micro_f1(Y_true, Y_predict_final)

    print(avg_haming_loss/2, avg_accuracy/2, avg_exact_match/2, avg_f1_measure/2, avg_macro_f1/2, avg_micro_f1/2)
    #
    # for l in range(len(avg_label_precision)):
    #     avg_label_precision[l] /= 10.
    #     avg_label_recall[l] /= 10.
    #     avg_label_F1[l] /= 10.
    #
    #     avg_label_precision_3[l] /= 10.
    #     avg_label_recall_3[l] /= 10.
    #     avg_label_F1_3[l] /= 10.
    #
    # print(avg_label_precision)
    # print(avg_label_recall)
    # print(avg_label_F1)
    #
    # print(avg_label_precision_3)
    # print(avg_label_recall_3)
    # print(avg_label_F1_3)
    #
    #
    #     # print(avg_precision/10)
    #
    #
    #
    #
    #







