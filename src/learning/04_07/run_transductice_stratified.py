import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
import words_sentences_svm as sent_svm
import sklearn
import random
import matplotlib.pyplot as plt
import inter_label_svm as int_svm
# import SVM_cvxopt as svopt
# import eval as eval_met
# from imblearn.over_sampling import SMOTE
# import itertools
# import cvxopt
import transductive_svm as tdsvm
from sklearn.model_selection import LeaveOneOut
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


def getFolds_LeaveOneOut(Y):
    loo = LeaveOneOut()
    train_folds = []
    test_folds = []
    for train_index, test_index in loo.split(Y):
        train_folds.append(train_index)
        test_folds.append(test_index)

    return train_folds, test_folds

def getFolds(Y):
    print(Y)
    X = np.zeros(Y.shape[0])
    train_folds = []
    test_folds = []
    kf = StratifiedKFold(n_splits=10)
    for train_index, test_index in kf.split(X, Y):
        train_folds.append(train_index)
        test_folds.append(test_index)

    return train_folds, test_folds


def main():
    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)
    cnt_fold = 0

    """ EVALUATION STRUCTURES  """
    avg_label_precision = [0. for _ in range(9)]
    avg_label_random = [0. for _ in range(9)]
    avg_label_recall = [0. for _ in range(9)]
    avg_label_F1 = [0. for _ in range(9)]

    avg_label_precision_3 = [0. for _ in range(9)]
    avg_label_recall_3 = [0. for _ in range(9)]
    avg_label_F1_3 = [0. for _ in range(9)]
    Y_actual = [[] for _ in range(1)]

    """" PHASE 1 DATA """
    """  Corpora words have dimension 100  """
    w2v_feat = pickle.load(open('../../../darkweb_data/3_25/word2vec_train_model_d100.pickle', 'rb'))
    docs = pickle.load(open('../../../darkweb_data/4_8/docs_corpora_label_tfidf.pickle', 'rb'))
    docs_unlabel = pickle.load(open('../../../darkweb_data/4_8/docs_corpora_unlabel_tfidf.pickle', 'rb'))

    Y_labels = np.array(forumsData.ix[:, 3:12])
    for idx in range(Y_labels.shape[0]):
        for idx_1 in range(Y_labels.shape[1]):
            if Y_labels[idx, idx_1] == 0.:
                Y_labels[idx, idx_1] = -1.
            else:
                Y_labels[idx, idx_1] = 1.

    corr = int_svm.labelCorrMatrix(np.array(Y_labels))  # CORRELATION MATRIX

    # train_fold, test_fold = getFolds_LeaveOneOut(Y_labels)

    """ SVM Classifier used """
    # clf = sklearn.svm.SVC(kernel='rbf', C=1000.1)
    clf = sklearn.linear_model.LogisticRegression()

    for col_p1 in range(3, 12):
        """ prepare the folds for CV test """
        train_fold, test_fold = getFolds(Y_labels[:, col_p1-3])

        X_inst_1, Y_inst_1, row_indices, pos_perc, neg_perc = \
            sent_svm.get_X_Y_data(forumsData, docs, docs_unlabel, w2v_feat, col_p1)

        X_inst_1 = np.array(X_inst_1)
        Y_inst_1 = np.array(Y_inst_1)

        X_unlabel = X_inst_1[np.where(Y_inst_1 == 2.)]
        X_inst_1 = X_inst_1[np.where(Y_inst_1 != 2.)]
        Y_inst_1 = Y_inst_1[Y_inst_1 != 2.]
        X_inst_1 = np.array(X_inst_1)
        Y_inst_1 = np.array(Y_inst_1)
        X_unlabel = np.array(X_unlabel)

        row_indices = np.array(row_indices)


    for idx_fold in range(len(train_fold)):
        print('Fold: ', idx_fold, '\n')

        Y_true = Y_labels[test_fold[idx_fold]]  # TRUE LABELS
        Y_predict_final = -np.ones(Y_true.shape)

        print("Phase 1....")
        map_test_indices = {}  # This is to keep track of the test indices
        for idx_indicator in range(len(test_fold[idx_fold])):
            map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator

        cnt_fold += 1
        Y_predict_1_out = -np.ones(Y_true.shape)

        sm = SMOTE(random_state=42)
        X_res, Y_res = sm.fit_sample(X_train_1, Y_train_1)

        for col_p1 in range(3, 4):
            # print("Col: ", col-3)
            """ PHASE 1 FEATURES"""
            X_inst_1, Y_inst_1, row_indices, pos_perc, neg_perc = \
                sent_svm.get_X_Y_data(forumsData, docs, docs_unlabel, w2v_feat, col_p1)

            X_inst_1 = np.array(X_inst_1)
            Y_inst_1 = np.array(Y_inst_1)

            X_unlabel = X_inst_1[np.where(Y_inst_1 == 2.)]
            X_inst_1 = X_inst_1[np.where(Y_inst_1 != 2.)]
            Y_inst_1 = Y_inst_1[Y_inst_1 != 2.]
            X_inst_1 = np.array(X_inst_1)
            Y_inst_1 = np.array(Y_inst_1)
            X_unlabel = np.array(X_unlabel)

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

            # Y_unlabel = -np.ones(X_unlabel.shape)
            # Y_train_1[Y_train_1 == -1.] = 0.
            X_test_1 = X_inst_1[test_indices]
            Y_test_1 = Y_inst_1[test_indices]  # TODO: NOT USED !!!

            Y_train_1 = Y_train_1.astype(int)
            Y_test_1 = Y_test_1.astype(int)

            sm = SMOTE(random_state=42)
            X_res, Y_res = sm.fit_sample(X_train_1, Y_train_1)

            # Y_actual[col_p1-3].append(Y_test_1[0])


            """ PHASE 1 RESULTS"""
            clf_tsvm = tdsvm.SKTSVM()
            print(X_train_1.shape, Y_train_1.shape)
            print(X_res.shape, Y_res.shape)
            # print(X_test_1.shape)

            Y_res = Y_res.astype(int)

            clf_tsvm.fit(X_res, Y_res, X_unlabel)
            clf.fit(X_train_1, Y_train_1)

            # unl_labels = clf_tsvm.predict(X_unlabel)
            #
            # cnt_pos = 0
            # for inst_ul in range(unl_labels.shape[0]):
            #     if unl_labels[inst_ul] == 1.:
            #         cnt_pos += 1

            Y_traditional = clf.predict(X_test_1)

            Y_random = []
            for idx_r in range(X_test_1.shape[0]):
                Y_random.append(random.sample([-1, 1], 1)[0])
            Y_random = np.array(Y_random).astype(int)

            avg_label_random[col_p1 - 3] += sklearn.metrics.f1_score(Y_test_1, Y_random)

            try:
                avg_label_precision[col_p1-3] += sklearn.metrics.f1_score(Y_test_1, Y_traditional)
            except:
                avg_label_precision[col_p1-2] += sklearn.metrics.f1_score(Y_test_1, Y_random)
            # avg_label_precision[col_p1-3].append(Y_traditional[0])


            # avg_label_random[col_p1-3].append(Y_random[0])

            Y_predict_1 = clf_tsvm.predict(X_test_1)
            Y_predict_1 = Y_predict_1.astype(int)
            try:
                avg_label_precision_3[col_p1-3] += sklearn.metrics.f1_score(Y_test_1, Y_predict_1)
            except:
                avg_label_precision_3[col_p1 - 3] += sklearn.metrics.f1_score(Y_test_1, Y_random)

            print(sklearn.metrics.f1_score(Y_test_1, Y_traditional), sklearn.metrics.f1_score(Y_test_1, Y_predict_1), sklearn.metrics.f1_score(Y_test_1, Y_random))
            if sklearn.metrics.f1_score(Y_test_1, Y_predict_1)< 0.3:
                print(Y_test_1, Y_predict_1)

            # avg_label_precision_3[col_p1 - 3].append(Y_predict_1[0])

    for col in range(len(avg_label_precision)):
        # print(sklearn.metrics.f1_score(Y_actual[col], avg_label_precision_3[col]))
        # print(sklearn.metrics.f1_score(Y_actual[col], avg_label_precision[col]))
        # print(sklearn.metrics.f1_score(Y_actual[col], avg_label_random[col]))
        avg_label_precision_3[col] /= len(train_fold)
        avg_label_precision[col] /= len(train_fold)
        avg_label_random[col] /= len(train_fold)

    print(avg_label_precision_3,
          avg_label_precision, avg_label_random)

if __name__ == "__main__":
    main()