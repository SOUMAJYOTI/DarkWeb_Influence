import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
import sentence_feat as sent_svm
import sklearn
import random
import transductive_svm as tdsvm
from sklearn.model_selection import LeaveOneOut

import warnings
import os
warnings.filterwarnings("ignore")


def getFolds_LeaveOneOut(Y):
    loo = LeaveOneOut()
    train_folds = []
    test_folds = []
    for train_index, test_index in loo.split(Y):
        train_folds.append(train_index)
        test_folds.append(test_index)

    return train_folds, test_folds


def getFolds(Y):
    # X = np.zeros(Y.shape[0])

    train_folds = []
    test_folds = []
    kf = KFold(n_splits=5)
    cnt_folds = 0
    for train_index, test_index in kf.split(Y):
        train_folds.append(train_index)
        test_folds.append(test_index)

        cnt_folds += 1
        if cnt_folds >= 5:
            break

    return train_folds, test_folds


def main():
    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)

    trans_data = [4000]

    """" PHASE 1 DATA """
    """  Corpora words have dimension 200  """
    w2v_feat = pickle.load(open('../../../darkweb_data/3_25/word2vec_train_model_d200_min4.pickle', 'rb'))
    docs = pickle.load(open('../../../darkweb_data/4_23/docs_label.pickle', 'rb'))
    docs_unlabel = pickle.load(open('../../../darkweb_data/4_23/unlabeled_final/docs_0.pickle', 'rb'))

    Y_labels = np.array(forumsData.ix[:, 3:12])

    """ prepare the folds for CV test """
    train_fold, test_fold = getFolds(Y_labels)

    pickle.dump((train_fold, test_fold), open('train_test_folds.pickle', 'wb'))
    # train_fold, test_fold = getFolds_LeaveOneOut(Y_labels)

    for idx_trans in range(1):
        print("Amount of unlabeled data: ", trans_data[idx_trans])
        cnt_fold = 0

        for idx_fold in range(1): #range(len(train_fold)):
            print('Fold: ', idx_fold, '\n')

            map_test_indices = {}  # This is to keep track of the test indices
            for idx_indicator in range(len(test_fold[idx_fold])):
                map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator

            cnt_fold += 1

            for col_p1 in range(3, 12):
                # print("Col: ", col-3)

                """ PHASE 1 FEATURES"""
                X_inst, X_inst_3D,  X_u, Y_inst, row_indices = \
                    sent_svm.get_X_Y_data(forumsData, docs, docs_unlabel, w2v_feat, col_p1, trans_data[idx_trans])

                X_inst = np.array(X_inst)
                Y_inst = np.array(Y_inst)
                X_inst_3D = np.array(X_inst_3D)
                X_unlabel_3D = np.array(X_u)

                print("Total samples: l+u: ", X_inst.shape)

                # Segment the label and unlabel examples
                X_unlabel = X_inst[np.where(Y_inst == 2.)]
                X_label = X_inst[np.where(Y_inst != 2.)]
                Y_label = Y_inst[np.where(Y_inst != 2.)]
                X_label_3D = X_inst_3D[np.where(Y_inst != 2.)]

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
                X_train = X_label[train_indices]
                X_train_3D = X_label_3D[train_indices]
                Y_train = Y_label[train_indices]

                X_test_3D = X_label_3D[test_indices]
                Y_test = Y_label[test_indices]

                Y_train = Y_train.astype(int)
                Y_test = Y_test.astype(int)

                clf_tsvm = tdsvm.SKTSVM()
                # print("Sample shapes: trainX, trainY, trainU: ", X_train.shape, Y_train.shape, X_unlabel.shape)

                clf_tsvm.fit(X_train, Y_train, X_unlabel)
                pred_proba = clf_tsvm.predict_proba(X_unlabel)

                X_safeU = []
                Y_safeu = []
                """ Choose high confidence unlabeled examples """
                for idx_prob in range(pred_proba.shape[0]):
                    # print(pred_proba[idx_prob, 0], np.log(pred_proba[idx_prob, 0]),
                    #       np.log(pred_proba[idx_prob, 1]))
                    lab_inst = np.argmax(pred_proba[idx_prob, :])
                    if lab_inst == 0:
                        if pred_proba[idx_prob, 0] > .99:
                            X_safeU.append(X_unlabel_3D[idx_prob])
                            Y_safeu.append(-1)
                    else:
                        if pred_proba[idx_prob, 1] > .99:
                            X_safeU.append(X_unlabel_3D[idx_prob])
                            Y_safeu.append(1)

                """ Append all the labeled and unlabeled examples for input to CNN """
                X_safeU = np.array(X_safeU)
                Y_safeu = np.array(Y_safeu)

                X_train = np.concatenate((X_train_3D, X_safeU), axis=0)
                Y_train = np.concatenate((Y_train, Y_safeu), axis=0)

                print("Sample shapes: X_train, X_test: ", X_train.shape, X_test_3D.shape,  "\n")
                """ Write the samples to disk """
                output_dir = 'data/unlabeled_' + str(idx_trans) + '/fold_' + str(idx_fold) + '/col/' + str(col_p1-3) + '/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                pickle.dump(X_train, open(output_dir + 'X_train.pickle', 'wb'))
                pickle.dump(Y_train, open(output_dir + 'Y_train.pickle', 'wb'))
                pickle.dump(X_test_3D, open(output_dir + 'X_test.pickle', 'wb'))
                pickle.dump(Y_test, open(output_dir + 'Y_test.pickle', 'wb'))

                # print(X_train.shape, Y_train.shape)
                # exit()


if __name__ == "__main__":
    main()