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

    # trans_data = [2000]

    """" PHASE 1 DATA """
    """  Corpora words have dimension 50  """
    w2v_feat = pickle.load(open('../../../darkweb_data/4_30/word2vec_train_model_d50_min3.pickle', 'rb'))
    docs = pickle.load(open('../../../darkweb_data/4_30/docs_label.pickle', 'rb'))

    Y_labels = np.array(forumsData.ix[:, 3:12])

    """ prepare the folds for CV test """
    train_fold, test_fold = getFolds(Y_labels)

    # pickle.dump((train_fold, test_fold), open('train_test_folds.pickle', 'wb'))
    # train_fold, test_fold = getFolds_LeaveOneOut(Y_labels)

    cnt_fold = 0

    for idx_fold in range(0, 2): #range(len(train_fold)):
        print('Fold: ', idx_fold, '\n')

        map_test_indices = {}  # This is to keep track of the test indices
        for idx_indicator in range(len(test_fold[idx_fold])):
            map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator

        cnt_fold += 1

        for col_p1 in range(3, 12):
            print("Column: ", str(col_p1))
            # print("Col: ", col-3)

            # Iterate over the unlabeled partitioned datasets
            for idx_trans in range(11):
                """ PHASE 1 FEATURES"""
                docs_unlabel = pickle.load(open('../../../darkweb_data/4_30/unlabeled_final/docs_'
                                                + str(idx_trans) + '.pickle', 'rb'))
                X_inst, X_inst_3D,  X_u, Y_inst, row_indices = \
                    sent_svm.get_X_Y_data(forumsData, docs, docs_unlabel, w2v_feat, col_p1)

                train_indices = []
                test_indices = []
                for inst_index in range(len(train_fold[idx_fold])):
                    train_indices.extend(list(np.where(row_indices == train_fold[idx_fold][inst_index])[0]))
                for inst_index in range(len(test_fold[idx_fold])):
                    test_indices.extend(list(np.where(row_indices == test_fold[idx_fold][inst_index])[0]))

                train_indices = np.array(train_indices)
                test_indices = np.array(test_indices)

                X_inst = np.array(X_inst)
                Y_inst = np.array(Y_inst)
                X_inst_3D = np.array(X_inst_3D)
                X_unlabel_3D = np.array(X_u)

                # print("Total samples: l+u: ", X_inst.shape)

                # Segment the label and unlabel examples
                X_unlabel = X_inst[np.where(Y_inst == 2.)]
                X_label = X_inst[np.where(Y_inst != 2.)]
                Y_label = Y_inst[np.where(Y_inst != 2.)]
                X_label_3D = X_inst_3D[np.where(Y_inst != 2.)]
                # print(X_label_3D.shape)

                """ SET THE INSTANCES FOR THIS COLUMN"""
                X_train = X_label[train_indices]
                X_train_3D = X_label_3D[train_indices]
                Y_train = Y_label[train_indices]

                X_test_3D = X_label_3D[test_indices]
                Y_test = Y_label[test_indices]

                Y_train = Y_train.astype(int)
                Y_test = Y_test.astype(int)

                print(Y_test[Y_test == 1].shape[0] / Y_test.shape[0])
                # print(Y_train[Y_train == 1].shape[0] / Y_train.shape[0])
                # exit()

                """
                Subsmaple the negative samples to balance the
                dataset for input to TSVM.
                """
                X_train_pos_3D = X_train_3D[Y_train == 1]
                X_train_neg_3D = X_train_3D[Y_train == -1]
                X_train_pos = X_train[Y_train == 1]
                X_train_neg = X_train[Y_train == -1]
                if X_train_pos_3D.shape[0] < X_train_neg_3D.shape[0]:
                    X_train_neg_3D = X_train_neg_3D[:X_train_pos_3D.shape[0]]
                    X_train_neg = X_train_neg[:X_train_pos.shape[0]]
                else:
                    X_train_pos_3D = X_train_pos_3D[:X_train_neg_3D.shape[0]]
                    X_train_pos = X_train_pos[:X_train_neg.shape[0]]

                X_train_final_3D = np.concatenate((X_train_neg_3D, X_train_pos_3D), axis=0)
                Y_train_final_3D = np.array([-1] * X_train_neg_3D.shape[0] + [1] * X_train_pos_3D.shape[0]) # TODO: this was wrong check it
                X_train_inp = np.concatenate((X_train_neg, X_train_pos), axis=0)
                Y_train_inp = np.array([-1]*X_train_neg.shape[0] + [1]*X_train_pos.shape[0]) # TODO: this was wrong check it

                # print(Y_train_inp[Y_train_inp == 1].shape[0] / Y_train_inp.shape[0])
                clf_tsvm = tdsvm.SKTSVM()
                # print("Sample shapes: trainX, trainY, trainU: ", X_train.shape, Y_train.shape, X_unlabel.shape)

                # print(Y_train)
                try:
                    clf_tsvm.fit(X_train_inp, Y_train_inp, X_unlabel)
                except:
                    continue
                pred_proba = clf_tsvm.predict_proba(X_unlabel)

                X_safeU = []
                Y_safeu = []
                # """ Choose high confidence unlabeled examples """
                for idx_prob in range(pred_proba.shape[0]):
                    lab_inst = np.argmax(pred_proba[idx_prob, :])
                    if lab_inst == 0:
                        if pred_proba[idx_prob, 0] > .98:
                            X_safeU.append(X_unlabel_3D[idx_prob])
                            Y_safeu.append(-1)
                    else:
                        if pred_proba[idx_prob, 1] > .98:
                            X_safeU.append(X_unlabel_3D[idx_prob])
                            Y_safeu.append(1)

                # """ Append all the labeled and unlabeled examples for input to CNN """
                X_safeU = np.array(X_safeU)
                Y_safeu = np.array(Y_safeu)

                # print(X_safeU.shape)
                # """ Subsample the unlabeled examples for balanced training set"""
                X_safeU_pos = X_safeU[Y_safeu == 1]
                X_safeU_neg = X_safeU[Y_safeu == -1]

                if X_safeU_pos.shape[0] > X_safeU_neg.shape[0]:
                    X_safeU_pos = X_safeU_pos[:X_safeU_neg.shape[0]]
                else:
                    X_safeU_neg = X_safeU_neg[:X_safeU_pos.shape[0]]
                X_safeU = np.concatenate((X_safeU_pos, X_safeU_neg), axis=0)
                Y_safeu = np.array([1] * X_safeU_pos.shape[0] + [-1]*X_safeU_neg.shape[0])

                # Add the labeled data only once
                if idx_trans == 0:
                    X_train_final = X_train_final_3D
                    Y_train_final = Y_train_final_3D

                if X_safeU.shape[0] > 0:
                    X_train_final = np.concatenate((X_train_final, X_safeU), axis=0)
                    Y_train_final = np.concatenate((Y_train_final, Y_safeu), axis=0)

                # print(Y_train_final[Y_train_final == 1].shape[0] / Y_train_final.shape[0])

            print(Y_train_final[Y_train_final == 1].shape[0] / Y_train_final.shape[0])
            print(Y_train_final_3D[Y_train_final_3D == 1].shape[0] / Y_train_final_3D.shape[0])

            # """ Write the samples to disk """
            output_dir = 'data/05_02/fold_' + str(idx_fold) + '/col/' + str(col_p1-3) + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pickle.dump(X_train_final_3D, open(output_dir + 'X_train_l.pickle', 'wb'))
            pickle.dump(Y_train_final_3D, open(output_dir + 'Y_train_l.pickle', 'wb'))
            pickle.dump(X_train_final, open(output_dir + 'X_train_l+u.pickle', 'wb'))
            pickle.dump(Y_train_final, open(output_dir + 'Y_train_l+u.pickle', 'wb'))
            pickle.dump(X_test_3D, open(output_dir + 'X_test.pickle', 'wb'))
            pickle.dump(Y_test, open(output_dir + 'Y_test.pickle', 'wb'))

            # print(X_train.shape, Y_train.shape)
            # exit()


if __name__ == "__main__":
    main()