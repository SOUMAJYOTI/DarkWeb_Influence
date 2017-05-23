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

        # cnt_folds += 1
        # if cnt_folds >= 5:
        #     break

    return train_folds, test_folds


# it gets the w2v for the phrases as well as words
# the sentence feature is the average of the word word2vec
def getFeatVectors(doc, w2v):
    doc_vec = []
    doc_mean_vec = []
    for d in range(len(doc)):
        sent_vec = []
        phrases = doc[d]
        phr_sorted = sorted(phrases.items(), key=operator.itemgetter(1), reverse=True)
        if len(phr_sorted) > 30:
            phr_sorted = phr_sorted[:30]

        for item in range(len(phr_sorted)):
            p, tv = phr_sorted[item]
            words = p.split('_')
            if len(words) >= 2:
                for w in words:
                    # if w in stopwords:
                    #     continue
                    if w in w2v:
                        sent_vec.append(w2v[w])

            else:
                if p in w2v:
                    sent_vec.append(w2v[p])

        if len(sent_vec) == 0:
            continue

        # Cases where the number of words is less than 30 for a sentence
        if len(sent_vec) > 30:
            sent_vec = sent_vec[:30]
        else:
            l = len(sent_vec)
            for idx_fill in range(30-l):
                sent_vec.append(np.random.uniform(-1, 1, (50))) # pad with random distribution

        doc_mean_vec.append(np.mean(sent_vec, axis=0))
        doc_vec.append(sent_vec)

    return np.array(doc_vec), np.array(doc_mean_vec)


# Schema for instances for recognition
# 1. - positive instance
# -1. - negative instance
def get_X_Y_data(forumsData, docs, w2v_feat, column):
    dict_doc_sent = {}  # hashmap for documnet sentence map
    dict_doc_sent_3D = {}

    positive_posts = []
    negative_posts = []

    positive_indices = []
    negative_indices = []

    positive_row_indices = [] # keep track of +ve instance number for enumerated sentence instances
    negative_row_indices = []  # keep track of -ve instance number for enumerated sentence instances
    row_instances = [] # rows of the filtered sentences
    for row in range(len(forumsData)):
        if len(docs[row]) == 0:
            act_v = list(np.zeros((1, 30, 50)))
            v = [list(np.random.uniform(-1, 1, (50)))]
        else:
            act_v, v = getFeatVectors(docs[row], w2v_feat)
            # print(np.array(act_v).shape, len(v))
        temp_idx = []
        row_indices = []
        for idx in range(len(v)):
            temp_idx.append((row, idx))
            row_indices.append(row)
            # print(np.array(act_v[idx]).shape)
            dict_doc_sent[(row, idx)] = v[idx]
            dict_doc_sent_3D[(row, idx)] = act_v[idx]

        # separate the posts into positive and negative posts
        if forumsData.ix[row, column] == 1.0:
            positive_posts.extend(v)
            positive_indices.extend(temp_idx)
            positive_row_indices.extend(row_indices)
        else:
            negative_posts.extend(v)
            negative_indices.extend(temp_idx)
            negative_row_indices.extend(row_indices)

    X_inst_2D = []
    X_inst_3D = []
    Y_inst = []

    """ Positive instances """
    # pos_tags = clusterDoc(positive_posts, pref=-30)
    cnt_pos = 0
    for idx in range(len(positive_row_indices)):
        if positive_row_indices[idx] not in row_instances:
            for pos_idx in range(50):
                try:
                    X_inst_2D.append(dict_doc_sent[(positive_row_indices[idx], pos_idx)])
                    X_inst_3D.append(dict_doc_sent_3D[(positive_row_indices[idx], pos_idx)])
                    row_instances.append(positive_row_indices[idx])
                    Y_inst.append(1.)
                    cnt_pos += 1
                except:
                    break

    # print(np.array(X_inst_2D).shape)
    """ Negative instances """
    cnt_neg = 0
    for idx in range(len(negative_row_indices)):
        if negative_row_indices[idx] not in row_instances:
            for neg_idx in range(50):
                try:
                    X_inst_2D.append(dict_doc_sent[(negative_row_indices[idx], neg_idx)])
                    X_inst_3D.append(dict_doc_sent_3D[(negative_row_indices[idx], neg_idx)])
                    row_instances.append(negative_row_indices[idx])
                    Y_inst.append(-1.)
                    cnt_neg += 1
                except:
                    break

    return X_inst_2D,  X_inst_3D, Y_inst, row_instances



def main():
    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)

    """" PHASE 1 DATA """
    """  Corpora words have dimension 50  """
    w2v_feat = pickle.load(open('../../../darkweb_data/5_10/word2vec_train_model_d50_min2.pickle', 'rb'))
    docs = pickle.load(open('../../../darkweb_data/5_10/docs_label.pickle', 'rb'))

    Y_labels = np.array(forumsData.ix[:, 3:12])

    """ prepare the folds for CV test """
    train_fold, test_fold = getFolds(Y_labels)

    # pickle.dump((train_fold, test_fold), open('train_test_folds.pickle', 'wb'))
    # train_fold, test_fold = getFolds_LeaveOneOut(Y_labels)

    cnt_fold = 0

    for idx_fold in range(0, 1): #range(len(train_fold)):
        print('Fold: ', idx_fold, '\n')

        map_test_indices = {}  # This is to keep track of the test indices
        for idx_indicator in range(len(test_fold[idx_fold])):
            map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator

        cnt_fold += 1

        for col_p1 in range(3, 12):
            print("Column: ", str(col_p1))
            # print("Col: ", col-3)

            X_inst, X_inst_3D,  Y_inst, row_indices = \
                get_X_Y_data(forumsData, docs, w2v_feat, col_p1)

        #     train_indices = []
        #     test_indices = []
        #     for inst_index in range(len(train_fold[idx_fold])):
        #         train_indices.extend(list(np.where(row_indices == train_fold[idx_fold][inst_index])[0]))
        #     for inst_index in range(len(test_fold[idx_fold])):
        #         test_indices.extend(list(np.where(row_indices == test_fold[idx_fold][inst_index])[0]))
        #
        #     train_indices = np.array(train_indices)
        #     test_indices = np.array(test_indices)
        #
        #     X_inst = np.array(X_inst)
        #     Y_inst = np.array(Y_inst)
        #     X_inst_3D = np.array(X_inst_3D)
        #     X_unlabel_3D = np.array(X_u)
        #
        #     # print("Total samples: l+u: ", X_inst.shape)
        #
        #     # Segment the label and unlabel examples
        #     X_unlabel = X_inst[np.where(Y_inst == 2.)]
        #     X_label = X_inst[np.where(Y_inst != 2.)]
        #     Y_label = Y_inst[np.where(Y_inst != 2.)]
        #     X_label_3D = X_inst_3D[np.where(Y_inst != 2.)]
        #     # print(X_label_3D.shape)
        #
        #     """ SET THE INSTANCES FOR THIS COLUMN"""
        #     X_train = X_label[train_indices]
        #     X_train_3D = X_label_3D[train_indices]
        #     Y_train = Y_label[train_indices]
        #
        #     X_test_3D = X_label_3D[test_indices]
        #     Y_test = Y_label[test_indices]
        #
        #     Y_train = Y_train.astype(int)
        #     Y_test = Y_test.astype(int)
        #
        #     print(Y_test[Y_test == 1].shape[0] / Y_test.shape[0])
        #     # print(Y_train[Y_train == 1].shape[0] / Y_train.shape[0])
        #     # exit()
        #
        #     """
        #     Subsmaple the negative samples to balance the
        #     dataset for input to TSVM.
        #     """
        #     X_train_pos_3D = X_train_3D[Y_train == 1]
        #     X_train_neg_3D = X_train_3D[Y_train == -1]
        #     X_train_pos = X_train[Y_train == 1]
        #     X_train_neg = X_train[Y_train == -1]
        #     if X_train_pos_3D.shape[0] < X_train_neg_3D.shape[0]:
        #         X_train_neg_3D = X_train_neg_3D[:X_train_pos_3D.shape[0]]
        #         X_train_neg = X_train_neg[:X_train_pos.shape[0]]
        #     else:
        #         X_train_pos_3D = X_train_pos_3D[:X_train_neg_3D.shape[0]]
        #         X_train_pos = X_train_pos[:X_train_neg.shape[0]]
        #
        #     X_train_final_3D = np.concatenate((X_train_neg_3D, X_train_pos_3D), axis=0)
        #     Y_train_final_3D = np.array([-1] * X_train_neg_3D.shape[0] + [1] * X_train_pos_3D.shape[0]) # TODO: this was wrong check it
        #     X_train_inp = np.concatenate((X_train_neg, X_train_pos), axis=0)
        #     Y_train_inp = np.array([-1]*X_train_neg.shape[0] + [1]*X_train_pos.shape[0]) # TODO: this was wrong check it
        #
        #     # print(Y_train_inp[Y_train_inp == 1].shape[0] / Y_train_inp.shape[0])
        #     clf_tsvm = tdsvm.SKTSVM()
        #     # print("Sample shapes: trainX, trainY, trainU: ", X_train.shape, Y_train.shape, X_unlabel.shape)
        #
        #     # print(Y_train)
        #     try:
        #         clf_tsvm.fit(X_train_inp, Y_train_inp, X_unlabel)
        #     except:
        #         continue
        #     pred_proba = clf_tsvm.predict_proba(X_unlabel)
        #
        #     X_safeU = []
        #     Y_safeu = []
        #     # """ Choose high confidence unlabeled examples """
        #     for idx_prob in range(pred_proba.shape[0]):
        #         lab_inst = np.argmax(pred_proba[idx_prob, :])
        #         if lab_inst == 0:
        #             if pred_proba[idx_prob, 0] > .98:
        #                 X_safeU.append(X_unlabel_3D[idx_prob])
        #                 Y_safeu.append(-1)
        #         else:
        #             if pred_proba[idx_prob, 1] > .98:
        #                 X_safeU.append(X_unlabel_3D[idx_prob])
        #                 Y_safeu.append(1)
        #
        #     # """ Append all the labeled and unlabeled examples for input to CNN """
        #     X_safeU = np.array(X_safeU)
        #     Y_safeu = np.array(Y_safeu)
        #
        #     # print(X_safeU.shape)
        #     # """ Subsample the unlabeled examples for balanced training set"""
        #     X_safeU_pos = X_safeU[Y_safeu == 1]
        #     X_safeU_neg = X_safeU[Y_safeu == -1]
        #
        #     if X_safeU_pos.shape[0] > X_safeU_neg.shape[0]:
        #         X_safeU_pos = X_safeU_pos[:X_safeU_neg.shape[0]]
        #     else:
        #         X_safeU_neg = X_safeU_neg[:X_safeU_pos.shape[0]]
        #     X_safeU = np.concatenate((X_safeU_pos, X_safeU_neg), axis=0)
        #     Y_safeu = np.array([1] * X_safeU_pos.shape[0] + [-1]*X_safeU_neg.shape[0])
        #
        #     # Add the labeled data only once
        #     if idx_trans == 0:
        #         X_train_final = X_train_final_3D
        #         Y_train_final = Y_train_final_3D
        #
        #     if X_safeU.shape[0] > 0:
        #         X_train_final = np.concatenate((X_train_final, X_safeU), axis=0)
        #         Y_train_final = np.concatenate((Y_train_final, Y_safeu), axis=0)
        #
        #     # print(Y_train_final[Y_train_final == 1].shape[0] / Y_train_final.shape[0])
        #
        # print(Y_train_final[Y_train_final == 1].shape[0] / Y_train_final.shape[0])
        # print(Y_train_final_3D[Y_train_final_3D == 1].shape[0] / Y_train_final_3D.shape[0])
        #
        # # """ Write the samples to disk """
        # output_dir = 'data/05_02/fold_' + str(idx_fold) + '/col/' + str(col_p1-3) + '/'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # pickle.dump(X_train_final_3D, open(output_dir + 'X_train_l.pickle', 'wb'))
        # pickle.dump(Y_train_final_3D, open(output_dir + 'Y_train_l.pickle', 'wb'))
        # pickle.dump(X_train_final, open(output_dir + 'X_train_l+u.pickle', 'wb'))
        # pickle.dump(Y_train_final, open(output_dir + 'Y_train_l+u.pickle', 'wb'))
        # pickle.dump(X_test_3D, open(output_dir + 'X_test.pickle', 'wb'))
        # pickle.dump(Y_test, open(output_dir + 'Y_test.pickle', 'wb'))

        # print(X_train.shape, Y_train.shape)
        # exit()


if __name__ == "__main__":
    main()