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

        cnt_folds += 1
        if cnt_folds >= 5:
            break

    return train_folds, test_folds


# Schema for instances for recognition
# 1. - positive instance
# 0.. - negative instance
# Maximum 10 words for each sentence
def get_X_Y_data(docs, labels, w2v_feat, col):
    row_instances = [] # rows of the filtered sentences
    vocab_dict = {}
    X_ind = []
    X_words = []
    X_w2v = []
    Y = []
    count_words = 1
    for row in range(len(docs)):
        l = labels[row][col]
        for idx_sent in range(len(docs[row])):
            sent = docs[row][idx_sent]
            words = sent.split(' ')
            sent_index = []
            for w in range(len(words)):
                if words[w] not in vocab_dict:
                    vocab_dict[words[w]] = count_words
                    count_words += 1
                sent_index.append(vocab_dict[words[w]])

            if len(sent_index) > 10:
                sent_index = sent_index[:10]
                words = words[:10]
            else:
                len_cur = len(sent_index)
                for idx_fill in range(10 - len_cur):
                    sent_index.append(0) # 0 --> DUMMY
                    words.append('')

            X_ind.append(sent_index)
            X_words.append(words)

            context_input = []
            for w in range(len(words)):
                if words[w] in w2v_feat:
                    context_input.append(w2v_feat[words[w]])
                else:
                    context_input.append(np.random.uniform(-0.25, 0.25, (50)))

            # Sentences should be of length 15
            if len(context_input) < 10:
                len_cur = len(context_input)
                for idx_fill in range(10 - len_cur):
                    context_input.append(np.random.uniform(-0.25, 0.25, (50)))
            else:
                context_input = context_input[:10]

            X_w2v.append(context_input)

            if l == 1.:
                Y.append([0, 1])
            else:
                Y.append([1, 0])

            row_instances.append(row)

    return np.array(X_ind),  np.array(X_words), np.array(X_w2v), np.array(Y), row_instances, vocab_dict


def main():
    # trans_data = [2000]

    """" PHASE 1 DATA """
    """  Corpora words have dimension 50  """
    w2v_feat = pickle.load(open('../../darkweb_data/5_10/word2vec_train_model_d50_min2.pickle', 'rb'))
    docs = pickle.load(open('../../darkweb_data/5_10/forum_40_label_input_docs.pickle', 'rb'))
    labels = pickle.load(open('../../darkweb_data/5_10/forum_40_input_labels.pickle', 'rb'))

    Y_labels = np.array(labels)
    print(Y_labels.shape)

    """ prepare the folds for CV test """
    train_fold, test_fold = getFolds(Y_labels)

    # pickle.dump((train_fold, test_fold), open('train_test_folds.pickle', 'wb'))
    # train_fold, test_fold = getFolds_LeaveOneOut(Y_labels)

    cnt_fold = 0
        # print(X_ind.shape, X_words.shape, X_w2v.shape, Y.shape)
    for idx_fold in range(0, 5): #range(len(train_fold)):
        print('Fold: ', idx_fold, '\n')
        cnt_fold += 1

        map_test_indices = {}  # This is to keep track of the test indices
        for idx_indicator in range(len(test_fold[idx_fold])):
            map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator

        for col in range(9):
            X_ind, X_words, X_w2v, Y, row_indices, vocab_dict = \
                get_X_Y_data(docs, Y_labels, w2v_feat, col)

            train_indices = []
            test_indices = []
            for inst_index in range(len(train_fold[idx_fold])):
                train_indices.extend(list(np.where(row_indices == train_fold[idx_fold][inst_index])[0]))
            for inst_index in range(len(test_fold[idx_fold])):
                test_indices.extend(list(np.where(row_indices == test_fold[idx_fold][inst_index])[0]))

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            """ SET THE INSTANCES FOR THIS COLUMN"""
            X_train = X_ind[train_indices]
            Y_train = Y[train_indices]

            X_test = X_ind[test_indices]
            Y_test = Y[test_indices]

            """
            Subsmaple the negative samples to balance the
            dataset for the training data
            """
            X_train_pos = []
            X_train_neg = []
            for idx in range(X_train.shape[0]):
                if np.array_equal(Y_train[idx], [0, 1]):
                    X_train_pos.append(X_train[idx])
                else:
                    X_train_neg.append(X_train[idx])

            X_train_pos = np.array(X_train_pos)
            X_train_neg = np.array(X_train_neg)

            if X_train_pos.shape[0] < X_train_neg.shape[0]:
                X_train_neg = X_train_neg[:X_train_pos.shape[0]]
            else:
                X_train_pos = X_train_pos[:X_train_neg.shape[0]]

            X_train_final = np.concatenate((X_train_neg, X_train_pos), axis=0)
            Y_train_final = np.array([[1, 0]] * X_train_neg.shape[0] + [[0, 1]] * X_train_pos.shape[0])

            # print(X_train_final.shape, Y_train_final.shape, X_test.shape, Y_test.shape)

            # # """ Write the samples to disk """
            output_dir = '../../darkweb_data/5_10/data/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            pickle.dump(X_train_final, open(output_dir + 'X_train_l.pickle', 'wb'))
            pickle.dump(Y_train_final, open(output_dir + 'Y_train_l.pickle', 'wb'))
            pickle.dump(X_test, open(output_dir + 'X_test.pickle', 'wb'))
            pickle.dump(Y_test, open(output_dir + 'Y_test.pickle', 'wb'))
            pickle.dump(vocab_dict, open('../../darkweb_data/5_10/data/vocab_dict.pickle', 'wb'))



if __name__ == "__main__":
    main()