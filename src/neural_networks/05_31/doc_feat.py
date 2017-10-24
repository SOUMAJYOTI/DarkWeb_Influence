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

warnings.filterwarnings("ignore")


def getStopWords(data):
    for line in data:
        words = line.split(' ')
    # print(len(words))
    return words


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


def calculateTfIdf(docs):
    bloblist = []
    for d in range(len(docs)):
        doc = docs[d]
        doc = doc[:len(doc)]
        doc_str = ''
        for w in range(len(doc)):
            doc_str += doc[w] + ' '
        doc_str = doc_str[:len(doc_str)-1]
        bloblist.append(tb(doc_str))

    ScoresDoc = [{} for _ in range(len(docs))]
    for i, blob in enumerate(bloblist):
        # print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            ScoresDoc[i][word] = score
            # print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))

    return ScoresDoc


# Schema for instances for recognition
# [0, 1] - positive instance
# [1, 0] - negative instance
# Maximum 10 words for each sentence
def get_X_Y_data(docs, labels, d2v_feat, col, sen_length, vocab_dict, stop, tfidf_score):
    row_instances = [] # rows of the filtered sentences
    # vocab_dict = {}
    X_ind = [] # vocab index for words in sentences
    X_words = [] # words for sentences
    X_d2v = [] # w2v features for sentences
    X_tfidf = []
    Y = []
    Y_all = []

    count_words = len(vocab_dict)+1
    for row in range(len(docs)):
        if len(docs) > 5:
            random_ind = [random.randint(0, len(docs)) for _ in range(5)]

        doc_vec = d2v_feat.infer_vector(docs, alpha=0.025, min_alpha=0.0001)
        tfidf_doc = tfidf_score[row]
        l = labels[row]
        for idx_sent in range(len(docs[row])):
            sent = docs[row][idx_sent]
            words = sent.split(' ')

            tfidf_vals = []
            sent_index = []
            words_filter = []
            for w in range(len(words)):
                if words[w] in stop:
                    continue
                words_filter.append(words[w])
                if words[w] not in vocab_dict:
                    vocab_dict[words[w]] = count_words
                    count_words += 1
                if words[w] in tfidf_doc:
                    tfidf_vals.append(tfidf_doc[words[w]])
                sent_index.append(vocab_dict[words[w]])

            if len(docs[row]) > 1 and len(sent_index) < 3:
                continue

            # if len(sent_index) > sen_length:
            #     sent_index = sent_index[:sen_length]
            #     words_filter = words_filter[:sen_length]
            # else:
            #     len_cur = len(sent_index)
            #     for idx_fill in range(sen_length - len_cur):
            #         sent_index.append(0) # 0 --> DUMMY
            #         words_filter.append('') # '' (empty string) --> DUMMY

            sorted_tfidf = sorted(tfidf_vals, reverse=True)[:10]
            len_sent = len(sorted_tfidf)
            for idx_pad in range(10-len_sent):
                sorted_tfidf.append(0.)

            print(words_filter)
            X_tfidf.append(sorted_tfidf)
            X_ind.append(sent_index)
            X_words.append(words_filter)

            # for w2v feat
            context_input = []
            for w in range(len(words)):
                if words[w] in stop:
                    continue
                if words[w] in w2v_feat:
                    context_input.append(w2v_feat[words[w]])
                else:
                    context_input.append(np.random.uniform(-0.25, 0.25, (50)))

            # Sentences should be of length 15
            # if len(context_input) < sen_length:
            #     len_cur = len(context_input)
            #     for idx_fill in range(sen_length - len_cur):
            #         context_input.append(np.random.uniform(-0.25, 0.25, (50)))
            # else:
            #     context_input = context_input[:sen_length]

            # print(np.mean(np.array(context_input), axis=0))
            X_w2v.append(np.mean(np.array(context_input), axis=0)) # for SVM - mean of words
            # X_w2v.append(context_input)

            if l[col] == 1.:
                Y.append([0, 1])
            else:
                Y.append([1, 0])

            Y_all.append(l)
            row_instances.append(row)

    return np.array(X_ind),  np.array(X_words), np.array(X_tfidf),\
           np.array(X_w2v), np.array(Y), np.array(Y_all), \
           row_instances, vocab_dict


def clusterDoc(featDocs, Y_all, pref):

    # Compute Affinity Propagation
    af = AffinityPropagation(preference=pref).fit(featDocs)
    labels = af.labels_

    # max_occur = max(set(labels), key=labels.count)
    max_label = scst.mode(labels, axis=None)[0][0]
    # print(labels)

    cluster_feat = []
    Y_all_new = []
    for l in range(len(featDocs)):
        if labels[l] == max_label:
            cluster_feat.append(featDocs[l])
            Y_all_new.append(Y_all[l])
    return np.array(cluster_feat), np.array(Y_all_new)


def clusterDBSCAN(featDocs):
    db = DBSCAN(eps=1, min_samples=10).fit(featDocs)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # max_occur = max(set(labels), key=labels.count)
    max_label = scst.mode(labels, axis=None)[0][0]
    print(max_label)

    cluster_feat = []
    for l in range(len(featDocs)):
        if labels[l] == max_label:
            cluster_feat.append(featDocs[l])
    return np.array(cluster_feat)


def main():
    # trans_data = [2000]

    stopwords_file = open('../../../darkweb_data/Stop_Words.txt', 'r')
    stopwords = getStopWords(stopwords_file)
    vocab_dict = pickle.load(open('../../../darkweb_data/05/5_15/vocab_dict.pickle', 'rb'))

    """" PHASE 1 DATA """
    """  Corpora words have dimension 50  """
    w2v_feat = pickle.load(open('../../../darkweb_data/05/5_15/word2vec_train_model_d50_min2.pickle', 'rb'))
    d2v_feat = pickle.load(open('../../../darkweb_data/05/5_31/doc2vec_sent_d50.pickle', 'rb'))
    docs = pickle.load(open('../../../darkweb_data/05/5_15/forum_40_label_input_docs.pickle', 'rb'))
    labels = pickle.load(open('../../../darkweb_data/05/5_15/forum_40_input_labels.pickle', 'rb'))

    # tfidf_scores = calculateTfIdf(docs)
    # pickle.dump(tfidf_scores, open('../../../darkweb_data/05/5_19/tfidf_f40_labels.pickle', 'wb'))
    tfidfScore = pickle.load(open('../../../darkweb_data/05/5_19/tfidf_f40_labels.pickle', 'rb'))

    Y_labels = np.array(labels)
    print(Y_labels.shape)

    """ prepare the folds for CV test """
    train_fold, test_fold = getFolds(Y_labels)

    # pickle.dump((train_fold, test_fold), open('train_test_folds.pickle', 'wb'))
    # train_fold, test_fold = getFolds_LeaveOneOut(Y_labels)

    cnt_fold = 0
    random_f1 = [0. for _ in range(Y_labels.shape[1])]
        # print(X_ind.shape, X_words.shape, X_w2v.shape, Y.shape)
    for idx_fold in range(0, 1): #len(train_fold)): #range(len(train_fold)):
        cnt_fold += 1

        for col in range(2, 3):
            print('Fold: {}, Col: {}'.format(idx_fold, col), '\n')
            X_ind, X_words, X_tfidf, X_w2v, Y, Y_all, row_indices, vocab_dict = \
                get_X_Y_data(docs, Y_labels, w2v_feat, col-2, 30, vocab_dict, stopwords, tfidfScore)


            # TODO: CHECK THIS PART - IT LOOKS CORRECT !!!!
            train_indices = []
            test_indices = []
            for inst_index in range(len(train_fold[idx_fold])):
                train_indices.extend(list(np.where(row_indices == train_fold[idx_fold][inst_index])[0]))
            for inst_index in range(len(test_fold[idx_fold])):
                test_indices.extend(list(np.where(row_indices == test_fold[idx_fold][inst_index])[0]))

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            map_test_indices = {}  # This is to keep track of the test indices
            for idx_indicator in range(len(test_fold[idx_fold])):
                map_test_indices[test_fold[idx_fold][idx_indicator]] = idx_indicator
            #
            X_count_words = []
            for idx_c in range(X_words.shape[0]):
                X_count_words.append([len(list(set(X_words[idx_c])))])

            X_count_words = np.array(X_count_words)

            X_all = np.concatenate((X_w2v, X_tfidf), axis=1)
            X_all = np.concatenate((X_all, X_count_words), axis=1)
            # # print(X_count_words)
            #
            # X_count_chars = []
            # for idx_c in range(X_words.shape[0]):
            #     count_chars = 0
            #     for w in X_words[idx_c]:
            #         num_chars = len(w)
            #         count_chars += num_chars
            #     X_count_chars.append([count_chars])
            #
            # X_count_chars = np.array(X_count_chars)
            # # X_count = np.hstack((X_count_words, X_count_chars))
            # X_count = X_count_chars

            """ SET THE INSTANCES FOR THIS COLUMN"""
            X_train = X_all[train_indices]
            Y_train = Y[train_indices]
            Y_train_all = Y_all[train_indices]

            X_test = X_all[test_indices]
            Y_test = Y[test_indices]
            Y_test_all = Y_all[test_indices]

            for idx_test_r in range(Y_test_all.shape[0]):
                for idx_test_c in range(Y_test_all.shape[1]):
                    if Y_test_all[idx_test_r, idx_test_c] == 0.:
                        Y_test_all[idx_test_r, idx_test_c] = -1.

            # print(X_ind.shape[0])

            """ Sample test data """
            X_test_pos = []
            X_test_neg = []
            for idx in range(X_test.shape[0]):
                if np.array_equal(Y_test[idx], [0, 1]):
                    X_test_pos.append(X_test[idx])
                else:
                    X_test_neg.append(X_test[idx])

            X_test_pos = np.array(X_test_pos)
            X_test_neg = np.array(X_test_neg)

            if X_test_pos.shape[0] < X_test_neg.shape[0]:
                X_test_neg = X_test_neg[:X_test_pos.shape[0]]
            else:
                X_test_pos = X_test_pos[:X_test_neg.shape[0]]

            X_test_final = np.concatenate((X_test_neg, X_test_pos), axis=0)
            Y_test_final = np.array([-1.] * X_test_neg.shape[0] + [1.] * X_test_pos.shape[0])


            """
            Subsmaple the negative samples to balance the
            dataset for the training data
            """

            X_train_pos = []
            X_train_neg = []
            for idx in range(X_train.shape[0]):
                # if np.array_equal(Y_train[idx], [0, 1]):
                if np.array_equal(Y_train[idx], [0, 1]):
                    X_train_pos.append(X_train[idx])
                else:
                    X_train_neg.append(X_train[idx])

            X_train_pos = np.array(X_train_pos)
            X_train_neg = np.array(X_train_neg)

            Y_train_new = []
            for idx_train in range(Y_train.shape[0]):
                if np.array_equal(Y_train[idx_train], [0, 1]):
                    Y_train_new.append(1.)
                else:
                    Y_train_new.append(-1.)

            Y_train = np.array(Y_train_new)
            # print(X_test_final)
            Y_all_pos = Y_train_all[Y_train == 1.]
            Y_all_neg = Y_train_all[Y_train == -1.]

            if X_train_pos.shape[0] < X_train_neg.shape[0]:
                X_train_neg = X_train_neg[:X_train_pos.shape[0]]
                Y_all_neg = Y_all_neg[:X_train_pos.shape[0]]
            else:
                X_train_pos = X_train_pos[:X_train_neg.shape[0]]
                Y_all_pos = Y_all_pos[:X_train_neg.shape[0]]

            X_train_final = np.concatenate((X_train_neg, X_train_pos), axis=0)
            Y_train_final = np.array([-1.] * X_train_neg.shape[0] + [1.] * X_train_pos.shape[0])
            Y_all_final = np.concatenate((Y_all_neg, Y_all_pos), axis=0)

            # TODO: try DBSCAN or affinity propagation for clustering subsamples
            #  from the negative samples

            # X_train_pos = X_train[Y_train == 1.]
            # X_train_neg = X_train[Y_train == -1.]
            # Y_all_pos = Y_train_all[Y_train == 1.]
            # Y_all_neg = Y_train_all[Y_train == -1.]
            #
            # print(X_train_neg.shape[0]/X_train_pos.shape[0])
            # # X_train_cluster, Y_all_new = clusterDoc(X_train_neg, Y_all_neg, pref=-100)
            # X_train_cluster = clusterDBSCAN(X_train_neg)
            # print(X_train_cluster.shape[0]/X_train_pos.shape[0])
            # perc_neg = X_train_cluster.shape[0]/X_train_pos.shape[0]
            # if perc_neg > 1:
            #     X_train_cluster = X_train_cluster[:X_train_pos.shape[0]]
            #     Y_all_new = Y_all_new[:X_train_pos.shape[0]]
            # #
            # print(X_train_cluster.shape[0]/X_train_pos.shape[0])
            # exit()
            #
            # X_train_final = np.concatenate((X_train_pos, X_train_cluster), axis=0)
            # Y_train_final = np.array([1.]*X_train_pos.shape[0] + [-1.]*X_train_cluster.shape[0])
            # Y_all_final = np.concatenate((Y_all_pos, Y_all_neg), axis=0)

            # Prediction Model
            # print(X_train_final, Y_train_final)
            # clf_svm = svm.LinearSVC(penalty='l2')
            # clf_svm.fit(X_train_final, Y_train_final)
            #
            # Y_predict = clf_svm.predict(X_test_final)
            #
            # Y_random = []
            # for idx_r in range(X_test_final.shape[0]):
            #     Y_random.append(random.sample([-1, 1], 1)[0])
            # Y_random = np.array(Y_random).astype(int)
            #
            # random_f1[col - 2] += sklearn.metrics.f1_score(Y_test_final, Y_predict)
            # print(sklearn.metrics.f1_score(Y_test_final, Y_predict), sklearn.metrics.f1_score(Y_test_final, Y_random))

            #  Write the samples to disk
            output_dir = '../../../darkweb_data/05/5_31/data_test/v1/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # print("Fold: {}, Column: {}".format(idx_fold, col))
            # print(X_test_final[0], Y_test_final[0])
            # pickle.dump(X_train_final, open(output_dir + 'X_train_l.pickle', 'wb'))
            # pickle.dump(Y_train_final, open(output_dir + 'Y_train_l.pickle', 'wb'))
            # pickle.dump(X_test, open('../../../darkweb_data/05/5_31/data_test/v1/fold_' + str(idx_fold) +
            #                          '/' + 'X_test.pickle', 'wb'))
            # pickle.dump(Y_test, open('../../../darkweb_data/05/5_31/data_test/v1/fold_' + str(idx_fold) +
            #                          '/' + 'Y_test.pickle', 'wb'))
            # pickle.dump(Y_all_final, open(output_dir + 'Y_train_all.pickle', 'wb'))
            # pickle.dump(Y_test_all, open('../../../darkweb_data/05/5_31/data_test/v1/fold_' + str(idx_fold) +
            #                          '/' + 'Y_test_all.pickle', 'wb'))



    # print("random: ", np.array(random_f1) / len(train_fold))

if __name__ == "__main__":
    main()