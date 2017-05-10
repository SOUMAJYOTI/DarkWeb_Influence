# Use the affinity propagation algorithm to cluster sentences
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import scipy.stats as scst
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import gensim
from imblearn.over_sampling import SMOTE

def createSentences(data):
    sent_list = []
    docs_list = []
    cur_doc = 0
    for line in data:
        words = line.split(' ')
        sent_index = int(words[0])
        if sent_index != cur_doc:
            cur_doc += 1
            docs_list.append(sent_list)
            sent_list = []

        line = line[:len(line)-3]
        sent_words = line.split(' ')[1:]
        sent_list.append(sent_words)

    return docs_list


def getDocFeatVectors(doc, d2v_feat):
    doc_vec = []
    for d in range(len(doc)):
        sent = doc[d]
        sent_vec = d2v_feat.infer_vector(sent, alpha=0.025, min_alpha=0.0001)

        doc_vec.append(sent_vec)

    return np.array(doc_vec)

def getFeatVectors(doc, w2v_feat):
    doc_vec = []
    for d in range(len(doc)):
        sent = doc[d]
        sum_w2v = []
        for s in sent:
            if s in w2v_feat:
                sum_w2v.append(w2v_feat[s])

        sent_vec = np.mean(np.array(sum_w2v), axis=0)
        doc_vec.append(sent_vec)

    return np.array(doc_vec)

# def getDocsFeat(docs, w2v_feat):

def clusterDoc(featDocs, pref):

    # Compute Affinity Propagation
    af = AffinityPropagation(preference=pref).fit(featDocs)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    # max_occur = max(set(labels), key=labels.count)
    max_label = scst.mode(labels, axis=None)[0][0]

    tagged_indices = []
    for l in range(len(labels)):
        if labels[l] == max_label:
            tagged_indices.append(l)
    return tagged_indices


def get_X_Y_data(forumsData, docs, w2v_feat):
    dict_doc_sent = {}  # hashmap for documnet sentence map
    positive_posts = []
    negative_posts = []
    positive_indices = []
    negative_indices = []
    for row in range(len(forumsData) - 1):
        v = getFeatVectors(docs[row], w2v_feat)
        temp_idx = []
        for idx in range(len(v)):
            temp_idx.append((row, idx))
            dict_doc_sent[(row, idx)] = v[idx]
        if forumsData.ix[row, col] == 1.0:
            positive_posts.extend(v)
            positive_indices.extend(temp_idx)
        else:
            negative_posts.extend(v)
            negative_indices.extend(temp_idx)

    X_inst = [[] for _ in range(len(forumsData) - 1)]
    Y_inst = [0.0 for _ in range(len(forumsData) - 1)]

    pos_tags = clusterDoc(positive_posts, pref=-50)
    # print()
    sent_doc_idx = []
    for idx in range(len(pos_tags)):
        sent_doc_idx.append(positive_indices[pos_tags[idx]])
    # print(sent_doc_idx)

    pca = PCA(n_components=1)
    positive_feat = []
    count_pos = 0
    for row in range(len(forumsData) - 1):
        if forumsData.ix[row, col] == 1.0:
            count_pos += 1
            v = getFeatVectors(docs[row], w2v_feat)
            row_feat = np.zeros((1, 100))
            for idx in range(len(v)):
                if (row, idx) in sent_doc_idx:
                    # row_feat.append(np.array(v[idx]))
                    row_feat = np.vstack((row_feat, np.array(v[idx])))
                    # if row_feat.shape[0] >= 500:
                    #     break

            row_feat = np.mean(row_feat, axis=0)
            row_feat = row_feat.transpose()

            X_inst[row] = row_feat
            Y_inst[row] = np.array([1.0])
    # print(np.array(positive_feat).shape)

    neg_tags = clusterDoc(negative_posts, pref=-10)
    # print(len(neg_tags) / len(negative_indices))
    sent_doc_idx = []
    for idx in range(len(neg_tags)):
        sent_doc_idx.append(negative_indices[neg_tags[idx]])
    # print(sent_doc_idx)

    pca = PCA(n_components=1)
    negative_feat = []
    count_neg = 0
    for row in range(len(forumsData) - 1):
        if forumsData.ix[row, col] == 0.0:
            count_neg += 1
            v = getFeatVectors(docs[row], w2v_feat)
            row_feat = np.zeros((1, 100))
            for idx in range(len(v)):
                if (row, idx) in sent_doc_idx:
                    # row_feat.append(np.array(v[idx]))
                    row_feat = np.vstack((row_feat, np.array(v[idx])))
                    # if row_feat.shape[0] >= 500:
                    #     break

            row_feat = np.mean(row_feat, axis=0)
            row_feat = row_feat.transpose()

            X_inst[row] = row_feat
            Y_inst[row] = np.array([0.0])
            # if count_neg > 3*count_pos:
            #     break
    # print(np.array(negative_feat).shape)

    # print(count_neg)
    return X_inst, Y_inst, len(pos_tags) / len(positive_indices), len(neg_tags) / len(negative_indices)


if __name__ == "__main__":
    # Load the trained word2vec model and the sentences
    sentences_data = open('../../../darkweb_data/3_20/forum_40_input_phrases_indexed.txt', 'r')
    docs = createSentences(sentences_data)
    w2v_feat = pickle.load(open('word2vec_train_model.pickle', 'rb'))
    d2v_feat = gensim.models.Doc2Vec.load('doc2vec_train_model.model')

    # For each label, create the clusters of positive and negative sentences
    forumsData = pd.read_csv('Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)

    for col in range(7, 14):
        X_inst, Y_inst , pos_perc, neg_perc = get_X_Y_data(forumsData, docs, w2v_feat)

        train_inst = int(0.8 * len(X_inst))
        # print(train_inst)
        # print(len(X_inst))
        X_train = np.array(X_inst[train_inst:])
        Y_train = np.array(Y_inst[train_inst:])
        X_test = np.array(X_inst[:train_inst])
        Y_test = np.array(Y_inst[:train_inst])

        new_X = []
        new_Y = []
        count_neg = 0
        count_pos = 0
        for y in range(len(Y_train)):
            if Y_train[y] == 1.0:
                count_pos += 1
                new_X.append(X_train[y])
                new_Y.append(Y_train[y])
            else:
                if count_neg > 400:
                    continue
                new_X.append(X_train[y])
                new_Y.append(Y_train[y])
                count_neg += 1

        print(count_pos, count_neg)
        from sklearn import linear_model

        logistic = linear_model.LogisticRegression(C=1e5)
        logistic.fit(new_X, new_Y)
        # clf = SVC()
        # print(X_train.shape, Y_train.shape)
        # clf.fit(new_X, new_Y)
        predicted = logistic.predict(X_test)
        # print(predicted)
        if col == 3:
            Y_predict = predicted
            Y_test_all = Y_test
        else:
            Y_predict = np.dstack((Y_predict, predicted))
            Y_test_all = np.dstack((Y_test_all, Y_test))

    Y_test_all = np.squeeze(Y_test_all, axis=(1,))
    Y_predict = np.squeeze(Y_predict, axis=(0,))

    h = 0
    for idx in range(len(Y_predict)):
        from sklearn.metrics import hamming_loss

        # print(Y_test[idx])
        # print(Y_predict[0][idx])
        h += hamming_loss(Y_test_all[idx], Y_predict[idx])

    print(h / 87)

    # sentences_data = open('../../../darkweb_data/3_20/forum_40_input_phrases_indexed.txt', 'r')
    # # forumsData = pd.read_csv('Forum40_120labels.csv', encoding="ISO-8859-1")
    # # forumsData = forumsData[np.isfinite(forumsData['scrapedDate'])]
    # postures_names = forumsData.columns.values[3:16]
    #
    # # sentences = pickle.load(open('train_posts_60.pickle', 'rb'))
    # # w2v_feat, vocab = word2vec_model(sentences)
    #
    # docs = createSentences(sentences_data)
    # # print(docs[:2])
    #
    # w2v_feat = pickle.load(open('word2vec_train_model.pickle', 'rb'))
    # vocab = pickle.load(open('word2vec_train_vocab.pickle', 'rb'))
    #
    # docs_feat = []
    # for d in range(len(docs)):
    #     doc = docs[d]
    #     feat = clusterDoc(doc)
    #     docs_feat.append(feat)
    #
    # print(len(docs_feat))
    # pickle.dump(docs_feat, open('docs_feat_all.pickle', 'wb'))
    #
