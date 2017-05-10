
from gensim.models import Word2Vec
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def word2vec_model(sentences):
    min_count = 2
    size = 50
    window = 3

    model = Word2Vec(sentences, min_count=min_count, size=size, window=window, sg=1)
    vocab = list(model.vocab.keys())

    return model, vocab


def make_ClfData(sentences, w2v_model, fData, start):
    sent_featVec = []
    labels = []
    for s in range(len(sentences)):
        temp_vec = []
        labels_vec = []
        label_temp = fData.ix[start+s, 3:16].tolist()
        # for l in range(len(label_temp)):
        #     if label_temp[l] == 1.0:
        #         labels_vec.append(l)
        words = sentences[s]
        for w in range(len(words)):
            if words[w] not in w2v_model:
                continue
            temp_vec.append(w2v_model[words[w]])
        sent_featVec.append(temp_vec)
        labels.append(label_temp)

    # mlb = MultiLabelBinarizer()
    # labels = mlb.fit_transform(labels)
    return sent_featVec, labels


def multiLabelClf_train(X, Y, subplot, title, transform):
    # if transform == "pca":
    #     X = PCA(n_components=2).fit_transform(X)
    # elif transform == "cca":
    #     X = CCA(n_components=2).fit(X, Y).transform(X)
    # else:
    #     raise ValueError

    # min_x = np.min(X[:, 0])
    # max_x = np.max(X[:, 0])
    #
    # min_y = np.min(X[:, 1])
    # max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC())
    classif.fit(X, Y)

    return classif


def multiLabelClf_test(classif, X):
    # mlb = MultiLabelBinarizer()
    predicted = classif.predict(X)
    # all_labels = mlb.inverse_transform(predicted)
    return predicted


def getFeat_sum(sent_vec, labels):
    X = np.array([]).reshape(0, 20)
    Y = np.array([]).reshape(0, 13)
    for idx_vec in range(len(sent_vec)):
        # pca = PCA(n_components=4)
        sum_vec = np.mean(sent_vec[idx_vec], axis=0)
        # print(sent_vec[idx_vec], sum_vec)
        if len(sent_vec[idx_vec]) <= 4:
            continue

        # x_dash = pca.fit_transform(np.transpose(sent_vec[idx_vec]))
        # # X.append(np.transpose(x_dash))
        Y = np.vstack([Y, labels[idx_vec]])
        X = np.vstack([X, sum_vec])

    return X, Y


def createSentences(data):
    sent_list = []
    for line in data:
        line = line[:len(line)-3]
        sent_words = line.split(' ')
        sent_list.append(sent_words)

    return sent_list

if __name__ == "__main__":
    sentences_data = open('../../../darkweb_data/3_20/forums_all_preTrain_sent.txt', 'r')
    forumsData = pd.read_csv('Forum40_120labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData[np.isfinite(forumsData['scrapedDate'])]
    # postures_names = forumsData.columns.values[3:16]

    # sentences = pickle.load(open('train_posts_60.pickle', 'rb'))
    # w2v_feat, vocab = word2vec_model(sentences)

    sentences = createSentences(sentences_data)
    w2v_feat, vocab = word2vec_model(sentences)

    print(len(vocab))

    pickle.dump(w2v_feat, open('word2vec_train_model.pickle', 'wb'))
    pickle.dump(vocab, open('word2vec_train_vocab.pickle', 'wb'))

    # sent_vec, labels = make_ClfData(sentences, w2v_feat, forumsData, 0)
    #
    # X_train, Y_train = getFeat_sum(sent_vec, labels)
    # Y_train = Y_train.astype(int)
    # # print(Y_train[:20])
    # clf = multiLabelClf_train(X_train, Y_train, 2, "With unlabeled samples + PCA", "pca")
    # #
    # sentences = pickle.load(open('test_posts_60.pickle', 'rb'))
    # sent_vec, labels = make_ClfData(sentences, w2v_feat, forumsData, 61)
    # X_test, Y_test = getFeat_sum(sent_vec, labels)
    # Y_test = Y_test.astype(int)
    # all_labels = multiLabelClf_test(clf, X_test)
    # # print(Y_test)
    # print(all_labels)
    # X_test, Y_test = getFeat_sum(sent_vec, labels)
    #
    # for r in range(X_test.shape[0]):
    #     x = X_test[r,:]
        # print(x)
        # print(multiLabelClf_test(clf, x))


