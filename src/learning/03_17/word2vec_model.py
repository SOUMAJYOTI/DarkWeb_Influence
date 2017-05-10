
from gensim.models import Word2Vec
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def word2vec_model(sentences):
    min_count = 3
    size = 200
    window = 4

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

    sentences = createSentences(sentences_data)
    w2v_feat, vocab = word2vec_model(sentences)

    print(type(w2v_feat.vocab))

    print(len(vocab))

    # pickle.dump(w2v_feat, open('../../../darkweb_data/3_25/word2vec_train_model_d200_min4.pickle', 'wb'))
    # pickle.dump(vocab, open('../../../darkweb_data/3_25/word2vec_train_vocab_d200_min4.pickle', 'wb'))



