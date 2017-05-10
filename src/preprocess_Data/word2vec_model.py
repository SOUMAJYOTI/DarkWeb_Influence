
from gensim.models import Word2Vec
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from bs4 import BeautifulSoup
import re

def getStopWords(data):
    for line in data:
        words = line.split(' ')
    # print(len(words))
    return words

def word2vec_model(sentences):
    min_count = 2
    size = 50
    window = 4

    model = Word2Vec(sentences, min_count=min_count, size=size, window=window, sg=1)
    vocab = list(model.wv.vocab)

    return model, vocab

def createSentences(data, stop):
    sent_list = []
    for line in data:
        line = line[:len(line)-3]
        sent_words = line.split(' ')
        temp = []
        for w in range(len(sent_words)):
            word = sent_words[w]
            if word not in stop:
                temp.append(word)
        if len(temp) >= 5:
            sent_list.append(sent_words)

    return sent_list

def createVocab(data):
    vocab = {}
    vocab_count = 0
    for line in data:
        line = line[:len(line)-3]
        sent_words = line.split(' ')
        for w in range(len(sent_words)):
            word = sent_words[w]
            if word not in vocab:
                vocab[word] = vocab_count
                vocab_count += 1

    return vocab


if __name__ == "__main__":
    stopwords_file = open('../../darkweb_data/Stop_Words.txt', 'r')
    stopwords = getStopWords(stopwords_file)

    sentences_data = open('../../darkweb_data/3_20/forum_40_all_input_phrases.txt', 'r')

    sentences = createSentences(sentences_data, stopwords)
    w2v_feat, vocab = word2vec_model(sentences)
    sentences_data = open('../../darkweb_data/3_20/forum_40_all_input_phrases.txt', 'r')
    vocab_dict = createVocab(sentences_data)

    print(len(sentences))
    # print(type(w2v_feat.vocab))

    k = 0
    vocab_inv = {}
    for v in vocab:
        vocab_inv[vocab_dict[v]] = v

    # print(len(vocab))

    # pickle.dump(w2v_feat, open('../../darkweb_data/4_23/word2vec_train_model_d50_min3.pickle', 'wb'))
    pickle.dump(w2v_feat, open('../../darkweb_data/5_10/word2vec_train_vocab_d50_min2.pickle', 'wb'))
    pickle.dump(vocab_dict, open('../../darkweb_data/5_10/vocab_dict.pickle', 'wb'))
    pickle.dump(vocab_inv, open('../../darkweb_data/5_10/vocab_inv_dict.pickle', 'wb'))



