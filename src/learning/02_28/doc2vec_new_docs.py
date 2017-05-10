# Get the document vectors for training + test corpus

import datetime as dt
import pandas as pd
import requests
import operator
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import gensim
from gensim import models
import preProcess as ppr
import re
from bs4 import BeautifulSoup


def getSentences(cont):
    if len(cont) < 1:
        return ''

    try:
        cont = BeautifulSoup(cont).get_text()
        # print(cont)
    except:
        return ''

    cont = re.sub(r'http\S+', '', cont)
    cont = cont[:len(cont)-1]
    cont = re.sub("[^a-zA-Z']",  # The pattern to search for
                  " ",  # The pattern to replace it with
                  cont)  # The text to search

    cont = " ".join(cont.split())
    words = cont.split(' ')
    temp = ''
    for w in range(len(words)):
        if re.search(r'((\w)\2{2,})', words[w]):
            continue

        if "xx" in words[w] or "yy" in words[w] or "zz" in words[w] or "yx" in words[w] or "zx" in words[w] or "xz" in \
                words[w]:
            continue
        if len(words[w]) < 15 and words[w] != 'quote':
            temp += (words[w] + ' ')
        if words[w] == 'quote' and (w+1 < len(range(len(words)))):
            words[w+1] = ''
    if temp == '':
        # print('hello', cont_temp)
        return ''
    # temp = cont
    if temp[0] == ' ':
        temp = temp[1:]
    temp = temp[:len(temp) - 1]
    return temp


def getCorpusPosts(fPosts):
    mdocs = []

    docPosts = {}
    for idx_p, row in fPosts.iterrows():
        cont = row['postContent']

        temp = getSentences(cont)
        if temp == '' or temp == ' ':
            continue
        docPosts[idx_p] = temp
        # if len(docPosts) > 3:
        #     break

    cnt_doc = 0
    for d in docPosts:
        temp = ''
        cont = docPosts[d]
        words = cont.split(' ')
        for w in range(len(words)):
            # countW_Docs = wordsinPostsCount(words[w], docPosts)
            # if countW_Docs >= 10:
                temp += (words[w] + ' ')
        if temp == '' or temp == ' ':
            continue
        temp = temp[:len(temp)-1]
        cnt_doc += 1
        # print(cnt_doc)
        # if len(words) < 3:
        #     continue
        mdocs.append(temp)

        # if cnt_doc > 5:
        #     break

    return (mdocs)


def segmentPosts_Labels(pnames):
    for p_n in pnames:
        print(p_n)
        corpus_df = forumsData[forumsData[p_n] == 1.0]
        # print(corpus_df)
        # corpus = corpus_df['postContent'].tolist()
        processedPosts = getCorpusPosts(corpus_df)

        directory = './postures_seg_posts/'

        if not os.path.exists(directory):
            os.makedirs(directory)
        thefile = open(directory + p_n + '_posts.txt', 'w')

        for item in range(len(processedPosts)):
            thefile.write("%s\n" % processedPosts[item])
        thefile.close()


if __name__ == "__main__":
    # forumsData = pd.read_csv('../../../darkweb_data/2_2/Forumdata_40.csv', encoding="ISO-8859-1")
    forumsData = pd.read_csv('../../../darkweb_data/2_28/Forum40_labels.csv', encoding="ISO-8859-1")
    # forumsData = forumsData[np.isfinite(forumsData['scrapedDate'])]

    postures_names = forumsData.columns.values[3:15]

    sentences = []
    forumsData = forumsData.fillna(0)

    corpus = getCorpusPosts(forumsData)

    # start = 'labeled'
    # directory = '../../../darkweb_data/2_28'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # thefile = open(directory + '/forum_' + str(40) + '_' + str(start) + '.txt', 'w')
    # # thefile = open(directory + '/total_corpus.txt', 'w')
    #
    # for item in range(len(corpus)):
    #     thefile.write("%s\n" % corpus[item])

    train_model = gensim.models.Doc2Vec.load('../../../darkweb_data/2_28/trained_doc2vec_forums_phrases.model')
    # corpus = open('../../../darkweb_data/2_28/corpus_forums_sentences.txt', 'r')
    train_phrases = pickle.load(open('../../../darkweb_data/2_28/docs_phrases_f40_labels.pickle', 'rb'))
    lineId = 0
    document_vectors = []
    document_labels = []

    # Sentences features !!!
    for line in corpus:
        line = line[:len(line)-1]
        # print(line)
        sentence = line.split(' ')
        doc_vec = train_model.infer_vector(sentence, alpha=0.025, min_alpha=0.0001)

        l = forumsData.ix[lineId, 3:15].astype(int)
        label_temp = np.array(l.tolist())
        document_vectors.append(doc_vec)
        document_labels.append(label_temp)
        # labeled_sent = models.doc2vec.TaggedDocument(
        #         words=sentence, tags=["SENT_%s" % lineId])
        # lineId += 1
        # sentences.append(labeled_sent)

    # phrases features !!!
    # for tp in range(len(train_phrases)):
    #     if len(train_phrases[tp]) == 1:
    #         lineId += 1
    #         continue
    #     train_phrases[tp] = train_phrases[tp][1:]
    #     doc_vec = train_model.infer_vector(train_phrases[tp], alpha=0.025, min_alpha=0.0001)
    #
    #     l = forumsData.ix[lineId, 3:15].astype(int)
    #     label_temp = np.array(l.tolist())
    #     document_vectors.append(doc_vec)
    #     document_labels.append(label_temp)
        # labeled_sent = models.doc2vec.TaggedDocument(
        #         words=train_phrases[tp], tags=["SENT_%s" % lineId])
        lineId += 1

    pickle.dump((document_vectors, document_labels), open('../../../darkweb_data/2_28/forums40_phrases_train_test.pickle', 'wb'))




