# Train the doc2vec model

import datetime as dt
import pandas as pd
import requests
import operator
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime
import time
from collections import Counter
import calendar
import calendar
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import gensim
from gensim import models

# Plot utilities for  a bars chart
def plot_bars(x, y, x_label, y_label, col, xTitles=[]):
    width = 1
    plt.bar(x, y, width, color=col)
    if len(xTitles) > 0:
        major_ticks = np.arange(0, len(xTitles), 2)
        labels = []
        for i in major_ticks:
            labels.append(str(xTitles[i])[:10])

        plt.xticks(major_ticks, labels, rotation=45, size=20)
    else:
        plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel(x_label, size=25)
    plt.ylabel(y_label, size=25)
    # plt.title('Month-wise post counts', size=20)

    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.95)
    plt.grid(True)
    plt.show()

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
    corpus = open('../../../darkweb_data/2_28/corpus_forums_sentences.txt', 'r')
    lineId = 1
    for line in corpus:
        line = line[:len(line)-1]
        sentence = line.split(' ')
        labeled_sent = models.doc2vec.TaggedDocument(
                words=sentence, tags=["SENT_%s" % lineId])
        lineId += 1
        sentences.append(labeled_sent)

    # doc_phrases = pickle.load(open('../../../darkweb_data/2_28/docs_phrases.pickle', 'rb'))
    # for dp in range(len(doc_phrases)):
    #     labeled_sent = models.doc2vec.TaggedDocument(
    #         words=doc_phrases[dp], tags=["SENT_%s" % lineId])
    #     lineId += 1
    #     sentences.append(labeled_sent)
    # print(len(doc_phrases))

    print('Start of doc2vec...')
    model = gensim.models.Doc2Vec(sentences, dm=0, alpha=0.025, size=100, min_alpha=0.025, min_count=0)

    model.save('../../../darkweb_data/2_28/trained_doc2vec_forums_sentences_d100.model')
    # print(model.docvecs['SENT_1'])
        # sentences = getSentences()

    # train_sentences = []
    # for i, r in forumsData.iterrows():
    #     count_postures = 0
    #     for pn in postures_names:
    #         if r[pn] == 1.:
    #             count_postures += 1
    #     if count_postures <= 2:
    #         print(r['postContent'])




