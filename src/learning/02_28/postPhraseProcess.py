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
import scipy.stats as scs
from scipy.stats import gaussian_kde
from textblob import TextBlob as tb
# from tf_idf import tfidf


def getDictPhrases(dict_phrases):
    phrasesWords = {}
    for line in dict_phrases:
        w = line.split(',')
        phrasesWords[w[0]] = int(w[2][:len(w[2]) - 1])

    return phrasesWords


def getInputPartitionedPhrases(data, dict):
    index_doc = 0
    doc = []
    for line in data:

        # if index_doc > 10:
        #     break
        # print(index_doc)

        # for train crisis events data, append the doc number at front
        line_phrase = [index_doc]
        index_doc += 1
        line = line[:len(line)-1]
        doc_phr = line.split(',')
        for ph in range(len(doc_phr)):
            phrases = doc_phr[ph]
            p = phrases.split(':')
            count = dict[p[0]]
            words = p[0].split(' ')

            # separate conditions for single words and sentences !!!
            #1. Single word filter
            if len(words) < 2:
                if count > 3 and (len(words[0]) > 3):
                    line_phrase.append(p[0])
            #2. Phrase filter
            else:
                if (count <= 500 and count > 2) or len(words) > 3:
                    merged_phrase = ''
                    for w in range(len(words)):
                        merged_phrase += (words[w] + '_')
                    line_phrase.append(merged_phrase[:len(merged_phrase)-1])
        if len(line_phrase) == 0:
            continue
        # if index_doc > 1000:
        #     break
        doc.append(line_phrase)

    return doc


def calculateTfIdf(docs):
    bloblist = []
    for d in range(len(docs)):
        doc = docs[d]
        doc = doc[:len(doc) - 1]
        bloblist.append(tb(doc))

    ScoresDoc = [{} for _ in range(len(docs))]
    for i, blob in enumerate(bloblist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            ScoresDoc[i][word] = score
            # print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))

    return ScoresDoc


if __name__ == "__main__":

    print('Load files....')
    output_dir = '../../../darkweb_data/3_20/'
    dict_phrases_file = output_dir + 'dict_phrases_sg_3.txt'
    dict_phrases = open(dict_phrases_file, 'r')

    inputData = open(output_dir + 'partitionOut_sg_3.txt', 'r')

    dictionary = getDictPhrases(dict_phrases)
    print('Start partitioning...')
    docs = getInputPartitionedPhrases(inputData, dictionary)
    print(len(docs))
    # tfidfScore = calculateTfIdf(docs)

    pickle.dump(docs, open(output_dir + 'docs_phrases_f40_labels.pickle', 'wb'))

    # stopfile = open('../../../darkweb_data/nlp_process/Stop_Words_post.txt', 'r')
    # for line in stopfile:
    #     stop_w = line.split(' ')
    # tfidf_score = pickle.load(open(output_dir + 'tfIDf_score.pickle', 'rb'))
    #
    # newDocs = []
    # cnt_doc = 0
    # newTimes = []
    # for d in range(len(docs)):
    #     doc = docs[d]
    #     tfidf_scores = tfidf_score[d]
    #     words = doc.split(' ')
    #     newDoc = ''
    #     cnt = 0
    #     for w in words:
    #         if w in tfidf_scores:
    #             if tfidf_scores[w] < 0.05 or w in stopfile:
    #                 continue
    #             # else:
    #             #     print(tfidf_scores[w])
    #         newDoc += (w + ' ')
    #         cnt += 1
    #     if cnt < 4:
    #         continue
    #     newDoc = newDoc[:len(newDoc)-1] + '\n'
    #     newDocs.append(newDoc)
    #     newTimes.append(times[d])
    #
    # directory = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/all/v1/'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # thefile = open(directory + '/all.txt', 'w')
    #
    # for item in range(len(newDocs)):
    #     thefile.write("%s" % newDocs[item])
    # thefile.close()
    #
    # thefile = open(directory + '/timestamps.txt', 'w')
    # for item in range(len(newTimes)):
    #     thefile.write("%s" % newTimes[item])
    #
    #
    #
    #
    #
    #
