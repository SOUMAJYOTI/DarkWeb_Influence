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
from tf_idf import tfidf

# Check in how many documents does word exist - need better data structures for fast search !!!
def wordsinPostsCount(word, docPosts):
    countWordInDocs = 1
    for d in docPosts:
        words = docPosts[d].split(' ')
        if word in words:
            countWordInDocs += 1
    return countWordInDocs


def plotDistPhrases(dict_phrases):
    phrase_freq = {}
    word_freq = {}
    phrase_count = 0
    word_count = 0
    phrases = []
    words = []
    for line in dict_phrases:
        w = line.split(',')
        wds = w[0].split(' ')
        wds = list(set(wds))
        if len(wds) >= 2:
            # consider phrases of frequency > 5
            if int(w[2][:len(w[2]) - 1]) <= 5:
                continue
            phrases.append(w[0])
            phrase_count += int(w[2][:len(w[2]) - 1])
            phrase_freq[w[0]] = int(w[2][:len(w[2]) - 1])

        else:
            # consider words of frequency > 10
            if int(w[2][:len(w[2]) - 1]) <= 10:
                continue
            words.append(w[0])
            word_count += int(w[2][:len(w[2]) - 1])
            word_freq[w[0]] = int(w[2][:len(w[2]) - 1])

    phrases = list(set(phrases))
    words = list(set(words))
    print(phrase_count, word_count, len(phrases), len(words))
    # return phrase_freq

    probDist_phrases = []
    probDist_words = []
    probDensity_values = []
    arr = np.array(list(phrase_freq.values()))
    arrSet = list(set(arr))
    s = len(arr)
    counts_phrases= []

    for i in range(len(arrSet)):
        # print(arr[i])
        counts_phrases.append(arrSet[i])
        probDist_phrases.append(np.count_nonzero(arr == arrSet[i])/s)

    # print(max(probDensity_values))
    probDensity_values = []
    arr = np.array(list(word_freq.values()))
    arrSet = list(set(arr))
    s = len(arr)
    counts_words = []

    for i in range(len(arrSet)):
        # print(arr[i])
        counts_words.append(arrSet[i])
        probDist_words.append(np.count_nonzero(arr == arrSet[i]) / s)

    # plt.plot(counts_phrases, probDist_phrases, color="blue", linewidth=2.0, linestyle="-")
    # hfont = {'fontname': 'Arial'}
    # plt.ylabel('Prob. of occurrence', size=40, **hfont)
    # plt.xlabel('Phrase counts', size=40, **hfont)
    # plt.title('Phrase density function', size=30)
    # # plt.xlim([0, 500])
    # # plt.ylim([0, 2 ** (12)])
    # plt.grid(True)
    # plt.xticks(size=25)
    # plt.yticks(size=25)
    # plt.show()

    return {}


def getDictPhrases(dict_phrases):
    phrasesWords = {}
    for line in dict_phrases:
        w = line.split(',')
        phrasesWords[w[0]] = int(w[2][:len(w[2]) - 1])

    return phrasesWords


def getInputPartitionedPhrases(data, dict, tstamps):
    times = []
    index_doc = 0
    lines = []
    doc = []
    for line in data:
        # if index_doc > 10:
        #     break
        line_phrase = ''
        line = line[:len(line)-1]
        doc_phr = line.split(',')
        for ph in range(len(doc_phr)):
            phrases = doc_phr[ph]
            p = phrases.split(':')
            count = dict[p[0]]
            words = p[0].split(' ')

            if len(words) < 2:
                if count <= 300 and count >= 10:
                    line_phrase += (p[0] + ' ')
            else:
                if count <= 200:
                    for w in range(len(words)):
                        line_phrase += (words[w] + '_')
                    line_phrase = line_phrase[:len(line_phrase)-1] + ' '
        if len(line_phrase) == 0:
            continue
        line_phrase = line_phrase[:len(line_phrase)-1] + '\n'
        times.append(tstamps[index_doc])
        index_doc += 1
        # print(line_phrase)
        lines.append(line_phrase)
        doc.append(line_phrase[:len(line_phrase)-1])


    # directory = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/all'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # thefile = open(directory + '/all.txt', 'w')
    # # pickle.dump(word_list, open(directory + '/phrases_month_' + str(start) + '/word_list_indices.pickle', 'wb'))
    # # thefile = open(directory + '/total_corpus.txt', 'w')
    #
    # # thefile.write("vocabSize:%d\tdocNum:%d\n" % (len(word_list)+1, index_doc+1))
    # for item in range(len(lines)):
    #     thefile.write("%s" % lines[item])
    # thefile.close()
    #
    # thefile = open(directory + '/timestamps.txt', 'w')
    # for item in range(len(times)):
    #     thefile.write("%s" % times[item])

    return (doc, times)


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

# def getLdaInputDoc(docs, tfIdfScores):
#     for d in range(len(docs)):
#         doc = docs[d]
#         words = doc.split(' ')
#
#         for w in words:
#             if

if __name__ == "__main__":
    forumsId = [40]
    # date = '02_01_2014'
    for f in range(len(forumsId)):
        fId = forumsId[f]
        output_dir = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/all/' #+ date + '/'
        dict_phrases_file = output_dir + 'dict_phrases.txt'
        dict_phrases = open(dict_phrases_file, 'r')

        inputData = open(output_dir + 'inputPartitioned.txt', 'r')

        tstamps = []
        timestamps_file = output_dir + '/timestamps_all.txt'
        for t in open(timestamps_file, 'r'):
            tstamps.append(t)

        dictionary = getDictPhrases(dict_phrases)
        docs, times = getInputPartitionedPhrases(inputData, dictionary, tstamps)
        # tfidfScore = calculateTfIdf(docs)

        # pickle.dump(tfidfScore, open(output_dir+'tfIDf_score.pickle', 'wb'))

        stopfile = open('../../../darkweb_data/nlp_process/Stop_Words_post.txt', 'r')
        for line in stopfile:
            stop_w = line.split(' ')
        tfidf_score = pickle.load(open(output_dir + 'tfIDf_score.pickle', 'rb'))

        newDocs = []
        cnt_doc = 0
        newTimes = []
        for d in range(len(docs)):
            doc = docs[d]
            tfidf_scores = tfidf_score[d]
            words = doc.split(' ')
            newDoc = ''
            cnt = 0
            for w in words:
                if w in tfidf_scores:
                    if tfidf_scores[w] < 0.05 or w in stopfile:
                        continue
                    # else:
                    #     print(tfidf_scores[w])
                newDoc += (w + ' ')
                cnt += 1
            if cnt < 4:
                continue
            newDoc = newDoc[:len(newDoc)-1] + '\n'
            newDocs.append(newDoc)
            newTimes.append(times[d])

        directory = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/all/v1/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        thefile = open(directory + '/all.txt', 'w')

        for item in range(len(newDocs)):
            thefile.write("%s" % newDocs[item])
        thefile.close()

        thefile = open(directory + '/timestamps.txt', 'w')
        for item in range(len(newTimes)):
            thefile.write("%s" % newTimes[item])






