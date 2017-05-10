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
    for line in dict_phrases:
        w = line.split(',')
        words = w[0].split(' ')
        words = list(set(words))
        if True: #len(words) >= 2:
            # print(w[0])
            count = int(w[2][:len(w[2]) - 1])

            # if count <= 10:
            #     continue
            # if count >= 1000:
            #     # continue
            #     # print(w[0])
            #     continue
            phrase_freq[w[0]] = count

    return phrase_freq
    # print(len(phrase_freq))
    # plt.figure(figsize=(12, 8))
    # hfont = {'fontname': 'Arial'}
    # n, bins, patches = plt.hist(phrase_freq, 30, lw=3, facecolor='b')
    # # plt.yscale('log', nonposy='clip', basey=2)
    # # plt.xlabel('Length of the time series', size=40, **hfont)
    # plt.ylabel('Frequency', size=40, **hfont)
    # # plt.title('Histogram of')
    # # plt.xlim([0, 700])
    # # plt.ylim([0, 2 ** (12)])
    # plt.grid(True)
    # plt.xticks(size=25)
    # plt.yticks(size=25)
    # # file_save = dir_save + '/' + 'count_motif_' + str(m) + '.png'
    # # plt.savefig(file_save)
    # plt.subplots_adjust(left=0.16, bottom=0.16)
    # plt.show()
    # plt.close()

def getInputPartitionedPhrases(data, dict, start, tstamps):
    word_list = []
    index_vocab = 1
    times = []
    index_doc = 0
    phrase_freq = []
    lines = []
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
            # print(p[0], count)
            words = p[0].split(' ')
            # print(words)
            # words = list(set(words))
            if len(words) < 2:
                if count <= 300 and count >= 10:
                    # phrase_freq.append(int(p[1]))
                    # for w in words:
                    #     if w not in word_list:
                    #         word_list.append(w)
                    # line_phrase += (str(word_list.index(w)) + ' ')
                    line_phrase += (p[0] + ' ')
            else:
                if count <= 200:
                    for w in range(len(words)):
                        line_phrase += (words[w] + '_')
                    line_phrase = line_phrase[:len(line_phrase)-1] + ' '
                    # line_phrase += (p[0] + ' ')
        if len(line_phrase) == 0:
            continue
        line_phrase = line_phrase[:len(line_phrase)-1] + '\n'
        times.append(tstamps[index_doc])
        index_doc += 1
        # print(line_phrase)
        lines.append(line_phrase)
    # index_doc -= 1
    # index_doc = int(index_doc/2)
    # lines = lines[:index_doc]
    # print(index_doc, len(lines))
    #
    # directory = '../../../darkweb_data/nlp_process/forum/' + str(fId)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # thefile = open(directory + '/phrases_month_' + str(start) + '/inputInter.txt', 'w')
    # pickle.dump(word_list, open(directory + '/phrases_month_' + str(start) + '/word_list_indices.pickle', 'wb'))
    # # thefile = open(directory + '/total_corpus.txt', 'w')
    #
    # thefile.write("vocabSize:%d\tdocNum:%d\n" % (len(word_list)+1, index_doc+1))
    # for item in range(len(lines)):
    #     thefile.write("%s" % lines[item])

    directory = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/v1'
    if not os.path.exists(directory):
        os.makedirs(directory)
    thefile = open(directory + '/inputInter_' + start + '.txt', 'w')
    # pickle.dump(word_list, open(directory + '/phrases_month_' + str(start) + '/word_list_indices.pickle', 'wb'))
    # thefile = open(directory + '/total_corpus.txt', 'w')

    # thefile.write("vocabSize:%d\tdocNum:%d\n" % (len(word_list)+1, index_doc+1))
    for item in range(len(lines)):
        thefile.write("%s" % lines[item])

    thefile = open(directory + '/timestamps_' + start + '.txt', 'w')
    for item in range(len(times)):
        thefile.write("%s" % times[item])


if __name__ == "__main__":
    fId = 40
    date = '02_01_2014'
    output_dir = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/v1/' #+ date + '/'
    dict_phrases_file = output_dir + 'dict_phrases.txt'
    dict_phrases = open(dict_phrases_file, 'r')

    inp_phrases_file = output_dir + 'inp_month_' + date + '.txt'
    inp_phrases = open(inp_phrases_file, 'r')

    tstamps = []
    timestamps_file  = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/v1' + '/timestamps_' + date + '.txt'
    for t in open(timestamps_file, 'r'):
        tstamps.append(t)
    # stopfile = open('../../../darkweb_data/nlp_process/forum/' + 'Stop_Words_Post.txt', 'r')
    # for line in stopfile:
    #     stop_w = line.split(' ')

    dict_phrases = plotDistPhrases(dict_phrases)
    getInputPartitionedPhrases(inp_phrases, dict_phrases, date, tstamps)