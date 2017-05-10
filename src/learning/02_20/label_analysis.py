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
    cont = re.sub("[^a-zA-Z]",  # The pattern to search for
                  " ",  # The pattern to replace it with
                  cont)  # The text to search

    words = cont.split(' ')
    temp = ''
    for w in range(len(words)):
        if re.search(r'((\w)\2{2,})', words[w]):
            continue
        if "xx" in words[w] or "yy" in words[w] or "zz" in words[w] or "yx" in words[w] or "zx" in words[w] or "xz" in \
                words[w]:
            continue
        if len(words[w]) < 12 and len(words[w]) > 3 and words[w] != 'quote':
            temp += (words[w] + ' ')
        if words[w] == 'quote':
            words[w+1] = ''
    if temp == '':
        # print('hello', cont_temp)
        return ''
    # temp = cont
    temp = temp[:len(temp) - 1]
    return temp


# Preprocess forum posts
def preProcessPosts(cont, contentSeen):
    stopfile = open('../../../darkweb_data/nlp_process/Stop_Words.txt', 'r')
    for line in stopfile:
        stop_w = line.split(' ')
    # print(len(fPosts))
    # print(cont)
    if len(cont) < 1:
        return ''
    if cont[0] == ' ':
        cont = cont[1:]
    if cont in contentSeen:
        return ''
    contentSeen.append(cont)

    try:
        cont = BeautifulSoup(cont).get_text()
        # print(cont)
    except:
        return ''
    #remove the quote from sentences
    cont = re.sub(r'http\S+', '', cont)
    sentences = re.split('[.?]', cont)
    cont = ''
    for s in sentences:
        s = s.lstrip()
        # print(s)
        words = s.split(' ')
        if len(words) > 4:
            if 'quote' in words[0]:
                continue
            cont += (s + ' ')
    cont = cont[:len(cont)-1]
    cont = re.sub("[^a-zA-Z]",  # The pattern to search for
                  " ",  # The pattern to replace it with
                  cont)  # The text to search

    words = cont.split(' ')
    temp = ''
    for w in range(len(words)):
        if re.search(r'((\w)\2{2,})', words[w]):
            continue
        if "xx" in words[w] or "yy" in words[w] or "zz" in words[w] or "yx" in words[w] or "zx" in words[w] or "xz" in \
                words[w]:
            continue
        if words[w] not in stop_w and len(words[w]) < 12 and len(words[w]) > 3:
            temp += (words[w] + ' ')
    if temp == '':
        # print('hello', cont_temp)
        return ''
    # temp = cont
    temp = temp[:len(temp) - 1]
    return temp


def getCorpusPosts(fPosts):
    mdocs = []

    docPosts = {}
    contentSeen = []
    for idx_p, row in fPosts.iterrows():
        cont = row['postContent']

        temp = preProcessPosts(cont, contentSeen)
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
        temp = temp[:len(temp)-1] + "."
        cnt_doc += 1
        # print(cnt_doc)
        words = temp.split(' ')
        if len(words) < 3:
            continue
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
    forumsData = pd.read_csv('Forum40_120labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData[np.isfinite(forumsData['scrapedDate'])]
    postures_names = forumsData.columns.values[3:16]

    train_sentences = []
    for i, r in forumsData.iterrows():
        if i > 60:
            continue
        processPost = getSentences(r['postContent'])

        words = processPost.split(' ')
        train_sentences.append(words)

    pickle.dump(train_sentences, open('train_posts_60.pickle', 'wb'))




