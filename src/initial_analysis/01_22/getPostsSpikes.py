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

ps = PorterStemmer()

#Longest common substring
def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]

# For spelling correction
def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(open('../../../darkweb_data/nlp_process/dictionary.txt', 'r').read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts = [a + c + b for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or  known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

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

# Preprocess forum posts
def preProcessPosts(cont, idx, contentSeen):
    stopfile = open('../../../darkweb_data/nlp_process/Stop_Words.txt', 'r')
    for line in stopfile:
        stop_w = line.split(' ')
    # print(len(fPosts))
    if len(cont) < 1:
        return ''
    if cont[0] == ' ':
        cont = cont[1:]
    if cont in contentSeen:
        return ''
    contentSeen.append(cont)

    try:
        cont = BeautifulSoup(cont).get_text()
    except:
        return ''
    cont = re.sub(r'http\S+', '', cont)
    cont = re.sub("[^a-zA-Z]",  # The pattern to search for
                  " ",  # The pattern to replace it with
                  cont)  # The text to search

    # No spell checker at this moment !!!!!

    # tokens = nltk.word_tokenize(cont)
    # tokens = nltk.pos_tag(tokens)
    # cont = ''
    # for (word, tag) in tokens:
    #     if tag != 'JJ' and tag != 'VBN' and tag != 'RB':
    #         cont += (word + ' ')
    #         continue
    #
    #     word_correct = correct(word)
    #     cont += (word_correct + ' ')
    #     # print(word, word_correct)

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
        # cont = str(month_docs[idx_p])
        temp = preProcessPosts(cont, row['idx'], contentSeen)
        if temp == '' or temp == ' ':
            continue
        docPosts[row['idx']] = temp

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
        temp = temp[:len(temp)-1] + ".\n"
        cnt_doc += 1
        print(cnt_doc)
        words = temp.split(' ')
        if len(words) <= 3:
            continue
        mdocs.append(temp)

        # if idx_p > 500:
        #     break

    return mdocs

# Check in how many documents does word exist - need better data structures for fast search !!!
def wordsinPostsCount(word, docPosts):
    countWordInDocs = 1
    for d in docPosts:
        words = docPosts[d].split(' ')
        if word in words:
            countWordInDocs += 1
    return countWordInDocs

if __name__ == "__main__":
    forumsData = pd.read_csv('../../../darkweb_data/2_2/Forumdata_84.csv', encoding="ISO-8859-1")
    forumsData.columns = ['idx', 'boardsName', 'financialTags', 'forumsId', 'language', 'postContent',
                                     'postCve', 'postMs', 'postedDate', 'postsId',
                                     'scrapedDate', 'softwareTags', 'topicId', 'topicsName', 'usersId']
    forumsData = forumsData.drop_duplicates(subset=['postsId'])
    forumsData = forumsData.drop_duplicates(subset=['boardsName', 'postContent'
                                                    ,'postedDate',
                                                     'topicsName', 'usersId'])
    forumsData = forumsData[pd.notnull(forumsData['postContent'])]
    print('Number of de-duplicated forum posts', len(forumsData))

    plot_bars()

    # User posts for spikes - Need to automate this !!!!!
    spikeDates = [('11-01-2013', '12-31-2017')]

    fId = int(list(forumsData['forumsId'])[0])
    for start, end in spikeDates:
        spikePosts = forumsData[pd.to_datetime(forumsData['postedDate']) >= pd.to_datetime(start)]
        spikePosts = spikePosts[pd.to_datetime(spikePosts['postedDate']) <= pd.to_datetime(end)]
        corpus = getCorpusPosts(spikePosts)

        # corpus = getCorpusPosts(forumsData)
        directory = '../../../darkweb_data/nlp_process/forum/' + str(fId)
        if not os.path.exists(directory):
            os.makedirs(directory)
        thefile = open(directory + '/month_' + str(start) + '.txt', 'w')
        # thefile = open(directory + '/total_corpus.txt', 'w')

        for item in corpus:
            thefile.write("%s " % item)
    # getTopUsers(forumsData)
