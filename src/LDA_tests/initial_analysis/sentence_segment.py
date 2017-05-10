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
    stopfile = open('../../darkweb_data/nlp_process/Stop_Words.txt', 'r')
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
        # print(cont)
    except:
        return ''
    #remove the qute from sentences
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
        return ''
    # temp = cont
    temp = temp[:len(temp) - 1]
    return temp


def getCorpusPosts(fPosts):
    mdocs = []
    mtimes = []

    docPosts = {}
    timePosts = {}
    contentSeen = []
    for idx_p, row in fPosts.iterrows():
        cont = row['postContent']
        times = row['postedDate'] + " 00:00:00"
        time_struct = datetime.datetime.strptime(times, '%Y-%m-%d %H:%M:%S')
        time_tuple = time.mktime(time_struct.timetuple())
        # cont = str(month_docs[idx_p])
        temp = preProcessPosts(cont, row['idx'], contentSeen)
        if temp == '' or temp == ' ':
            continue
        docPosts[row['idx']] = temp
        timePosts[row['idx']] = time_tuple
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
        temp = temp[:len(temp)-1] + ".\n"
        cnt_doc += 1
        print(cnt_doc)
        words = temp.split(' ')
        if len(words) <= 3:
            continue
        mdocs.append(temp)
        mtimes.append(str(timePosts[d]) + '\n')

        # if cnt_doc > 5:
        #     break

    return (mdocs, mtimes)

# Check in how many documents does word exist - need better data structures for fast search !!!
def wordsinPostsCount(word, docPosts):
    countWordInDocs = 1
    for d in docPosts:
        words = docPosts[d].split(' ')
        if word in words:
            countWordInDocs += 1
    return countWordInDocs

if __name__ == "__main__":
    forumsData = pd.read_csv('../../darkweb_data/2_2/Forumdata_40.csv', encoding="ISO-8859-1")
    forumsData.columns = ['idx', 'boardsName', 'financialTags', 'forumsId', 'language', 'postContent',
                                     'postCve', 'postMs', 'postedDate', 'postsId',
                                     'scrapedDate', 'softwareTags', 'topicId', 'topicsName', 'usersId']
    forumsData = forumsData.drop_duplicates(subset=['postsId'])
    forumsData = forumsData.drop_duplicates(subset=['boardsName', 'postContent'
                                                    ,'postedDate',
                                                     'topicsName', 'usersId'])
    forumsData = forumsData[pd.notnull(forumsData['postContent'])]
    print('Number of de-duplicated forum posts', len(forumsData))

    # User posts for spikes - Need to automate this !!!!!
    spikeDates = [('02-01-2014', '02-28-2014')]

    fId = int(list(forumsData['forumsId'])[0])
    for start, end in spikeDates:
        spikePosts = forumsData[pd.to_datetime(forumsData['postedDate']) >= pd.to_datetime(start)]
        spikePosts = spikePosts[pd.to_datetime(spikePosts['postedDate']) <= pd.to_datetime(end)]
        corpus, timestamps = getCorpusPosts(spikePosts)

        print(len(corpus), len(timestamps))
        # corpus = getCorpusPosts(forumsData)
        directory = '../../darkweb_data/2_2/nlp_process/forum/' + str(fId)
        if not os.path.exists(directory):
            os.makedirs(directory)
        thefile = open(directory + '/month_' + str(start) + '.txt', 'w')
        # thefile = open(directory + '/total_corpus.txt', 'w')

        for item in range(len(corpus)):
            thefile.write("%s" % corpus[item])

        thefile = open(directory + '/timestamps_' + str(start) + '.txt', 'w')

        for item in range(len(timestamps)):
            thefile.write("%s" % timestamps[item])

    # getTopUsers(forumsData)
