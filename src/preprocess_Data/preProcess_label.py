## GEt the corpus sentences forumwise

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
import os
import pickle

def getSentences(content):
    sent_list = content.split('.')
    # print(sent_list)

    #Now filter some sentences which are actually not period aborted sentences
    new_sent_list = []
    for s in sent_list:
        if len(s) < 5:
            continue
        new_sent_list.append(s.lstrip())

    if len(new_sent_list) == 0:
        return []

    process_sent_list = []
    for s in new_sent_list:
        cont = s
        try:
            cont = BeautifulSoup(cont).get_text()
            # print(cont)
        except:
            return ''

        cont = re.sub(r'http\S+', '', cont)
        # cont = cont[:len(cont)-1]
        cont = re.sub("[^a-zA-Z']",  # The pattern to search for
                      " ",  # The pattern to replace it with
                      cont)  # The text to search

        cont = " ".join(cont.split())
        words = cont.split(' ')
        temp = ''
        for w in range(len(words)):
            if re.search(r'((\w)\2{2,})', words[w]):
                continue

            if "aa" in words[w] or "xx" in words[w] or "yy" in words[w] or \
                            "zz" in words[w] or "yx" in words[w] or "zx" in words[w] or "xz" in \
                    words[w]:
                continue
            if len(words[w]) < 15 and len(words[w]) > 2 and words[w] != 'quote':
                temp += (words[w] + ' ')
            # if words[w] == 'quote' and (w+1 < len(range(len(words)))):
            #     words[w+1] = ''
        if temp == '':
            # print('hello', cont_temp)
            continue
        # temp = cont
        # if temp[0] == ' ':
        #     temp = temp[1:]
        temp = temp[:len(temp)-1]
        process_sent_list.append(temp)
    # print(process_sent_list)
    return process_sent_list


def getStopWords(data):
    for line in data:
        words = line.split(' ')
    # print(len(words))
    return words


# Smartly remove duplicate sentences
def getCorpusPosts(fPosts, stop):
    docPosts = []
    contentSeen = []
    labels = [] # keep track of the labels when removing rows

    for idx_p, row in fPosts.iterrows():
        cont = row['postContent']
        if row['Uncodeable'] == 1:
            continue
        temp = getSentences(cont)
        # print(cont, idx_p)
        if len(temp) == 0:
            # print(cont, idx_p, row['usersId'])
            continue
        l = np.array(list(row.ix[3:12]))
        if np.count_nonzero(l) == 0:
            continue
        labels.append(l)
        docPosts.append(temp)
        # if len(docPosts) > 3:
        #     break

    labels_filtered = []
    mdocs = []
    for d in range(len(docPosts)):
        temp_sent = []
        sent_list = docPosts[d]
        for cont in sent_list:
            temp = ''
            words = cont.split(' ')
            # if len(words) <= 2:
            #     continue
            cnt_words = 0
            # Remove stopwords
            for w in range(len(words)):
                if words[w] not in stop:
                    cnt_words += 1
                    temp += (words[w] + ' ')

            if temp == '' or temp == ' ' or cnt_words < 3:
                continue
            temp = temp[:len(temp)-1]

            if temp.lower() not in contentSeen:
                temp_sent.append(temp)
                contentSeen.append(temp.lower())
        if len(temp_sent) == 0:
            continue

        labels_filtered.append(labels[d])
        mdocs.append(temp_sent)

        # if cnt_doc > 5:
        #     break

    return mdocs, labels_filtered


if __name__ == "__main__":
    stopwords_file = open('../../darkweb_data/Stop_Words.txt', 'r')
    stopwords = getStopWords(stopwords_file)

    forumsData = pd.read_csv('../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)

    print('Number of de-duplicated forum posts', len(forumsData))

    corpus, labels = getCorpusPosts(forumsData, stopwords)

    # Save the preprocessed data to file
    fId = forumsData['forumsId'][0]

    start = 'all'
    directory = '../../darkweb_data/5_10'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle.dump(corpus, open(directory + '/forum_' + str(fId) + '_' + str(start) + '_input_docs.pickle', 'wb'))
    pickle.dump(labels, open(directory + '/forum_' + str(fId) + '_' + str(start) + '_input_labels.pickle', 'wb'))
