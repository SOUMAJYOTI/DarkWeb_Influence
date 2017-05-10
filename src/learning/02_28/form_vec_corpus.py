## Get the sentences of the crisis events posts

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
        temp = temp[:len(temp)-1] + '.'
        cnt_doc += 1
        # print(cnt_doc)
        # if len(words) <= 3:
        #     continue
        mdocs.append(temp)

        # if cnt_doc > 5:
        #     break

    return mdocs


if __name__ == "__main__":
    # forumsData = pd.read_csv('../../../darkweb_data/2_2/Forumdata_40.csv', encoding="ISO-8859-1")
    forumsData = pd.read_csv('../../../darkweb_data/2_28/Forum40_labels.csv', encoding="ISO-8859-1")
    # forumsData = forumsData[np.isfinite(forumsData['scrapedDate'])]

    postures_names = forumsData.columns.values[3:15]

    sentences = []
    forumsData = forumsData.fillna(0)

    corpus = getCorpusPosts(forumsData)
    print(len(corpus))

    # Save the preprocessed data to file

    directory = '../../../darkweb_data/2_28'
    if not os.path.exists(directory):
        os.makedirs(directory)
    thefile = open(directory + '/forum40_labels_corpus.txt', 'w')

    for item in range(len(corpus)):
        thefile.write("%s\n" % corpus[item])

    # thefile = open(directory + '/timestamps_' + str(start) + '.txt', 'w')
    #
    # for item in range(len(timestamps)):
    #     thefile.write("%s" % timestamps[item])

