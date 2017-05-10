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


def getSentences(content):
    sent_list = content.split('.')
    # print(sent_list)

    # Now filter some sentences which are actually not period aborted sentences
    new_sent_list = []
    for s in sent_list:
        if len(s) < 10:
            continue
        new_sent_list.append(s.lstrip())

    if len(new_sent_list) == 0:
        return []

    process_sent_list = []
    for s in new_sent_list:
        print(s)
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


def getCorpusPosts(fPosts, timecheck=True):
    mdocs = []
    mdocs_indexed = []
    mtimes = []

    docPosts = {}
    timePosts = {}
    contentSeen = []
    for idx_p, row in fPosts.iterrows():
        cont = row['postContent']

        temp = getSentences(cont)
        if temp == []:
            continue
        docPosts[idx_p] = temp

        # if len(docPosts) > 3:
        #     break

    cnt_doc = 0
    # Keep two DS - one for indexing the sentence with the document number
    # and the other for phrase partitioning in Apache Spark Java
    for d in docPosts:
        sent_list = docPosts[d]
        for cont in sent_list:
            temp = ''
            temp_idx = str(d) + ' '
            words = cont.split(' ')
            if len(words) < 3:
                continue
            for w in range(len(words)):
                # countW_Docs = wordsinPostsCount(words[w], docPosts)
                # if countW_Docs >= 10:
                    temp += (words[w] + ' ')
            if temp == '' or temp == ' ':
                continue
            temp = temp[:len(temp)-1] + '.'
            temp_idx += temp
            cnt_doc += 1
            # print(cnt_doc)
            # words = temp.split(' ')
            # if len(words) <= 3:
            #     continue
            mdocs.append(temp)
            mdocs_indexed.append(temp_idx)


        # if cnt_doc > 5:
        #     break

    return (mdocs, mdocs_indexed)


if __name__ == "__main__":
    forumsData = pd.read_csv('../../../darkweb_data/2_28/Forum40_labels.csv', encoding="ISO-8859-1")
    # forumsData = forumsData[np.isfinite(forumsData['scrapedDate'])]

    postures_names = forumsData.columns.values[3:15]

    sentences = []
    forumsData = forumsData.fillna(0)

    corpus, corpus_indexed = getCorpusPosts(forumsData)
    # print(corpus[1])

    print(len(corpus), len(corpus_indexed))

    # corpus = getCorpusPosts(forumsData)

    # Save the preprocessed data to file
    fId = forumsData['forumsId'][0]

    start = 'all'
    directory = '../../../darkweb_data/3_20'
    if not os.path.exists(directory):
        os.makedirs(directory)
    thefile = open(directory + '/forum_' + str(fId) + '_' + '_input_phrases.txt', 'w')
    thefile_idx = open(directory + '/forum_' + str(fId) + '_' + '_input_phrases_indexed.txt', 'w')
    # thefile = open(directory + '/total_corpus.txt', 'w')

    for item in range(len(corpus)):
        # print(corpus[item])
        thefile.write("%s \n" % corpus[item])

    for item in range(len(corpus_indexed)):
        # print(corpus[item])
        thefile_idx.write("%s \n" % corpus_indexed[item])
