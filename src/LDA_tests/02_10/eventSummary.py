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
from test_viz import VisualizeTopics
from test_viz import VisualizeEvolution
from prePhraseProcess import preProcessPosts

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


# Check in how many documents does word exist - need better data structures for fast search !!!
def wordsinPostsCount(word, docPosts):
    countWordInDocs = 1
    for d in docPosts:
        words = docPosts[d].split(' ')
        if word in words:
            countWordInDocs += 1
    return countWordInDocs


def deDuplicate(fData):
    fData.columns = ['idx', 'boardsName', 'financialTags', 'forumsId', 'language', 'postContent',
                               'postCve', 'postMs', 'postedDate', 'postsId',
                               'scrapedDate', 'softwareTags', 'topicId', 'topicsName', 'usersId']
    fData_deDup = fData.drop_duplicates(subset=['postsId'])
    fData_deDup = fData_deDup.drop_duplicates(subset=['boardsName', 'postContent'
        , 'postedDate', 'topicsName', 'usersId'])
    fData_deDup = fData_deDup[pd.notnull(fData_deDup['postContent'])]
    return fData_deDup


# Preprocess forum posts to get sentences
def getSentencesDocs(doc):
    # stopfile = open('../../darkweb_data/nlp_process/Stop_Words.txt', 'r')
    # for line in stopfile:
    #     stop_w = line.split(' ')
    # print(len(fPosts))
    if len(doc) < 4:
        return ''

    try:
        cont = BeautifulSoup(doc).get_text()
        # print(cont)
    except:
        return ''

    #remove the quote from sentences
    cont = re.sub(r'http\S+', '', cont)
    # Regex to split by period but avoid float numbers.
    sentences = re.split('(?<!\d)([.?])(?!\d)', cont)
    # print(sentences)
    cont = ''
    for sent in range(len(sentences)):
        s = sentences[sent]
        s = s.lstrip()
        # print(s)
        words = s.split(' ')
        if len(words) > 4:
            if 'quote' in words[0]:
                continue
            cont += (s + ' ')
            if sent+1 < len(sentences) and (sentences[sent+1] == '.' or sentences[sent+1] == '?'
                                            or sentences[sent+1] == ';'):
                cont = cont[:len(cont)-1] + sentences[sent+1] + ' '

    cont = cont[:len(cont)-1]
    cont = re.sub("[^a-zA-Z?.0-9]",  # The pattern to search for
                  " ",  # The pattern to replace it with
                  cont)  # The text to search

    return cont


def eventSlice(doc, time_window, DocTopic, topicWord, time):
        return []


def eventSummary(cont):
    cont = re.sub(r'http\S+', '', cont)
    # Regex to split by period but avoid float numbers.
    sentences = re.split('(?<!\d)([.?])(?!\d)', cont)

    # for s in sentences:

def getTopUsersContent(fPosts):
    usersPosts = {}
    for id, row in fPosts.iterrows():
        if row['usersId'] not in usersPosts:
            usersPosts[row['usersId']] = 0
        usersPosts[row['usersId']] += 1

    topUsers = sorted(usersPosts.items(), key=operator.itemgetter(1), reverse=True)

    topUsersDoc = {}
    count = 0
    for u, cnt in topUsers:
        print(u)
        # print(fPosts[fPosts['usersId'] == u])
        cont = fPosts[fPosts['usersId'] == u]['postContent'].tolist()
        times = fPosts[fPosts['usersId'] == u]['postedDate'].tolist()
        for c in range(len(cont)):
            doc = getSentencesDocs(cont[c])
            if len(doc) == 0:
                continue
            topUsersDoc[cont[c]] = times[c]

        if count > 200:
            break
        count += cnt

    sortedDocs = sorted(topUsersDoc.items(), key=operator.itemgetter(1))
    sortedTimesDoc = []

    for cont, ts in sortedDocs:
        sortedTimesDoc.append(cont)
    return sortedTimesDoc


def vizTopicsWords(par, dictionary):

    spikePosts = fdata[pd.to_datetime(fdata['postedDate']) >= pd.to_datetime(start)]
    spikePosts = spikePosts[pd.to_datetime(spikePosts['postedDate']) <= pd.to_datetime(end)]

    theta = par['m']
    phi = par['n']

    TopicWordsDist = [[] for _ in range(30) for _ in range(par['T'])]

    for t in range(len(phi)):
        mapIndToWords = {}
        for r in range(len(phi[t])):
            mapIndToWords[r] = phi[t][r]

        topWords = sorted(mapIndToWords.items(), key=operator.itemgetter(1), reverse=True)[:10]
        for (ind, val) in topWords:
            TopicWordsDist[t].append(dictionary[ind])

        print(TopicWordsDist[t])


if __name__ == "__main__":
    forumsData = [pd.DataFrame() for _ in range(10)]
    forums = [34]
    for f1 in range(len(forums)):
        forumsData[f1] = pd.read_csv('../../../darkweb_data/2_2/Forumdata_' + str(forums[f1])
                                     + '.csv', encoding="ISO-8859-1")
        fdata = deDuplicate(forumsData[f1])

        print('Number of de-duplicated forum posts', len(fdata))

        # User posts for spikes - Need to automate this !!!!!
        spikeDates = [('06_01_2015', '07_01_2015')]

        sentencDoc = []
        # fPosts = fdata['postContent'].tolist()
        # for fp in range(len(fPosts)):
        #     # print(fp, fPosts[fp])
        #     processedSentence = preProcessDoc(fPosts[fp])
        #     sentencDoc.append(processedSentence)
        #         # print(processedSentence)
        #         # exit()
        #
        # print('Done processing sentences....')

        # Start extracting the topics and word distributions and visualize them

        # tstamps = []
        # timestamps_file = '../../../../darkweb_data/2_2/nlp_process/forum/' + str(forums[f1]) + '/phrases_months/v1' + '/timestamps.txt'
        # for t in open(timestamps_file, 'r'):
        #     tstamps.append(t)

        start = '2015-06-01 00:00:00'
        end = '2015-07-01 00:00:00'
        time_struct = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        start_date = time.mktime(time_struct.timetuple())

        time_struct = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        end_date = time.mktime(time_struct.timetuple())

        spikePosts = fdata[pd.to_datetime(fdata['postedDate']) >= pd.to_datetime(start)]
        spikePosts = spikePosts[pd.to_datetime(spikePosts['postedDate']) <= pd.to_datetime(end)]

        spikePosts['postedDate'] = pd.to_datetime(spikePosts['postedDate'])
        spikePosts = spikePosts.sort(['postedDate'], ascending=[True])

        topUsersDoc = getTopUsersContent(spikePosts)

        newDocs = []
        for cont in topUsersDoc:
            filterDoc = preProcessPosts(cont, [])
            newDocs.append(filterDoc)

        directory = '../../../darkweb_data/2_2/nlp_process/forum/' + str(forums[f1]) + '/phrases_months/v1'
        if not os.path.exists(directory):
            os.makedirs(directory)
        thefile = open(directory + '/month_' + str(spikeDates[0][0]) + '.txt', 'w')
        # thefile = open(directory + '/total_corpus.txt', 'w')

        for item in range(len(newDocs)):
            thefile.write("%s\n" % newDocs[item])

        # print(topUsersDoc)

        # results_path = '../../../darkweb_data/2_2/results/forum/' + str(forums[f1]) + '/v3'
        # tot_pickle_path = results_path + '/' + 'all_tot.pickle'
        # dictionary = pickle.load(open(results_path + '/dict.pickle', 'rb'))
        # tot_pickle = open(tot_pickle_path, 'rb')
        # par = pickle.load(tot_pickle)
        #
        # vizTopicsWords(par, dictionary)