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
# import seaborn

def plot_line(x, y, x_title=[], legends=''):
    fig = plt.figure()
    ax = plt.subplot(111)  # row x col x position (here 1 x 1 x 1)

    plt.xticks(x, x_title, size=30, rotation=0)  # rotate x-axis labels to 75 degree
    plt.yticks(size=30)
    # for i in range(len(y)):
    ax.plot(x, y,  marker='o', linestyle='-', label=legends, linewidth=3)

    # plt.xlim(0, len(var) + 1)
    plt.tight_layout()  # showing xticks (as xticks have long names)
    ax.grid()

    # plt.title('Network_cover vs Motif edge density ', color='#000000', weight="bold", size=50)
    plt.ylabel('Number of posts', size=40)
    plt.xlabel('Number of postures', size=40)
    # plt.ylim([0, 100])
    # ax.legend(loc='upper right', fancybox=True, shadow=True, fontsize=25)
    plt.grid(True)
    plt.show()


def plot_box(data_to_plot, titles):
    fig = plt.figure(1, figsize=(12, 8))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # label = ['Database', 'Grade Changes', 'Phone']
    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    # colors_face = ['white', 'white', '#DCDCDC', '#DCDCDC', '#696969', '#696969']
    # hatch_pattern = ['|X', '|X', '', '', '', '']
    idx = 0
    for box in bp['boxes']:
        # change outline color
        box.set(color='#000000', linewidth=4)
        # change fill color
        box.set(facecolor='#FFFFFF')
        # box.set(hatch=hatch_pattern[idx])
        idx += 1

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#000000', linewidth=4)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#000000', linewidth=4)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#000000', linewidth=6)

    # change the style of fliers    and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    third_quartile = [item.get_ydata()[0] for item in bp['whiskers']]
    third_quartile = max(third_quartile)

    first_quartile = [item.get_ydata()[1] for item in bp['whiskers']]
    first_quartile = max(first_quartile)

    plt.grid(True)
    hfont = {'fontname': 'Arial'}
    ax.set_xlabel('Number of postures', fontsize=30, **hfont)
    ax.set_ylabel('Document length (sentences)', fontsize=30, **hfont)
    plt.ylim([0, 50])
    # plt.xlim([0, 5])
    plt.tick_params('y', labelsize=30)
    ax.set_xticklabels(titles, size=30, rotation=45, ha='right', **hfont)

    plt.show()


def getSentences(content):
    sent_list = content.split('.')
    # print(sent_list)

    #Now filter some sentences which are actually not period aborted sentences
    new_sent_list = []
    for s in sent_list:
        if len(s) < 3:
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


def avg_sentences(fPosts, stop):
    docPosts = []
    contentSeen = []
    labels = []  # keep track of the labels when removing rows

    posture_label = [0. for _ in range(11)]
    for idx_p, row in fPosts.iterrows():
        cont = row['postContent']
        if row['Uncodeable'] == 1:
            continue
        temp = getSentences(cont)
        # print(cont, idx_p)
        if len(temp) == 0:
            # print(cont, idx_p, row['usersId'])
            continue
        l = np.array(list(row.ix[2:12]))
        if np.count_nonzero(l) == 0:
            continue
        num_postures = np.count_nonzero(l)
        posture_label[num_postures] += 1

    print(posture_label)
    posture_label = posture_label[1:]
    # plot_box(posture_label, range(1, 11))
    plot_line(list(range(1, 11)), posture_label, x_title=list(range(1, 11)))


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
        l = np.array(list(row.ix[2:12]))
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
        if len(temp_sent) == 0 and len(docPosts) > 1:
            continue
        else:
            temp_sent = sent_list

        labels_filtered.append(labels[d])
        mdocs.append(temp_sent)

        # if cnt_doc > 5:
        #     break

    return mdocs, labels_filtered


if __name__ == "__main__":
    stopwords_file = open('../../darkweb_data/Stop_Words.txt', 'r')
    stopwords = getStopWords(stopwords_file)

    forumsData = pd.read_csv('../../darkweb_data/5_15/Forum_40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)

    print('Number of de-duplicated forum posts', len(forumsData))

    # avg_sentences(forumsData, stopwords) # just for analysis
    corpus, labels = getCorpusPosts(forumsData, stopwords)
    print(len(corpus), len(labels))
    #
    # Save the preprocessed data to file
    fId = forumsData['forumsId'][0]

    start = 'label'
    directory = '../../darkweb_data/5_15'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle.dump(corpus, open(directory + '/forum_' + str(fId) + '_' + str(start) + '_input_docs.pickle', 'wb'))
    pickle.dump(labels, open(directory + '/forum_' + str(fId) + '_input_labels.pickle', 'wb'))