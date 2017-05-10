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
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.metrics import edit_distance
import os

labels = [datetime.date(2014, 4, 10), datetime.date(2014, 5, 11), datetime.date(2014, 6, 12), datetime.date(2014, 7, 13), datetime.date(2014, 8, 14), datetime.date(2014, 9, 15), datetime.date(2014, 10, 16), datetime.date(2014, 11, 17), datetime.date(2014, 12, 18), datetime.date(2015, 1, 19), datetime.date(2015, 2, 21), datetime.date(2015, 3, 22), datetime.date(2015, 4, 23), datetime.date(2015, 5, 24), datetime.date(2015, 6, 25), datetime.date(2015, 7, 26), datetime.date(2015, 8, 27), datetime.date(2015, 9, 28), datetime.date(2015, 10, 29), datetime.date(2015, 11, 30), datetime.date(2016, 1, 1), datetime.date(2016, 2, 2), datetime.date(2016, 3, 3), datetime.date(2016, 4, 4), datetime.date(2016, 5, 5), datetime.date(2016, 6, 6), datetime.date(2016, 9, 15)]
fId = 40
output_dir = '../output_phrases_topics/' + str(fId) + '/phrases_3/'
out_phrases_file = output_dir + 'output_phrases.txt'
out_phrases = open(out_phrases_file, 'r')

topic_phrase_dict = {}
cur_topic = ''
phrases = []
for line in out_phrases:
    line = line[:len(line)-1]
    if line[:5] == "Topic":
        t = line[:len(line)-1]
        phrases = []
        continue
    elif line == '\n':
        continue
    elif line == '':
        topic_phrase_dict[t] = phrases
    else:
        line = line.replace(" ", "")
        w = line.split(':')
        if w[0] == 'K':
            break
        if w[0] == '':
            continue
        phrases.append(w[0])


# Load the global phrase-topic data
fId = 40
output_dir = '../output_months/' + str(fId)
dirs = os.listdir(output_dir)
topic_month = {}
cnt_months = 0
count_phrases = {}
for d in dirs:
    # if cnt_months > 1:
    #     break

    files = os.listdir(output_dir + '/' + str(d))
    dict_phrases = open(output_dir + '/' + str(d) + '/dict_phrases.txt', 'r')
    phrase_freq_dict = {}
    cnt_phrases = 0
    for line in dict_phrases:
        cnt_phrases += 1
        w = line.split(',')
        phrase_freq_dict[w[0]] = int(w[2][:len(w[2]) - 1])
    count_phrases[str(d)] = cnt_phrases
    sorted_x = sorted(phrase_freq_dict.items(), key=operator.itemgetter(1), reverse=True)

    for p, freq in sorted_x:
        if freq <= 3:
            continue
        tokens = p.split(' ')
        for t in topic_phrase_dict:
            if t not in topic_month:
                topic_month[t] = {}
            if str(d) not in topic_month[t]:
                topic_month[t][str(d)] = 0
            # if t not in topic_month[str(d)]:
            #     topic_month[str(d)][t] = 0
            if p in topic_phrase_dict[t]:
                topic_month[t][str(d)] += 1
            elif len(tokens) > 1:
                for w in tokens:
                    if w in topic_phrase_dict[t]:
                        topic_month[t][str(d)] += 1
                        break
    cnt_months += 1

for t in topic_month:
    t_freq = []
    sorted_month = sorted(topic_month[t].items(), key=operator.itemgetter(0))
    for d, val in sorted_month:
        t_freq.append(val/count_phrases[d])
        # labels.append(t)
    countTopics = range(len(topic_month[t]))

    hfont = {'fontname': 'Arial'}
    plt.figure(figsize=(30, 14))
    width=0.5
    major_ticks = np.arange(min(countTopics), max(countTopics), 4)
    titles = []
    for i in major_ticks:
        titles.append(str(labels[int(i)]))
    # plt.xticks(titles)
    print(len(labels))
    plt.xticks(major_ticks, titles, rotation=70)
    plt.bar(countTopics, t_freq, width, color="blue")
    plt.ylabel('# Phrases in ' + t + ' / Total # phrases (all topics)', size=30, **hfont)
    plt.title('Monthwise topic distribution - ' + t, size=30)
    plt.tick_params('x', labelsize=25)
    plt.tick_params('y', labelsize=25)
    plt.subplots_adjust(left=0.13, bottom=0.25)
    plt.grid(True)
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.tight_layout()
    # plt.show()
    plt.savefig('topics_months/' + t +".png")