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

# Load the global phrase-topic data
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

thefile = open('topic_phrases.txt', 'w')
cnt_topic = 1
for t in topic_phrase_dict:
    # print(item)
    thefile.write("\n%s: \n\n " % t)
    cnt_topic += 1
    for item1, item2 in topic_phrase_dict[t]:
        thefile.write("%s, %s\n" %( item1,  item2))

exit()
# print(topic_phrase_dict['Topic 14'])

stopwords_custom = ['time', 'people', 'day', 'read', 'things', 'list', 'check']

# Load the global phrase dict data
dict_phrases_file = output_dir + 'dict_phrases.txt'
dict_phrases = open(dict_phrases_file, 'r')

phrase_freq_dict = {}
for line in dict_phrases:
    w = line.split(',')
    phrase_freq_dict[w[0]] = int(w[2][:len(w[2])-1])

sorted_x = sorted(phrase_freq_dict.items(), key=operator.itemgetter(1), reverse=True)
cnt = 0
for p, freq in sorted_x:
    # print(p, freq)
    w = p.split(' ')
    if len(w) < 2:
        continue
    print(p, freq)
    # if w[0] in stopwords_custom:
    #     continue
    cnt += 1
    if cnt > 100:
        break

# Load the user topics data by month
output_dir = '../months_docs_users/forums' + str(fId)

# user_phrases_file = output_dir + 'dict_phrases.txt'
# dict_phrases = open(dict_phrases_file, 'r')
# files = os.listdir(user_phrases_file)
# for file in files:


