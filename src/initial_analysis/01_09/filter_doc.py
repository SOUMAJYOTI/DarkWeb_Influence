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
#
# file = 'stop_list.txt'
# wds = []
# with open(file, 'r') as f:
#     for line in f:
#         line = line[:len(line)-1]
#         words = line.split(' ')
#         for w in words:
#             w = w.split('\t')
#             print(w)
#             for i in w:
#                 if i == '':
#                     continue
#                 if i not in wds:
#                     wds.append(i)
#
# file = 'stopwords.txt'
# with open(file, 'r') as f:
#     for line in f:
#         line = line[:len(line)-1]
#         line = line.split(' ')
#         for w in line:
#             w = w.split('\t')
#             # print(w)
#             for i in w:
#                 if i == '':
#                     continue
#                 if i not in wds:
#                     wds.append(i)
#                     # print(i)
# print(len(wds))

# thefile = open('F:\Github\DarkWeb_Influence\src\initial_analysis\\01_09\Stop_Words.txt', 'w')
#
# for item in wds:
#     # print(item)
#     thefile.write("%s " % item)

stopfile = open('Stop_Words.txt', 'r')
for line in stopfile:
    stop_w = line.split(' ')

directory = 'user_docs'
doc = []
files = os.listdir(directory)
cnt_doc = 0
for file in files:
    # if cnt_doc > 1:
    #     break
    with open(directory + '/' + file, 'r') as f:
        for line in f:
            line = line.split('.')
            for s in line:
                if len(s) < 1:
                    continue
                if s[0] == ' ':
                    s = s[1:]
                words = s.split(' ')
                temp = ''
                for w in range(len(words)):
                    if words[w] not in stop_w:
                        temp += (words[w] + ' ')
                if temp == '':
                    continue
                temp = temp[:len(temp)-1] + '.\n'
                if temp not in doc:
                    doc.append(temp)
                # print(temp)
    cnt_doc += 1
thefile = open('corpus.txt', 'w')
for item in doc:
    # print(item)
    thefile.write("%s " % item)
# print(doc)