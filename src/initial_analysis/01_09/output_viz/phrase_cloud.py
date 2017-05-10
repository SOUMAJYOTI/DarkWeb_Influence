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
from wordcloud import WordCloud


# Plot utilities
def plot_bars(x, y, titles=[]):
    # print(len(titles))
    width = 0.8
    plt.bar(x, y, width, color="blue")
    if len(titles) > 0:
        major_ticks = np.arange(0, len(titles), 1)
        labels = ['']
        for i in major_ticks:
            labels.append(str(titles[i])[:10])

        plt.xticks(major_ticks, labels, rotation=70, size=20)
    else:
        plt.xticks(size=20)
    plt.yticks(size=20)
    # plt.xlabel('Month-Year (Time)', size=25)
    plt.ylabel('Count of phrases in posts', size=25)
    # plt.title('Month-wise post counts', size=20)

    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.95)
    plt.grid(True)
    plt.show()


# Load the global phrase-topic data
fId = 40
output_dir = '../../../../darkweb_data/nlp_process/forum/40/phrases_month_03-01-2015/'
out_phrases_file = output_dir + 'output_phrases.txt'
out_phrases = open(out_phrases_file, 'r')

topic_phrase_dict = {}
cur_topic = ''
phrases = []
phrase_freq = {}
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
        # line = line.replace(" ", "")
        line = line.lstrip()
        w = line.split(':')
        if w[0] == 'K':
            break
        if w[0] == '':
            continue
        phrases.append(w[0])
        w[1] = w[1].lstrip()
        # phrase_freq[w[0]] = float(w[1])


dict_phrases = open(output_dir + '/dict_phrases.txt', 'r')
phrase_freq_dict = {}
for line in dict_phrases:
    w = line.split(',')
    phrase_freq_dict[w[0]] = int(w[2][:len(w[2]) - 1])
# sorted_x = sorted(phrase_freq_dict.items(), key=operator.itemgetter(1), reverse=True)

phrase_cloud_doc = {}
cnt_topic = 1
topic_phrase_dist = []
topic_titles = []
x = []
for t in topic_phrase_dict:
    # if cnt_topic > 1:
    #     break
    print(t)
    topic_phrase_count = 0
    phrase_cloud_doc[t] = ''
    for p in topic_phrase_dict[t]:
        count_p = int(phrase_freq_dict[p])
        topic_phrase_count += count_p
        tokens = p.split(' ')
        if len(tokens) > 1:
            p_word = "_".join(tokens)
        else:
            p_word = p
        if p =='time':
            continue
        print(count_p)
        # count_p *= 100
        for idx in range(count_p):
            phrase_cloud_doc[t] += (p_word + ' ')
    topic_phrase_dist.append(topic_phrase_count)
    topic_titles.append(t)
    x.append(cnt_topic)
    cnt_topic += 1

plot_bars(x, topic_phrase_dist, topic_titles)
for t in phrase_cloud_doc:
    text = phrase_cloud_doc[t]
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud)
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=100, height=1280, width=1280).generate(text)
    WordCloud.to_file(wordcloud, '../../../../plots/wordle_topics/' + t + '.png')
    # plt.figure()
    # # plt.show(wordcloud)
    # plt.savefig()
    # plt.close()
    # plt.axis("off")
    # plt.show()

