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


if __name__ == "__main__":
    fId = 40
    date = '03-01-2015'
    output_dir = '../../../darkweb_data/nlp_process/forum/' + str(fId) + '/phrases_month_' + date + '/output/'
    dict_topics = output_dir + 'topics.txt'

    word_indices = pickle.load(open(output_dir +  'word_list_indices.pickle', 'rb'))
    for line in open(dict_topics, 'r'):
        line = line[:len(line)-2]
        ind = 0
        topic_words = line.split(' ')
        for w in range(len(topic_words)):
            if int(topic_words[w]) != 0:
                print(word_indices[w])
        exit()