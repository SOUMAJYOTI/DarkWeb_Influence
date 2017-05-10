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

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(open('dictionary.txt', 'r').read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts = [a + c + b for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

if __name__ == "__main__":
    impForums = [40, 48, 54, 84, 89, 125]

    forumsData = pd.read_csv('F:/Github/DarkWeb_Influence/darkweb_data/1_3/Forumdata_40.csv', encoding = "ISO-8859-1")
    forumsData.columns = ['idx', 'boardsName', 'forumsId', 'language', 'postContent',
                                     'postCve', 'postMs', 'postedDate', 'postsId',
                                     'scrapedDate', 'topicId', 'topicsName', 'usersId']
    forumsData = forumsData.drop_duplicates(subset=['boardsName', 'forumsId', 'language', 'postContent'
                                                    , 'postCve', 'postMs', 'postedDate',
                                                    'scrapedDate', 'topicId', 'topicsName', 'usersId'])
    userIDs = forumsData['usersId'].unique()
    fPosts= forumsData.sort_values(by='postedDate')
    print(len(forumsData))
    exit()

    # boards = forumsData['boardsName'].unique()
    # boardGroups = fPosts.groupby('boardsName').describe().unstack()
    # bgSorted = boardGroups.sort([('forumsId', 'count')], ascending=False)
    # print(bgSorted)


    port = PorterStemmer()
    wnl = WordNetLemmatizer()
    userPostsdict = {}
    for idx_users in range(len(list(userIDs))):
        fposts_user = fPosts[fPosts['usersId'] == userIDs[idx_users]]
        userPostsdict[userIDs[idx_users]] = len(fposts_user)
        if len(fposts_user) < 10:
            continue
        posts = list(fposts_user['postContent'])
        docs_user = []
        print(idx_users, len(posts))

        for idx_p in range(len(posts)):
            # print(idx_p)
            cont = posts[idx_p]
            try:
                cont = BeautifulSoup(cont).get_text()
            except:
                continue
            cont = re.sub(r'http\S+', '', cont)
            cont = re.sub("[^a-zA-Z]",  # The pattern to search for
                                  " ",  # The pattern to replace it with
                                  cont)  # The text to search
            # cont = cont.lower().split()
            # cont = " ".join(cont)
            # cont = " ".join([port.stem(i) for i in cont.split()])
            # cont = " ".join([wnl.lemmatize(i) for i in cont.split()])
            tokens = nltk.word_tokenize(cont)
            tokens = nltk.pos_tag(tokens)
            cont = ''
            for (word, tag) in tokens:
                if tag != 'JJ' and tag != 'VBN' and tag != 'RB':
                    cont += (word + ' ')
                    continue

                word_correct = correct(word)
                cont += (word_correct + ' ')
                # print(word, word_correct)

            cont = cont + "."
            docs_user.append(cont)
        thefile = open('user_docs/user_' + str(userIDs[idx_users]) + '.txt', 'w')
        for item in docs_user:
            thefile.write("%s " % item)

    # plt.hist(list(userPostsdict.values()), bins=30)
    # plt.show()