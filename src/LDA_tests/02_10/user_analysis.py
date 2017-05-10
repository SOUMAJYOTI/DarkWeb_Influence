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

# User Posts count by Months
def getUserPostsbyMonth(fPosts):
    fId = int(list(fPosts['forumsId'])[0])
    fPosts['postedDate'] = pd.to_datetime(fPosts['postedDate'])
    countPosts = fPosts.groupby([fPosts['postedDate'].dt.month, fPosts['postedDate'].dt.year]).agg({'count'})

    postsMonth = {}
    x = []
    y = []
    titles = []
    cnt = 0
    for id, row in countPosts.iterrows():
        mn, yr = row.name
        date = datetime.datetime.strptime(str(mn) + '-' + str(yr), '%m-%Y')
        postsMonth[date] = int(row['usersId'])

    sortedDates = sorted(postsMonth.keys())
    for d in range(len(sortedDates)):
        titles.append(str(sortedDates[d]))
        x.append(cnt)
        y.append(postsMonth[sortedDates[d]])
        cnt += 1


def getTopUsers(fPosts):
    fId = int(list(fPosts['forumsId'])[0])
    users = fPosts['usersId'].unique()

    usersPosts = {}
    for id, row in fPosts.iterrows():
        if row['usersId'] not in usersPosts:
            usersPosts[row['usersId']] = 0
        usersPosts[row['usersId']] += 1

    topUsers = sorted(usersPosts.items(), key=operator.itemgetter(1), reverse=True)[:5]
    # for u, c in topUsers:
    #     print(u, c)
    #     fposts_user = fPosts[fPosts['usersId']==u]
    #     fposts_user = fposts_user.reset_index(drop=True)
    #     getUserPostsbyMonth(fposts_user)


