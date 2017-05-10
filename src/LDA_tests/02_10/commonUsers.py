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

# Plot utilities
def plot_bars(x, y, titles=[]):
    width = 1
    plt.bar(x, y, width, lw=2, color="blue")
    if len(titles) > 0:
        major_ticks = np.arange(0, len(titles), 3)
        labels = []
        for i in major_ticks:
            labels.append(str(titles[i])[:10])

        plt.xticks(major_ticks, labels, rotation=45, size=20, ha='center')
    else:
        plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('Month-Year (Time)', size=25)
    plt.ylabel('Count of posts', size=25)
    # plt.title('Month-wise post counts', size=20)

    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.95)
    plt.grid(True)
    plt.show()


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
    plot_bars(x, y, titles)


# Top users by posts counts
def getTopUsers(fPosts):
    fId = int(list(fPosts['forumsId'])[0])
    users = fPosts['usersId'].unique()

    usersPosts = {}
    for id, row in fPosts.iterrows():
        if row['usersId'] not in usersPosts:
            usersPosts[row['usersId']] = 0
        usersPosts[row['usersId']] += 1

    topUsers = sorted(usersPosts.items(), key=operator.itemgetter(1), reverse=True)[:5]
    for u, c in topUsers:
        print(u, c)
        fposts_user = fPosts[fPosts['usersId']==u]
        fposts_user = fposts_user.reset_index(drop=True)
        getUserPostsbyMonth(fposts_user)


def deDuplicate(fData):
    fData.columns = ['idx', 'boardsName', 'financialTags', 'forumsId', 'language', 'postContent',
                               'postCve', 'postMs', 'postedDate', 'postsId',
                               'scrapedDate', 'softwareTags', 'topicId', 'topicsName', 'usersId']
    fData_deDup = fData.drop_duplicates(subset=['postsId'])
    fData_deDup = fData_deDup.drop_duplicates(subset=['boardsName', 'postContent'
        , 'postedDate', 'topicsName', 'usersId'])
    fData_deDup = fData_deDup[pd.notnull(fData_deDup['postContent'])]
    return fData_deDup

if __name__ == "__main__":
    forumsData = [pd.DataFrame() for _ in range(10)]
    forums = [40, 101, 84]
    for f1 in range(len(forums)):
        forumsData[f1] = pd.read_csv('../../../darkweb_data/2_2/Forumdata_' + str(forums[f1])
                                      + '.csv', encoding="ISO-8859-1")

        fdata_1 = deDuplicate(forumsData[f1])
        users_f1 = fdata_1['usersId']
        for f2 in range(f1+1, len(forums)):
            forumsData[f2] = pd.read_csv('../../../darkweb_data/2_2/Forumdata_' + str(forums[f2])
                                         + '.csv', encoding="ISO-8859-1")
            fdata_2 = deDuplicate(forumsData[f2])
            users_f2 = fdata_2['usersId']
            commUsers = set(users_f1).intersection(set(users_f2))
            print('No. of common users between ', forums[f1], 'and', forums[f2], ': ', len(list(commUsers)))

        # getUserPostsbyMonth(forumsData)

    # getTopUsers(forumsData)