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

def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum

def getHackingPosts(fromDate=dt.date.today(), toDate=dt.date.today(), start = 0, limNum=0,):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+"&start=" + str(start) + "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getForumData(fromDate=dt.date.today(), toDate=dt.date.today(), start = 0, limNum=0,):
    url = "https://apigargoyle.com/GargoyleApi/getHackingThreads?limit="+str(limNum)+"&start=" + str(start) + "&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

if __name__ == "__main__":
    impForums = [40, 48, 54, 84, 89, 125]

    forumsData = pd.read_csv('F:/Github/DarkWeb_Influence/darkweb_data/1_3/Forumdata_41.csv', encoding = "ISO-8859-1")
    forumsData.columns = ['idx', 'boardsName', 'forumsId', 'language', 'postContent',
                                     'postCve', 'postMs', 'postedDate', 'postsId',
                                     'scrapedDate', 'topicId', 'topicsName', 'usersId']
    forumsData = forumsData.drop_duplicates(subset=['boardsName', 'forumsId', 'language', 'postContent'
                                                    , 'postCve', 'postMs', 'postedDate', 'postsId',
                                                    'scrapedDate', 'topicId', 'topicsName', 'usersId'])
    userIDs = forumsData['usersId'].unique()
    print(len(userIDs))

    forumsData = forumsData.sort_values(by='postedDate')
    forums = forumsData.forumsId.unique() # unique forums
    posts = forumsData.postsId.unique()  # unique forums
    # print(len(posts))

    fNumPosts = {}
    for idx in forums:
        fNumPosts[idx] = len(forumsData[forumsData['forumsId'] == idx])

    # extract the top forums by number of posts
    forumSort = sorted(fNumPosts.items(), key=operator.itemgetter(1), reverse=True)
    topForums = forumSort[:6]

    users_list = {}
    colors = itertools.cycle(["red", "black", "green", "y", "darkcyan", "lightpink"])

    for fId, cntPosts in topForums:
        print(fId)

        if fId != 41:
            continue
        fPosts = forumsData[forumsData['forumsId'] == fId]
        print(len(fPosts))
        users_list[fId] = fPosts['usersId']
        users_list[fId] = list(set(users_list[fId]))

        dates_list = list(fPosts['postedDate'])
        # print(dates_list[0], dates_list[len(dates_list)-1])
        # continue
        fPosts['postedDate'] = pd.to_datetime(fPosts['postedDate'])
        countPosts = fPosts.groupby([fPosts['postedDate'].dt.month, fPosts['postedDate'].dt.year]).agg({'count'})
        dates = countPosts.index.levels
        months = dates[0]
        years = dates[1]

        postsMonth = {}
        x = []
        y = []
        titles = []
        cnt = 0
        for id, row in countPosts.iterrows():
            mn, yr = row.name
            date = datetime.datetime.strptime(str(mn) + '-' +str(yr), '%m-%Y')
            postsMonth[date] = int(row['usersId'])

        sortedDates = sorted(postsMonth.keys())
        for d in range(len(sortedDates)):
            titles.append(str(sortedDates[d]))
            x.append(cnt)
            y.append(postsMonth[sortedDates[d]])
            # x.append(cnt)
            # titles.append(str(mn)+ '-' +str(yr))
            # y.append(int(row['usersId']))
            cnt += 1

        width = 1
        plt.bar(x, y, width, color="blue")
        major_ticks = np.arange(0, len(titles), 1)
        labels=[]
        for i in major_ticks:
            labels.append(str(titles[i])[:10])
        plt.xticks(major_ticks, labels, rotation=45)
        plt.xlabel('Month-Year (Time)', size=20)
        plt.ylabel('Count of posts', size=20)
        plt.title('Month-wise post counts: Forum ' + str(fId), size=20)

        plt.subplots_adjust(left=0.13, bottom=0.20)
        plt.grid(True)
        plt.show()






