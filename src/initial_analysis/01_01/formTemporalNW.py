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

def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

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

    forumsData = forumsData.sort_values(by='postedDate')

    # Networks by time
    # Edges by sequence on same day

    fId = 41
    users_list = {}
    measure_tnw = []
    fPosts = forumsData[forumsData['forumsId'] == fId]
    users_list[fId] = fPosts['usersId']
    users_list[fId] = list(set(users_list[fId]))

    prev_date = 0
    edges_forum = []
    num_months = 0
    usersSeq = []
    datesSeen = []
    lastTime = pd.to_datetime('1900-01-01')
    num_posts = []
    countDict_global = Counter()
    countMonth = []
    labels = []

    rel_posts = pd.DataFrame()
    for id, row in fPosts.iterrows():
        cur_date = row['postedDate']
        rt_time = datetime.datetime.strptime(cur_date, '%Y-%m-%d')
        cur_date = time.mktime(rt_time.timetuple())

        if rt_time <= lastTime:
            continue
        num_months += 3
        monthDate = add_months(rt_time, 3)
        prevPost_count = len(rel_posts)
        rel_posts = fPosts[pd.to_datetime(fPosts['postedDate']) <= monthDate]
        rel_posts = rel_posts[pd.to_datetime(rel_posts['postedDate']) > lastTime]
        num_posts.append(len(rel_posts) - prevPost_count)

        # print(len(rel_posts))
        # if len(rel_posts) > 1:
        usersRel = list(rel_posts['usersId'])

        # print(len(usersRel))
        for idx in range(len(usersRel)-1):
            if usersRel[idx] == usersRel[idx+1]:
                continue
            edges_forum.append((usersRel[idx], usersRel[idx+1]))

        # print(edges_forum)
        # if len(edges_forum) > 1:
        edges_wt_forums = []
        countDict = Counter(edges_forum)
        items = countDict.most_common()
        for (a, b), c in items:
            edges_wt_forums.append((a, b, c))

        G = nx.DiGraph()
        G.add_weighted_edges_from(edges_wt_forums)
        print(G.number_of_nodes(), G.number_of_edges())

        # nx.draw(G)
        # nx.draw(G, pos=nx.spring_layout(G))  # use spring layout
        # plt.show()
        # plt.close()

        # largest_comp = max(nx.strongly_connected_components(G), key=len)
        # nodes = list(largest_comp)
        # new_edges = []
        # for a,b,c in edges_wt_forums:
        #     if a not in nodes or b not in nodes:
        #         continue
        #     new_edges.append((a,b,c))
        # G = nx.DiGraph()
        # G.add_weighted_edges_from(new_edges)
        # val = nx.diameter(G)

        val = nx.out_degree_centrality(G)

        # path_values = nx.shortest_path_length(G)
        # shortest_paths = 0
        # for v1 in G.nodes():
        #     try:
        #         for v2 in G.nodes():
        #             shortest_paths += path_values[v1][v2]
        #     except:
        #         continue
        #
        # wiener_index = float(shortest_paths / (G.number_of_nodes() * (G.number_of_nodes() - 1)))

        # print(len(list(degrees.values())))
        measure_tnw.append(np.mean(list(val.values())))

        countMonth.append(num_months)
        datesSeen.append(row['postedDate'])
        countDict_global = countDict
        lastTime = pd.to_datetime(add_months(rt_time, 1))
        labels.append(monthDate)
        edges_forum = []
        print(lastTime)

        # if lastTime > pd.to_datetime('2012-01-01'):
        #     break

        # if num_months > 10:
        #     break

    hfont = {'fontname': 'Arial'}
    plt.figure(figsize=(12, 8))
    major_ticks = np.arange(min(countMonth), max(countMonth), 24)
    titles = []
    for i in major_ticks:
        titles.append(str(labels[int(i/3)])[:10])
    plt.xticks(major_ticks, titles, rotation=45)
    plt.plot(countMonth, measure_tnw, linewidth=1.5, label='Forum ' + str(fId))
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.ylabel('Out-degrees', fontsize=30, **hfont)

    plt.tick_params('x', labelsize=25)
    plt.tick_params('y', labelsize=25)

    plt.subplots_adjust(left=0.13, bottom=0.25)

    plt.show()
    plt.close()

    # plt.figure(figsize=(12, 8))
    # width=1
    # major_ticks = np.arange(min(countMonth), max(countMonth), 24)
    # titles = []
    # for i in major_ticks:
    #     titles.append(str(labels[int(i / 3)]))
    # plt.xticks(major_ticks, titles, rotation=45)
    # plt.bar(countMonth, num_posts, width, color="blue")
    # plt.ylabel('Number of posts', size=30, **hfont)
    # plt.tick_params('x', labelsize=25)
    # plt.tick_params('y', labelsize=25)
    # plt.subplots_adjust(left=0.13, bottom=0.25)
    # plt.grid(True)
    # plt.show()
