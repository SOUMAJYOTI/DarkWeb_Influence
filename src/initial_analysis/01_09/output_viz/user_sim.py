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
import os
import pickle

def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

if __name__ == "__main__":
    months_users_topics = pickle.load(open('m_u_t.pickle', 'rb'))
    impForums = [40, 48, 54, 84, 89, 125]

    forumsData = pd.read_csv('F:/Github/DarkWeb_Influence/darkweb_data/1_3/Forumdata_40.csv', encoding = "ISO-8859-1")
    forumsData.columns = ['idx', 'boardsName', 'forumsId', 'language', 'postContent',
                                     'postCve', 'postMs', 'postedDate', 'postsId',
                                     'scrapedDate', 'topicId', 'topicsName', 'usersId']
    forumsData = forumsData.drop_duplicates(subset=['boardsName', 'forumsId', 'language', 'postContent'
                                                    , 'postCve', 'postMs', 'postedDate',
                                                    'scrapedDate', 'topicId', 'topicsName', 'usersId'])
    userIDs = forumsData['usersId'].unique()

    forumsData = forumsData.sort_values(by='postedDate')

    # Networks by time
    # Edges by sequence on same day

    fId = 40
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
    num_users = []
    sim_months_topics = {}
    for id, row in fPosts.iterrows():
        cur_date = row['postedDate']
        rt_time = datetime.datetime.strptime(cur_date, '%Y-%m-%d')
        cur_date = time.mktime(rt_time.timetuple())

        if rt_time <= lastTime:
            continue
        num_months += 3
        d = 'month_' + str(num_months)
        if d not in months_users_topics:
            continue

        sim_months_topics[d] = {}

        users_topics = months_users_topics[d]
        monthDate = add_months(rt_time, 3)
        prevPost_count = len(rel_posts)
        rel_posts = fPosts[pd.to_datetime(fPosts['postedDate']) <= monthDate]
        rel_posts = rel_posts[pd.to_datetime(rel_posts['postedDate']) > lastTime]
        num_posts.append(len(rel_posts))

        # print(len(rel_posts))
        # if len(rel_posts) > 1:
        usersRel = list(rel_posts['usersId'])

        usersSet = list(set(usersRel))
        num_users.append(len(usersSet))
        userPosts_dict = {}
        for idx in range(len(usersSet)):
            fPosts_user = rel_posts[rel_posts['usersId'] == usersSet[idx]]
            userPosts_dict[usersSet[idx]] = len(fPosts_user)

        sorted_users = sorted(userPosts_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_users = sorted_users[:10]

        clist = ['users', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6',
                                              'Topic 7', 'Topic 8', 'Topic 9', 'Topic 10', 'Topic 11', 'Topic 12',
                                              'Topic 13', 'Topic 14', 'Topic 15']
        user_topic_df = pd.DataFrame(columns=clist)

        for u, val in sorted_users:
            df_list = []
            df_list.append('user_' + str(u))
            topic_vals = users_topics['user_' + str(u)]
            t_list = []
            sorted_t = sorted(topic_vals.items(), key=operator.itemgetter(0), reverse=True)
            sum_t = 0
            for t, v in sorted_t:
                # print(topic_vals[t])
                df_list.append(topic_vals[t])

            user_topic_df = user_topic_df.append(pd.DataFrame([df_list], columns=clist), ignore_index=True)

        csv_path = 'user_topic_dist_csv/all' + ".csv"
        # user_topic_df.to_csv(csv_path)

        user_topic_df.to_csv(csv_path, mode='a')

        # with open(csv_path, 'a') as f:
        #     user_topic_df.to_csv(f)

        # for idx in range(len(usersRel)-1):
        #     if usersRel[idx] == usersRel[idx+1]:
        #         continue
        #     edges_forum.append((usersRel[idx], usersRel[idx+1]))
        #
        # # print(edges_forum)
        # # if len(edges_forum) > 1:
        # edges_wt_forums = []
        # countDict = Counter(edges_forum)
        # items = countDict.most_common()
        # for (a, b), c in items:
        #     edges_wt_forums.append((a, b, c))
        #
        # G = nx.DiGraph()
        # G.add_weighted_edges_from(edges_wt_forums)
        # print(G.number_of_nodes(), G.number_of_edges())
        #
        # # nx.draw(G)
        # # nx.draw(G, pos=nx.spring_layout(G))  # use spring layout
        # # plt.show()
        # # plt.close()
        #
        # # largest_comp = max(nx.strongly_connected_components(G), key=len)
        # # nodes = list(largest_comp)
        # # new_edges = []
        # # for a,b,c in edges_wt_forums:
        # #     if a not in nodes or b not in nodes:
        # #         continue
        # #     new_edges.append((a,b,c))
        # # G = nx.DiGraph()
        # # G.add_weighted_edges_from(new_edges)
        # # val = nx.diameter(G)
        #
        # val = nx.out_degree_centrality(G)
        # sorted_deg = sorted(val.items(), key=operator.itemgetter(1), reverse=True)
        # top_count = int (0.1 * len(val))
        # top_users = sorted_deg[:]
        #
        # for user, deg in top_users:
        #     u_1 = 'user_' + str(user)
        #     if u_1 not in users_topics:
        #         continue
        #     t_1 = users_topics[u_1]
        #     sum_t = 0
        #     for i in t_1:
        #         sum_t += t_1[i]
        #     if sum_t != 0:
        #         for i in t_1:
        #             t_1[i] /= sum_t
        #     diff = 0
        #     for n in G.neighbors(user):
        #         u_2 = 'user_' + str(n)
        #         if u_2 not in users_topics:
        #             continue
        #         t_2 = users_topics[u_2]
        #         sum_t = 0
        #         for i in t_2:
        #             sum_t += t_2[i]
        #         if sum_t != 0:
        #             for i in t_2:
        #                 t_2[i] /= sum_t
        #
        #         for topic in users_topics[u_2]:
        #             if topic not in sim_months_topics[d]:
        #                 sim_months_topics[d][topic] = []
        #             diff = abs(t_1[topic] - t_2[topic])
        #             sim_months_topics[d][topic].append(1-diff)
            # print(diff)


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
        # measure_tnw.append(np.mean(list(val.values())))

        countMonth.append(num_months)
        datesSeen.append(row['postedDate'])
        # countDict_global = countDict
        lastTime = pd.to_datetime(add_months(rt_time, 1))
        labels.append(monthDate)
        edges_forum = []
        print(lastTime)

    pickle.dump(sim_months_topics, open('sim_months_topics_1.pickle', 'wb'))
        # if lastTime > pd.to_datetime('2012-01-01'):
        #     break

        # if num_months > 10:
        #     break

    # print(labels)
    # hfont = {'fontname': 'Arial'}
    # plt.figure(figsize=(12, 8))
    # major_ticks = np.arange(min(countMonth), max(countMonth), 24)
    # titles = []
    # for i in major_ticks:
    #     titles.append(str(labels[int(i/3)])[:10])
    # plt.xticks(major_ticks, titles, rotation=45)
    # plt.plot(countMonth, measure_tnw, linewidth=1.5, label='Forum ' + str(fId))
    # plt.grid(True)
    # plt.legend(fontsize=20)
    # plt.ylabel('Out-degrees', fontsize=30, **hfont)
    #
    # plt.tick_params('x', labelsize=25)
    # plt.tick_params('y', labelsize=25)
    #
    # plt.subplots_adjust(left=0.13, bottom=0.25)
    #
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(12, 8))
    # width=1
    # major_ticks = np.arange(min(countMonth), max(countMonth), 10)
    # titles = []
    # for i in major_ticks:
    #     titles.append(str(labels[int(i / 3)]))
    # plt.xticks(major_ticks, titles, rotation=45)
    # plt.bar(countMonth, num_users, width, color="blue")
    # plt.ylabel('Number of users', size=30, **hfont)
    # plt.tick_params('x', labelsize=25)
    # plt.tick_params('y', labelsize=25)
    # plt.subplots_adjust(left=0.13, bottom=0.25)
    # plt.grid(True)
    # plt.show()
