import datetime as dt
import pandas as pd
import requests
import request_server_data as rsd
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

    forumsData = pd.read_csv('F:/Github/DarkWeb_Influence/darkweb_data/12_19/HackingPosts_new.csv', encoding = "ISO-8859-1")
    forumsData.columns = ['idx', 'boardsName', 'forumsId', 'language', 'postContent',
                                     'postCve', 'postMs', 'postedDate', 'postsId',
                                     'scrapedDate', 'topicId', 'topicsName', 'usersId']
    forumsData = forumsData.drop_duplicates(subset=['boardsName', 'forumsId', 'language', 'postContent'
                                                    , 'postCve', 'postMs', 'postedDate', 'postsId',
                                                    'scrapedDate', 'topicId', 'topicsName', 'usersId'])
    userIDs = forumsData['usersId'].unique()
    # forumsData['scrapedDate'] = pd.to_datetime(forumsData['scrapedDate'], format="%m/%d/%Y")
    # print(forumsData.ix[:,:100])
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

        fPosts = forumsData[forumsData['forumsId'] == fId]
        users_list[fId] = fPosts['usersId']
        users_list[fId] = list(set(users_list[fId]))

    for fId in users_list:
        for f2 in users_list:
            if f2 == fId:
                continue
            commonUsers = list(set(users_list[fId]).intersection(set(users_list[f2])))
            print(len(commonUsers))

    exit()

        datesPosted = list(fPosts['postedDate'])
        x = collections.Counter(datesPosted)
        l = range(len(x.keys()))

        first_time = datesPosted[0]
        rt_time = datetime.datetime.strptime(first_time, '%Y-%m-%d')
        first_time = time.mktime(rt_time.timetuple())

        users_cum = []
        time_points = []
        num_users = []
        cnt_row = 0

        dateUser_Seen = {}
        act_time = {}
        social_degree = {}
        posts_user = {}

        # if fId == 45:
        #     continue
        for id, row in fPosts.iterrows():
            # print(cnt_row)
            cnt_row += 1
            # if cnt_row > 200:
            #     break
            rt_time = datetime.datetime.strptime(row['postedDate'], '%Y-%m-%d')
            cur_time = time.mktime(rt_time.timetuple())
            if row['usersId'] not in dateUser_Seen:
                dateUser_Seen[row['usersId']] = cur_time
                posts_user[row['usersId']] = 1
                continue

            diff = int(int(cur_time - dateUser_Seen[row['usersId']]) / (60*60*24))
            posts_user[row['usersId']] += 1
            if diff not in social_degree:
                social_degree[diff] = []
                act_time[diff] = []
            act_time[diff].append(pd.to_datetime(row['postedDate']))
            social_degree[diff].append(posts_user[row['usersId']])

        max_time = max(social_degree.keys())
        norm_times = []
        degree_final = []
        for diff in social_degree:
            if diff > 365:
                break
            social_degree[diff] = np.mean(social_degree[diff])
            degree_final.append(np.mean(social_degree[diff]))
            norm_times.append(diff)
        Y = degree_final
        X = norm_times
        # if row['usersId'] not in users_cum:
        #     users_cum.append(row['usersId'])
        # num_users.append(len(users_cum))
        # rt_time = datetime.datetime.strptime(row['postedDate'], '%Y-%m-%d')
        # cur_time = time.mktime(rt_time.timetuple())
        # diff = int(int(cur_time - first_time) / (60*60*24))
        # time_points.append(diff)

        # plt.figure(figsize=(20, 12))
        [y, x] = zip(*sorted(zip(Y, X), key=lambda x: x[1]))
        plt.plot(x, y, linewidth=1.5, color=next(colors), label='Forum ' + str(fId))
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.ylabel('Mean number of posts by users', fontsize=20)
        plt.xlabel('Number of days after first appearance', fontsize=20)
        plt.title('User-posting trend', size=20)
        # plt.xticks(np.arange(0, max(x), 8000))
        #plt.xticks(np.arange(0,40000,5000))
        # plt.tick_params(axis='x', labelsize=50)
        # plt.tick_params(axis='y', labelsize=50)
        plt.tick_params('x', labelsize=20)
        plt.tick_params('y', labelsize=20)
        #plt.title('Cascade lifecyle')
        #plt.savefig(newpath + '/logistic_curve.png')
        # plt.subplots_adjust(left=0.16, bottom=0.16)
        plt.show()
        plt.close()
        # plt.close()
        # labels = []
        # keys = []
        # values = []
        # sortedDates = sorted(x.items())
        # for d, c in sortedDates:
        #     keys.append(d)
        #     values.append(c)
        # major_ticks = np.arange(0, len(keys), 3)
        # for i in major_ticks:
        #     labels.append(keys[i])
        # plt.close()
        # plt.bar(l, x.values(), align='center')
        # plt.xticks(major_ticks, labels, rotation=45)
        # # n, bins, patches = plt.hist(datesPosted, facecolor='g')
        # # plt.xlabel('Length of the time series', size=30)
        # # plt.ylabel('Frequency', size=30)
        # # # plt.title('Histogram of')
        # plt.xlabel('Dates posted', size=20)
        # plt.ylabel('Count of posts', size=20)
        # plt.title('Timeline of forum posts: Forum ' + str(fId), size=20)
        # plt.grid(True)
        # plt.xticks(size=10)
        # plt.yticks(size=10)
        # # plt.ylim([0, 2000])
        # # file_save = dir_save + '/' + 'count_motif_' + str(m) + '.png'
        # # plt.savefig(file_save)
        # plt.show()
        # plt.close()
    exit()

    # analysis forum wise - postids are unique
    for fId, cntPosts in topForums:
        edges_forum = []
        fPosts = forumsData[forumsData['forumsId'] == fId]
        usersSeen = []
        cnt_idx = 0
        # row_prev = pd.DataFrame()
        for f_idx, row in fPosts.iterrows():
            if cnt_idx == 0:
                row_prev = row
                cnt_idx += 1
                continue
            cnt_idx += 1
            if cnt_idx > 1000:
                break
            print(row['postedDate'], row_prev['postedDate'])
            if row['postedDate'] != row_prev['postedDate']:
                row_prev = row
                continue
            user_cur = row['usersId']
            user_prev = row_prev['usersId']
            if user_cur != user_prev:
                edges_forum.append((user_prev, user_cur))
        G = nx.Graph()
        edges_forum = list(set(edges_forum))
        G.add_edges_from(edges_forum)
        nx.draw(G)
        nx.draw(G, pos=nx.spring_layout(G))  # use spring layout
        plt.show()
        degrees = G.degree()  # dictionary node:degree
        # in_values = sorted(set(degrees.values()))
        # list_degree_values = list(degrees.values())
        #
        # in_hist = [list_degree_values.count(x) for x in in_values]
        # plt.figure()
        # plt.loglog(in_values, in_hist, 'ro', basex=2, basey=2)
        # # plt.xlim([1, 2**18])
        # plt.xlabel('Degree (log)', size=20)
        # plt.ylabel('Number of vertices(log)', size=20)
        # plt.title('Users network')
        # plt.tick_params('y', labelsize=20)
        # plt.tick_params('x', labelsize=20)
        # plt.grid(True)
        # plt.show()
        exit()

        # postIds = fPosts.postsId.unique()
        # for pid in postIds:
        #     threads = fPosts[fPosts['postsId'] == pid]
        #     users_t = list(set(threads['usersId']))
        #     if len(threads) > 1:
        #         print(threads)
        #         exit()
            # edges = list(itertools.combinations(users_t, 2))
            # print(edges)

    # Check if users are common across forums


            # print(forumsDateSorted.ix[:10, :])

    # for i, x in enumerate(userIDs):
    #     up = rsd.getUsersForums(x, limNum=10000)

    # limit = 100000
    # data_count = 0
    # # Hacking Posts Statistics
    # fileName = 'HackingPosts'
    # start_date = dt.datetime.strptime('2016-08-01', '%Y-%m-%d')
    # end_date = dt.datetime.strptime('2016-11-01', '%Y-%m-%d')
    # hp_df = pd.DataFrame()
    # while data_count < limit:
    #     hp = getHackingPosts(fromDate=start_date, toDate=end_date, start=data_count, limNum=10000)
    #     hpList = {}
    #     for hp_idx in range(len(hp)):
    #         hpList = hp[hp_idx]
    #         item_df = pd.DataFrame(hpList, index=[data_count+hp_idx])
    #         hp_df = hp_df.append(item_df)
    #     data_count += 10000
    #     print(data_count)
    # hp_df.to_csv(fileName + '.csv')

