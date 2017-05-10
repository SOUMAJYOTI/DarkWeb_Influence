import datetime as dt
import pandas as pd
import requests

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

def getHackingThreads(fid, start =0, limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+ "&start=" + str(start) + "&forumsId=" + str(fid)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getUserNet(uid, start = 0, limNum=0, threshold=3):
    url = "https://apigargoyle.com/GargoyleApi/generateUserNet?limit="+str(limNum)+"&start=" + str(start) + "&uid="+str(uid) + "&threshold=" + str(threshold)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']


if __name__ == "__main__":
    # forumsData = pd.read_csv('darkweb_data/12_19/HackingPosts_new.csv', encoding = "ISO-8859-1")
    # userIDs = forumsData['usersId'].unique()

    # for i, x in enumerate(userIDs):
    #     up = rsd.getUsersForums(x, limNum=10000)
    #
    # print(len(userIDs))
    limit = 100000
    data_count = 0
    start_date = dt.datetime.strptime('2014-01-01', '%Y-%m-%d')
    end_date = dt.datetime.strptime('2016-12-31', '%Y-%m-%d')

    # Hacking Posts Statistics
    fileName = 'Forumdata_40'

    fid = 40
    hp_df = pd.DataFrame()
    while data_count < limit:
        print('hello')
        hp = getHackingThreads(fid, data_count, limNum=10000)
        if not hp:
            break
        # hp = getHackingPosts(start_date, end_date, start=data_count, limNum=10000)
        # hpList = {}
        # for hp_idx in range(len(hp)):
        #     hpList = hp[hp_idx]
        #     item_df = pd.DataFrame(hpList, index=[data_count+hp_idx])
        #     hp_df = hp_df.append(item_df)
        data_count += 10000
        print(data_count)
    # hp_df.to_csv(fileName + '.csv')

