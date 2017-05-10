import requests
import datetime as dt
import csv

def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum

def strToDt(dateStr):
    lst = dateStr.split('-')
    year = int(lst[0])
    month = int(lst[1])
    day = int(lst[2])
    return dt.date(year,month,day)

def getHackingPosts(fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingPosts?limit="+str(limNum)+"&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getHackingItems(fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getHackingItems?limit="+str(limNum)+"&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getClusterStatistics(clusterName='', limNum=0):
    if clusterName == '':
        url = "https://apigargoyle.com/GargoyleApi/getClusterStatistics?limit=" + str(
            limNum)
    else:
        url = "https://apigargoyle.com/GargoyleApi/getClusterStatistics?limit=" + str(limNum) + "&clusterName=" + clusterName
    headers = {"userId": "labuser", "apiKey": "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getVulnerabilityInfo(fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getVulnerabilitiyInfo?limit="+str(limNum)+"&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()

def getZeroDayProducts(fromDate=dt.date.today(), toDate=dt.date.today(), limNum=0):
    url = "https://apigargoyle.com/GargoyleApi/getZeroDayProducts?limit="+str(limNum)+"&from="+dateToString(fromDate)+"&to="+dateToString(toDate)
    headers = {"userId" : "labuser", "apiKey" : "e47545c2-7222-44ed-a486-73ee8905582b"}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def getCallInfo(callName,filePath=r"../darkweb_data/DailyStatus-", endFn=".csv",day=dt.date.today()):
    if callName == "getHackingPosts":
        call = getHackingPosts
        siteKey = 'forumsId'
        postedDateKey='postedDate'
    if callName == "getHackingItems":
        call = getHackingItems
        siteKey = 'marketplaceId'
        postedDateKey='postedDate'
    if callName == "getVulnerabilityInfo":
        call = getVulnerabilityInfo
        siteKey = 'forumsId'
        postedDateKey='darkwebDate'
    if callName == "getZeroDayProducts":
        call = getZeroDayProducts
        siteKey = 'marketplaceId'
        postedDateKey='postedDate'

    fn = filePath+callName+"-"+str(day)+endFn
    return callName,call,siteKey,postedDateKey,fn

def retSitesVisited(queryRes, siteKey = 'forumsId'):
    siteList = []
    for item in queryRes:
        if not item[siteKey] in siteList:
            siteList.append(item[siteKey])
    return siteList

def retDateCnts(queryRes, lastDay = dt.date.today(), postedDateKey='postedDate', per = 7):
    perCnt = 0
    noneCnt = 0
    for item in queryRes:
        if item[postedDateKey] == None:
            noneCnt +=1
        elif strToDt(item[postedDateKey])>lastDay - dt.timedelta(per):
            perCnt += 1
    return perCnt, noneCnt

def movingAverage(day, dictIndexedByDay, field, per=7):
    res = 0.0
    for i in range(0,per):
        j = day-dt.timedelta(i)
        res += float(dictIndexedByDay[j][field])/float(per)
    return res

def listInLastPer(day, dictIndexedByDay, field, per=7):
    res = []
    for i in range(0,per):
        j = day-dt.timedelta(i)
        for item in dictIndexedByDay[j][field]:
            if not item in res:
                res.append(item)
    return res

def createTimeDict(day=dt.date.today(), span=30, per=7, getReq=getHackingPosts, siteKey = 'forumsId', postedDateKey='postedDate'):
    res = {}
    for i in range(0,span+per):
        j=day-dt.timedelta(i)
        queryRes=getReq(j,j,0)
        count = len(queryRes)
        perCnt, noneCnt = retDateCnts(queryRes,j,postedDateKey, per)
        siteList = retSitesVisited(queryRes,siteKey)
        res[j]=(count,perCnt,noneCnt,siteList)
    return res,0

def createMaDict(dictIndexedByDay,per=7):
    res = {}
    for day in dictIndexedByDay:
        if day-dt.timedelta(per) in dictIndexedByDay:
            countMa = movingAverage(day,dictIndexedByDay,0,per)
            perCountMa = movingAverage(day,dictIndexedByDay,1,per)
            noneCountMa = movingAverage(day,dictIndexedByDay,2,per)
            siteCntPer = len(listInLastPer(day,dictIndexedByDay,3,per))
            res[day]=(countMa,perCountMa,noneCountMa,siteCntPer)
    return res

def createLine(call,day,dictIndexedByDay,maDict):
    line = str(call)+","+str(day)+","
    for i in range(0,3):
        line += str(dictIndexedByDay[day][i])+","
    line+=str(len(dictIndexedByDay[day][3]))+","
    for i in range(0,4):
        line += str(maDict[day][i])+","
    line+="\n"
    return line

def createLineList(call,dictIndexedByDay,maDict):
    lineList = []
    for day in maDict:
        lineList.append((day,createLine(call,day,dictIndexedByDay,maDict)))
    lineList.sort(reverse=True)
    return lineList

def firstLine():
    return "call,day,recordsThatDay,recordsThatDayPostedLastWk,recordsThatDayNoDate,sitesThatDay,wkMaRecords,wkMaPostedLastWk,wkMaNoDate,sitesPastWeek\n"

def crFile(callName, day, filePath, per):
    callName,call,siteKey,postedDateKey,fn = getCallInfo(callName,filePath=filePath, endFn=".csv",day=day)

    timeDictRes = createTimeDict(day=day, span=30, per=per, getReq=call, siteKey = siteKey, postedDateKey=postedDateKey)
    timeDict = timeDictRes[0]
    maDict = createMaDict(timeDict,per=per)
    lineList=createLineList(callName,timeDict,maDict)
    with open(fn,'w') as fh:
        fh.write(firstLine())
        for line in lineList:
            fh.write(line[1])
    #return qRes  




        
                             
