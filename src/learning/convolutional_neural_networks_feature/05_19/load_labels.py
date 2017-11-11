import pickle
from textblob import TextBlob as tb
from tf_idf import tfidf

if __name__ == "__main__":
    file_data = open('../../../darkweb_data/05/5_19/simple.txt', 'rb')

    cnt_line = 0
    for line in file_data:
        line = line.decode("utf-8")
        line = line[1:len(line)-2]
        items = line.split(', ')
        print(items)
        items = items[2].lstrip()
        items = items[1:len(items)-1]
        print(items)
        cnt_line += 1
        if cnt_line > 10:
            exit()

    forumsData = pd.read_csv('../../../darkweb_data/05/5_15/Forum_40_labels.csv', encoding="ISO-8859-1")
