import matplotlib.pyplot as plt
import csv
import numpy as np

def format_data_for_grouped_barchart(filename):
    
    groupwise_sentiments = {}
    # Syntax: group => [positive,negative,neutral]
    groupwise_sentiments['Health'] = [0,0,0]
    groupwise_sentiments['Economy'] = [0,0,0]
    groupwise_sentiments['Culture'] = [0,0,0]
    groupwise_sentiments['Social'] = [0,0,0]
    
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter = ',')
        for row in lines:
            if row[12] in ['Health', 'Economy', 'Culture', 'Social']:
                if row[11] == 'negative':
                    groupwise_sentiments[row[12]][1] += 1
                elif row[11] == 'positive':
                    groupwise_sentiments[row[12]][0] += 1
                else:
                    groupwise_sentiments[row[12]][2] += 1
                    
    return groupwise_sentiments
    
    

def grouped_barchart( filename):
    data = format_data_for_grouped_barchart( filename)
    x = np.arange(4)
    # y1 = [34, 56, 12, 89, 67]
    # y2 = [12, 56, 78, 45, 90]
    # y3 = [14, 23, 45, 25, 89]
    y1 = []
    y2 = []
    y3 = []
    
    for key in data:
        y1.append(data[key][0])
        y2.append(data[key][1])
        y3.append(data[key][2])
    
    width = 0.2

    # plot data in grouped manner of bar type
    plt.bar(x-0.2, y1, width, color='cyan')
    plt.bar(x, y2, width, color='orange')
    plt.bar(x+0.2, y3, width, color='green')
    # plt.xticks(x, ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'])
    plt.xticks(x, data.keys())
    plt.xlabel("Tweet categories")
    plt.ylabel("Tweet counts")
    plt.legend(['Positive tweets', 'Negative tweets', 'Neutral tweets'])
    plt.show()


def visualise_pie(filename):
    X = ['Positive tweets', 'Negative tweets', 'Neutral tweets']
    Y = []
    neutral_count = 0
    positive_count = 0
    negative_count = 0
    
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter = ',')
        for row in lines:
            tpLen = len(row)
            if row[tpLen-1] == 'negative':
                negative_count += 1
            elif row[tpLen-1] == 'positive':
                positive_count += 1
            else:
                neutral_count += 1
    
    
    Y.append(positive_count)
    Y.append(negative_count)
    Y.append(neutral_count)
    
    plt.pie(Y,labels = X,autopct = '%.2f%%')
    plt.title('Sentiments analysis of tweets', fontsize = 20)
    plt.figure(1)
    plt.show()


def visualise_bar(filename):
    x = ['Economy', 'Social', 'Culture', 'Health']
    y = [0,0,0,0]
    
    with open(filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        
        for row in plots:
            if row[10] != 'None':
                if row[10] == 'Economy':
                    y[0] += 1
                elif row[10] == 'Social':
                    y[1] += 1
                elif row[10] == 'Culture':
                    y[2] += 1
                elif row[10] == 'Health':
                    y[3] += 1
    
    plt.bar(x, y, color = 'g', width = 0.72, label = "Tweets")
    plt.xlabel('Category')
    plt.ylabel('Tweet counts')
    plt.title('Visual representation of category-wise Tweets')
    plt.figure(1)
    plt.show()
    

visualise_pie('sentimentanalysis.csv')
visualise_bar('clusterclassification.csv')
grouped_barchart('sentiment_and_cluster.csv')