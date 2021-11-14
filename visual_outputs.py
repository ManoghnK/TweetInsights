import matplotlib.pyplot as plt
import csv

def visualise_pie(filename):
    X = ['Positive', 'Negative', 'Neutral']
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
    plt.title('Sentiments of users', fontsize = 20)
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