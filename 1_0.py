import re
from pandas.core.frame import DataFrame 
import tweepy 
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
nltk.download('punkt')   
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tweepy import OAuthHandler 
from textblob import TextBlob
from bs4 import BeautifulSoup

consumer_key="FGiELjdRAiyfaag6nOlbJVRIT"
consumer_secret="jZh2gxfucS3QTxsHQwC86BCidHWOKayBreLfgSwi26iZ8um7Ql"
access_token="1417382105479225346-QMcDG6ncTzNgcKz2Xk2WCiJ4D3RMOM"
access_token_secret="PdDyjlGFHZ6KOiWBX07xk24087XrugzJ3S7frV7XGOVrb"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def sentiment_analysis(search_query,numtweets):
    tweets=tweepy.Cursor(api.search,q=search_query+'-filter:retweets',lang="en",since="2020-10-13").items(numtweets)
    #tweetsfinal=[]
    polarity=0

    positive=0
    negative=0
    neutral=0

    #print(tweets)

    for tweet in tweets:
        textl1=tweet.text
        textl2=''.join([c for c in textl1 if ord(c) < 128])#removing weird text
        textl3=BeautifulSoup(textl2,'lxml').get_text()#removes html tags
        textl4 = ' '.join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)", " ", textl3).split())#removing mentions and hashtags
        textl5 = ' '.join(re.sub("http://\S+|https://\S+", " ", textl4).split())#removing links
        textl6 = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", textl5).split())#removing punctuations
        textl7 = textl6.lower()#converting to lower case
        #if final_text.startswith(' @'):
        #    position=final_text.index(':')
        #    final_text =final_text[position+2:]
        #if final_text.startswith('@'):
        #    position=final_text.index(' ')
        #    final_text =final_text[position+2:]
        #print(final_text.encode("utf-8"))
        analysis = TextBlob(textl7)      #main sentiment analysis
        tweet_polarity=analysis.polarity#main sentiment analysis
        
        if(tweet_polarity>0.00):
            positive+=1
            tweetsfinal.append((search_query,tweet,textl1,textl2,textl3,textl4,textl5,textl6,textl7,tweet_polarity,'positive'))
        elif(tweet_polarity<0.00):
            negative+=1
            tweetsfinal.append((search_query,tweet,textl1,textl2,textl3,textl4,textl5,textl6,textl7,tweet_polarity,'negative'))
        elif(tweet_polarity==0.00):
            neutral+=1
            tweetsfinal.append((search_query,tweet,textl1,textl2,textl3,textl4,textl5,textl6,textl7,tweet_polarity,'neutral'))
        polarity += analysis.polarity
    #print(final_text)
    print(polarity)
    print(f'Amount of positive tweets:{positive}')
    print(f'Amount of negative tweets:{negative}')
    print(f'Amount of neutral tweets:{neutral}')
    return tweetsfinal
    
tweetsfinal=[]
#search_query="death"
numtweets=10

woeid = 2295424
  
# fetching the trends
trends = api.trends_place(id = woeid, exclude = "hashtags")
  
# printing the information
#print("The top trends for the location are :")
  
for value in trends:
    for trend in value['trends']:
        search_query=trend['name']
        tweetsfinal=sentiment_analysis(search_query,numtweets)
        


df=pd.DataFrame(tweetsfinal,columns=['topic','Level 0','Level 1(Readable tweet)','Level 2(Remvoving weird data)','Level 3(removing html tags)','Level 4(removing hashtags and mentions)','Level 5(removing links)','Level 6(removing punctuations)','final tweets after procesing','polarity','sentiment analysis'])
#df=df.groupby(['topic','initial tweets','final tweets after procesing','polarity','sentiment analysis']).apply(lambda x: x.sort_values('topic')).reset_index(drop=True)
#df=df.drop_duplicates(subset=['topic','initial tweets','final tweets after procesing','polarity','sentiment analysis'],keep='first')
df=df.drop_duplicates(subset='final tweets after procesing')
df.to_csv('projectsample_1.csv',index=False)