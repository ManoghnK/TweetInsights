import re
from pandas.core.frame import DataFrame 
import tweepy 
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
import string
import collections
import matplotlib.pyplot as plt
import en_core_web_sm
from wordcloud import WordCloud,STOPWORDS
nltk.download('punkt')   
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tweepy import OAuthHandler 
from textblob import TextBlob
from bs4 import BeautifulSoup
from string import punctuation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


consumer_key="FGiELjdRAiyfaag6nOlbJVRIT"
consumer_secret="jZh2gxfucS3QTxsHQwC86BCidHWOKayBreLfgSwi26iZ8um7Ql"
access_token="1417382105479225346-QMcDG6ncTzNgcKz2Xk2WCiJ4D3RMOM"
access_token_secret="PdDyjlGFHZ6KOiWBX07xk24087XrugzJ3S7frV7XGOVrb"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

nlp = en_core_web_sm.load() 
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation) #already taken care of with the cleaning function.
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()




def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
       if i.lower() not in stop:
        word = lemmatizer.lemmatize(i)
        final_text.append(word.lower())
    return " " .join(final_text)

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

economy_related_words='''agriculture infrastructure capitalism trading service sector technology  economical supply 
                          industrialism efficiency frugality retrenchment downsizing   credit debit value 
                         economize   save  economically
                         economies sluggish rise   rising spending conserve trend 
                         low-management  decline   industry impact poor  
                            profession    surplus   fall
                         declining  accelerating interest sectors balance stability productivity increase rates
                            pushing expanding stabilize  rate industrial borrowing struggling
                           deficit predicted    increasing  data
                          economizer analysts investment market-based economy   debt free enterprise
                         medium  exchange metric savepoint scarcity capital bank company stockholder fund business  
                         asset treasury tourism incomes contraction employment jobs upturn deflation  macroeconomics
                         bankruptcies exporters hyperinflation dollar entrepreneurship upswing marketplace commerce devaluation 
                         quicksave deindustrialization stockmarket reflation downspin dollarization withholder bankroll venture capital
                         mutual fund plan economy mortgage lender unemployment rate credit crunch central bank financial institution
                         bank rate custom duties mass-production black-market developing-countries developing economic-growth gdp trade barter 
                         distribution downturn economist
                         business profits loss stocks shares start-up entrepreneur chairman CEO '''

social_related_words='''sociable, gregarious societal friendly society socialization political  sociality 
                        interpersonal  ethnic socially party welfare public community socialist societies development
                            network humans socialism collective personal corporation social constructivism
                        relations volition citizenship brute   attitude rights socio 
                        socioeconomic ethics civic communal marital  sociale socialized communities     
                         policy   unions        
                        institutions values     governmental   organizations jamboree 
                         festivity    fairness  support  care  
                         sides   activism     unsocial psychosocial 
                        socializing psychological distributional  demographic  participation reunion 
                        partygoer partyism festive power network gala housewarming celebration counterparty   social-war
                        particularist interactional ideational asocial 
                        instagram facebook whatsapp twitter internet email politics government president prime-minister minister law court 
                        lawyer sue case verdict bill crime police assault '''

culture_related_words='''ethnicity heritage modernity spirituality marxismmaterial culture 
                           ethos nationality humanism romanticism civilisation traditionalism genetics
                        kinship heredity marriage   indigenous  archeology  acculturate  
                       ontogenesis viniculture modern clothes     rooted 
                       cicero societies history roots influence geography historical folk origins 
                       phenomenon teleology ancient aspects perspective liberalism nowadays community style unique prevalent describes 
                         today  origin   modernity beliefs  genre barbarian ethnic 
                       colonization cultural universal organization western-civilization structuralism  culture 
                       heathen pagan transculturation culture peasant classicist nativism anarchy ungrown philosophic cult  
                       consciousness islamist bro-culture evolve cultic diaspora aftergrowth native cultural-relativism  
                       mongolian cosmopolitan epistemology lifestyles diversity chauvinism westernization materialism vernacular 
                       homogeneity otherness holism tusculanae disputationes primitivism superficiality hedonism discourse
                       puritanism modernism intellectualism  exclusiveness elitism  colonialism  
                       pentecostalism paganism nationwide expansion rural  auxesis kimono 
                       culturize alethophobia nettlebed japanification  dongyi clannishness insularity hybridity
                       westernisation foreignness worldview exclusionism enculturation ethnocentrism  confucianist vulgarization
                       shintoism  westernism denominationalism    deracination
                        eurocentrism  cosmologies  emotiveness bohemianism territorialism
                       philosophical-doctrine ethnic minority social-darwinism  theory cultural evolution belief systemfolk music 
                       traditional art house karl-marx   theorymedia  
                       film-theory art history museum studies cultural artifact 
                       sports cricket football basketball hockey chess ball goal foul green-card red-card yellow-card umpire refree
                       theatre movie cinema music director drama play art hero heroine villian stadium recreation parks malls'''

health_related_words='''disease obesity world health organization medicine nutrition well-being exercise welfare wellness health care public health 
                     nursing stress safety hygiene research social healthy condition aids epidemiology healthiness wellbeing
                     care illness medical dieteducation infectious disease environmental healthcare physical fitness hospitals 
                     health care provider doctors healthy community design insurance sanitation human body patient mental health
                      medicare agriculture health science fitnesshealth policy  weight loss physical therapy psychology pharmacy
                     metabolic organism human lifestyle status unhealthy upbeat vaccination sleep condom alcohol smoking water family
                     eudaimonia eudaemonia air house prevention genetics public families poor needs treatment communicable disease 
                     study protection malaria development food priority management healthful mental provide department administration
                     programs help assistance funding environment improving emergency need program affected schools private mental illness 
                     treat diseases preparedness perinatal fertility sickness veterinary sanitary pharmacists behavioral midwives
                     gerontology infertility hospitalization midwifery cholesterol childcare pediatrician pediatrics medicaid asthma 
                     pensions sicknesses push-up physical education body-mass-index eat well gymnastic apparatus tune up good morning 
                     bathing low blood-pressure heart attack health club ride-bike you feel good eczema urticaria dermatitis sunburn overwork 
                     manufacturing medical sociology need exercise run covid vaccination death chicken-guinea thyroid typhoid blood swine-flu pharmacy medicine'''

economy=furnished(economy_related_words)
social=furnished(social_related_words)
culture=furnished(culture_related_words)
health=furnished(health_related_words)
string1=economy
print(string1)
words=string1.split()
#print(words)
economy=" ".join(sorted(set(words),key=words.index))
print(economy)

string1=social
words=string1.split()
social=" ".join(sorted(set(words),key=words.index))
social

string1=health
words=string1.split()
health=" ".join(sorted(set(words),key=words.index))
health

string1=culture
words=string1.split()
culture=" ".join(sorted(set(words),key=words.index))
culture


def cluster_classification(df):
    twitterfinalpostclassification=[]
    for i in range(df.shape[0]):
        s = jaccard_similarity(economy, df.iat[i,9])
        t = jaccard_similarity(social,  df.iat[i,9])
        u = jaccard_similarity(culture, df.iat[i,9])
        v = jaccard_similarity(health,  df.iat[i,9])
        f=max(s,t,u,v)
        if(f>0.5):
            if(f==s):
                twitterfinalpostclassification.append((df.iat[i,0],df.iat[i,1],df.iat[i,2],df.iat[i,3],df.iat[i,4],df.iat[i,5],df.iat[i,6],df.iat[i,7],df.iat[i,8],df.iat[i,9],'Economy',f))
            elif(f==t):
                twitterfinalpostclassification.append((df.iat[i,0],df.iat[i,1],df.iat[i,2],df.iat[i,3],df.iat[i,4],df.iat[i,5],df.iat[i,6],df.iat[i,7],df.iat[i,8],df.iat[i,9],'Social',f))
            elif(f==u):
                twitterfinalpostclassification.append((df.iat[i,0],df.iat[i,1],df.iat[i,2],df.iat[i,3],df.iat[i,4],df.iat[i,5],df.iat[i,6],df.iat[i,7],df.iat[i,8],df.iat[i,9],'Cultural',f))
            elif(f==v):
                twitterfinalpostclassification.append((df.iat[i,0],df.iat[i,1],df.iat[i,2],df.iat[i,3],df.iat[i,4],df.iat[i,5],df.iat[i,6],df.iat[i,7],df.iat[i,8],df.iat[i,9],'Health',f))
        else:
                twitterfinalpostclassification.append((df.iat[i,0],df.iat[i,1],df.iat[i,2],df.iat[i,3],df.iat[i,4],df.iat[i,5],df.iat[i,6],df.iat[i,7],df.iat[i,8],df.iat[i,9],'None',f))
           
        #print(tweet+":")
        #print(s)
    return twitterfinalpostclassification


def sentimentanalysis(df1):
    
    polarity=0

    positive=0
    negative=0
    neutral=0
    for i in range(df1.shape[0]):

        analysis = TextBlob(df1.iat[i,9])      #main sentiment analysis
        tweet_polarity=analysis.polarity#main sentiment analysis

        if(tweet_polarity>0.00):
            #positive+=1
            tweetsfinalpostsentiment.append((df1.iat[i,0],df1.iat[i,1],df1.iat[i,2],df1.iat[i,3],df1.iat[i,4],df1.iat[i,5],df1.iat[i,6],df1.iat[i,7],df1.iat[i,8],df1.iat[i,9],tweet_polarity,'positive'))
        elif(tweet_polarity<0.00):
            #negative+=1
            tweetsfinalpostsentiment.append((df1.iat[i,0],df1.iat[i,1],df1.iat[i,2],df1.iat[i,3],df1.iat[i,4],df1.iat[i,5],df1.iat[i,6],df1.iat[i,7],df1.iat[i,8],df1.iat[i,9],tweet_polarity,'negative'))
        elif(tweet_polarity==0.00):
            #neutral+=1
            tweetsfinalpostsentiment.append((df1.iat[i,0],df1.iat[i,1],df1.iat[i,2],df1.iat[i,3],df1.iat[i,4],df1.iat[i,5],df1.iat[i,6],df1.iat[i,7],df1.iat[i,8],df1.iat[i,9],tweet_polarity,'neutral'))
        #polarity += analysis.polarity
        #print(final_text)
    #print(polarity)
    #print(f'Amount of positive tweets:{positive}')
    #print(f'Amount of negative tweets:{negative}')
    #print(f'Amount of neutral tweets:{neutral}')
    
    df=pd.DataFrame(tweetsfinalpostsentiment,columns=['topic','Level 0','Level 1(Readable tweet)','Level 2(Remvoving weird data)','Level 3(removing html tags)','Level 4(removing hashtags and mentions)','Level 5(removing links)','Level 6(removing punctuations)','Level 7(Converting into lower case)','final tweets after procesing','polarity','sentiment analysis'])
    df=df.drop_duplicates(subset='final tweets after procesing')
    df.to_csv('sentimentanalysis.csv',index=False)
    return tweetsfinalpostsentiment



def processing(search_query,numtweets):
    #tweetsfinal=[]
    tweets=tweepy.Cursor(api.search,q=search_query+'-filter:retweets',lang="en",since="2020-10-13").items(numtweets)

    #print(tweets)

    for tweet in tweets:
        textl1=tweet.text
        textl2=''.join([c for c in textl1 if ord(c) < 128])#removing weird text
        textl3=BeautifulSoup(textl2,'lxml').get_text()#removes html tags
        textl4 = ' '.join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)", " ", textl3).split())#removing mentions and hashtags
        textl5 = ' '.join(re.sub("http://\S+|https://\S+", " ", textl4).split())#removing links
        textl6 = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", textl5).split())#removing punctuations
        textl7 = textl6.lower()#converting to lower case
        textl8=furnished(textl7)
        tweetsfinalpostprocessing.append((search_query,tweet,textl1,textl2,textl3,textl4,textl5,textl6,textl7,textl8))
    df1=pd.DataFrame(tweetsfinalpostprocessing,columns=['topic','Level0','Level1(Readable_tweet)','Level2(Remvoving_weird_data)','Level3(removing_html_tags)','Level4(removing_hashtags_and_mentions)','Level5(removing_links)','Level6(removing_punctuations)','Level7(Converting_into_lower_case)','final_tweets_after_procesing'])
    df1 = df1.dropna()
    df1=df1.drop_duplicates(subset='final_tweets_after_procesing')
    df1.to_csv('preprocessing.csv',index=False)
    tweetsfinalpostsentiment=sentimentanalysis(df1)
    twitterfinalpostclassification=cluster_classification(df1)
    df2=pd.DataFrame(twitterfinalpostclassification,columns=['topic','Level 0','Level 1(Readable tweet)','Level 2(Remvoving weird data)','Level 3(removing html tags)','Level 4(removing hashtags and mentions)','Level 5(removing links)','Level 6(removing punctuations)','Level 7(Converting into lower case)','final tweets after procesing','category','Score'])
    df2=df2.drop_duplicates(subset='final tweets after procesing')
    df2.to_csv('clusterclassification.csv',index=False)
    return tweetsfinalpostprocessing




tweetsfinalpostprocessing=[]
tweetsfinalpostsentiment=[]
#search_query="death"
numtweets=5

woeid = 2459115
  
# fetching the trends
trends = api.trends_place(id = woeid, exclude = "hashtags")


  
for value in trends:
    for trend in value['trends']:
        search_query=trend['name']
        tweetsfinalpostprocessing=processing(search_query,numtweets)

df2=pd.read_csv(r'preprocessing.csv')
df2=df2.dropna(subset=['final_tweets_after_procesing'])       


