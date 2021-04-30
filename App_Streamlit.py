
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')


STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    st.title("Live Sentimental Analysis/Opinion Mining")
    st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    #html_temp = """
	#<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live Sentimental Analysis - #GCOE, JALGAON</p></div>
	#"""
    #st.markdown(html_temp, unsafe_allow_html=True)
    #st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    ################# Twitter API Connection #######################
    consumer_key = "EyDX1HOQKSTvUVIUGjOQdKNJP"
    consumer_secret = "9iq5BBtaz0p8ae06021tUoJ3WADoaLdlR2F1WBHHOh1rvJ58bH"
    access_token = "334727433-9g4DGd2BFwM6vBkg5sbI1Dc15syANDpBjGYEISTT"
    access_token_secret = "NGC0R27IfV8wfq1eR0TNLRpNz5xhZenkbpAmehzIuGzKV"



    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    ################################################################
    
    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    
    # Write a Function to extract tweets:
    def get_tweets(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
            #time.sleep(0.1)
            #my_bar.progress(i)
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            #df.to_csv("TweetDataset.csv",index=False)
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass
    # Function to Clean the Tweet.
    def clean_tweet(tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())
    
        
    # Funciton to analyze Sentiment
    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    
    #Function to Pre-process data for Worlcloud
    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
        ###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new

    
    #
    from PIL import Image
    #image = Image.open('Logo1.jpg')
    #st.image(image, caption='Twitter for Analytics',use_column_width=True)
    
    
    # Collect Input from user :
    Topic = str()
    Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))     
    
    if len(Topic) > 0 :
        
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            get_tweets(Topic , Count=200)
        st.success('Tweets have been Extracted !!!!')    
           
    
        # Call function to get Clean tweets
        df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
    
        # Call function to get the Sentiments
        df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
        
        
        # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
        st.write("Total Positive Tweets are : {} | Total Negative Tweets are : {} | Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"]),len(df[df["Sentiment"]=="Negative"]), len(df[df["Sentiment"]=="Neutral"])))#st.markdown(html_temp, unsafe_allow_html=True)
        st.write(df.head(50))
        st.subheader(" Count Plot for Different Sentiments")
        st.write(sns.countplot(df["Sentiment"]))
        st.pyplot()
        text = " ".join(review for review in df.clean_tweet)
        stopwords = set(STOPWORDS)
        text_newALL = prepCloud(text,Topic)
        wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
        st.write(plt.imshow(wordcloud, interpolation='bilinear'))
        st.pyplot()
        text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
        stopwords = set(STOPWORDS)
        text_new_positive = prepCloud(text_positive,Topic)
        #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
        wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_positive)
        st.write(plt.imshow(wordcloud, interpolation='bilinear'))
        st.pyplot()
        text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
        stopwords = set(STOPWORDS)
        text_new_negative = prepCloud(text_negative,Topic)
        #text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
        wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_negative)
        st.write(plt.imshow(wordcloud, interpolation='bilinear'))
        st.pyplot()
        
        
        
        
    
    st.sidebar.header("Government College of Engineering, JALGAON")
    st.sidebar.info("To Develop natural language processing (NLP), sentiment analysis and data mining technologies to build a public opinion analysis system to serve enterprises' need of online public opinion detection.")
    st.sidebar.text("Department of Computer Engineering")
    st.sidebar.header("Developed By:")
    st.sidebar.info("Jayashree Patil, Harshada Borude, Neha Jain")



    if st.button("Exit"):
        st.balloons()



if __name__ == '__main__':
    main()

