#- * -coding: utf - 8 - * -
"""
Created on Sat Feb 29 11: 25: 46 2020

@author: Aryaan
"""

import tweepy
import csv
from tweepy import OAuthHandler

class TwitterClient:
    '''
    Generic Twitter Class
    for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        #keys and tokens from the Twitter Dev Console
        consumer_key = '06zH3aTdWf2poSnQ9ECkP1jjO'
        consumer_secret='8YOB9JIGj0zT4gDDNJHueNrlpG0HpSb3OVGY3WFimJs3XLpmwt'
        access_token = '830106980517699584-HfcEVBl24xepBREfl6qKT2pk4bIDpst'
        access_token_secret = '4M1ndPHhATnMhRplBC5Ldkh1KLt58zDc8InZ2Ar9DVhLO'
        # attempt authentication
        try: #create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        except:
            print("Error: Authentication Failed")

    

    def save_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and store them.
        '''
        
        try:
            
            
            outtweets = [
                [tweet.id_str, tweet.created_at, tweet.text.encode('utf-8')] for tweet in tweepy.Cursor(self.api.search,q=query,lang="en",since="2020-02-23",count=count).items(60000)
                ]
            with open('train.csv', 'w', newline='\n') as csvFile:
                csvWriter = csv.writer(csvFile, delimiter = ',')
                csvWriter.writerow(["id", "created_at", "text"])
                csvWriter.writerows(outtweets)
        except tweepy.TweepError as e:
            #print error(if any)
            print("Error : " + str(e))

def main():
    #creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    api.save_tweets(query = '#JoeBiden', count = 200)

main()
