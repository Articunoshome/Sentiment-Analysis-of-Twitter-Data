# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:23:04 2020

@author: Aryaan
"""

import csv
from textblob import TextBlob
import sys

# Do some version specific stuff
if sys.version[0] == '3':
    from importlib import reload
    sntTweets = csv.writer(open("txsent.csv", "w", newline=''))

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    sntTweets = csv.writer(open("txsent.csv", "w"))

alltweets = csv.reader(open("political_social_media.csv", 'r',encoding='ANSI'))

for row in alltweets:
    blob = TextBlob(row[20])
    print (blob.sentiment.polarity)
    if blob.sentiment.polarity > 0:
        sntTweets.writerow(["Positive"])
    elif blob.sentiment.polarity < 0:
        sntTweets.writerow(["Negative"])
    elif blob.sentiment.polarity == 0.0:
        sntTweets.writerow(["Neutral"])


