import re
import nltk
import numpy as np
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from termcolor import colored
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Import datasets
print("Loading data")
train_data = pd.read_csv('Tweet.csv')


# Setting stopwords
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.remove("not")

# Function to expand tweet
def expand_tweet(tweet):
    expanded_tweet = []
    for word in tweet:
        if re.search("n't", word):
            expanded_tweet.append(word.split("n't")[0])
            expanded_tweet.append("not")
        else:
            expanded_tweet.append(word)
    return expanded_tweet

# Function to process tweets
def clean_tweet(data, wordNetLemmatizer, porterStemmer):
    data['Clean_tweet'] = data['Text']
    print(colored("Removing urls", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].replace(re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))"), "")    
    
    print(colored("Removing user handles starting with @", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].str.replace("@[\w]*","")
    print(colored("Removing numbers and special characters", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].str.replace("[^a-zA-Z' ]","")
    
    print(colored("Removing single characters", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].replace(re.compile(r"(^| ).( |$)"), " ")
    data['Clean_tweet'] = data['Clean_tweet'].replace(r"RT"," ")
    print(colored("Tokenizing", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].str.split()
    print(colored("Removing stopwords", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].apply(lambda tweet: [word for word in tweet if word not in STOPWORDS])
    print(colored("Expanding not words", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].apply(lambda tweet: expand_tweet(tweet))
    print(colored("Lemmatizing the words", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].apply(lambda tweet: [wordNetLemmatizer.lemmatize(word) for word in tweet])
    print(colored("Stemming the words", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].apply(lambda tweet: [porterStemmer.stem(word) for word in tweet])
    print(colored("Combining words back to tweets", "yellow"))
    data['Clean_tweet'] = data['Clean_tweet'].apply(lambda tweet: ' '.join(tweet))
    return data

# Define processing methods
wordNetLemmatizer = WordNetLemmatizer()
porterStemmer = PorterStemmer()

# Pre-processing the tweets
print(colored("Processing train data", "green"))
train_data = clean_tweet(train_data, wordNetLemmatizer, porterStemmer)
train_data.to_csv('clean_tweet.csv', index = False)
print(colored("Train data processed and saved to clean_train.csv", "green"))

