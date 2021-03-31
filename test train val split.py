# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:13:29 2020

@author: Aryaan
"""

import csv
import pandas as p
from termcolor import colored
from sklearn.model_selection import train_test_split

# Define variables
COLUMNS = ['Text','Sentiment','Clean_tweet']

# Read dataset
dataset = p.read_csv('clean_tweet.csv', names = COLUMNS, encoding = 'utf-8')
print(colored("Columns: {}".format(', '.join(COLUMNS)), "yellow"))

# Remove extra columns
print(colored("Useful columns: Sentiment and Clean Tweet", "yellow"))
print(colored("Removing other columns", "red"))
dataset.drop(['Text'], axis = 1, inplace = True)
print(colored("Columns removed", "red"))

# Train test split
print(colored("Splitting train and test dataset into 80:20", "yellow"))
X_train, X_test, y_train, y_test = train_test_split(dataset['Clean_tweet'], dataset['Sentiment'], test_size = 0.20, random_state = 100)
train_dataset = p.DataFrame({
	'Clean_tweet': X_train,
	'Sentiment': y_train
	})
print(colored("Train data distribution:", "yellow"))
print(train_dataset['Sentiment'].value_counts())
test_dataset = p.DataFrame({
	'Clean_tweet': X_test,
	'Sentiment': y_test
	})
print(colored("Test data distribution:", "yellow"))
print(test_dataset['Sentiment'].value_counts())
print(colored("Split complete", "yellow"))

# Save train data
print(colored("Saving train data", "yellow"))

train_dataset.to_csv('new_train.csv', index = False)
print(colored("Train data saved to data/train.csv", "green"))

# Save test data
print(colored("Saving test data", "yellow"))
test_dataset.to_csv('new_test.csv', index = False)
print(colored("Test data saved to data/test.csv", "green"))
