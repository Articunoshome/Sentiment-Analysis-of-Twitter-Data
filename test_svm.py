# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:43:07 2020

@author: Aryaan
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score






data=pd.read_csv("clean_tweet.csv")
dt=pd.read_csv("new_train.csv")
df=pd.read_csv("new_test.csv")

data['Clean_tweet']=data['Clean_tweet'].astype('U240')
dt['Clean_tweet']=dt['Clean_tweet'].astype('U240')
df['Clean_tweet']=df['Clean_tweet'].astype('U240')




Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(dt['Sentiment'])
Test_Y = Encoder.fit_transform(df['Sentiment'])

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['Clean_tweet'])
Train_X_Tfidf = Tfidf_vect.transform(dt['Clean_tweet'])
Test_X_Tfidf = Tfidf_vect.transform(df['Clean_tweet'])

print(Tfidf_vect.vocabulary_)

print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

