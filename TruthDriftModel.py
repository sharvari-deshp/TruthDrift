#TruthDrift: An ML model designed to detect hallucinations within AI generated text. 
import json

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


fullFileTrain = []
fullFileTest = []
trainSentence = []
testSentence = []
trainLabel = []
testLabel = []
accurate_text = []

#Parse through the text, remove claims where there's not enough information (this is not useful)
with open("FEVER Dataset JSON Files/train (1).jsonl", "r") as parse:
    for line in parse: 
        text = json.loads(line) #converts to dict type

        if (text['label'] != "NOT ENOUGH INFO"):
            fullFileTrain.append(text)

            trainSentence.append(text["claim"])
            
            #true claims are labelled as 1, false are labelled as 0
            if text["label"] == "SUPPORTS":
                trainLabel.append(1)
            else: 
                trainLabel.append(0)

with open("FEVER Dataset JSON Files/shared_task_dev.jsonl", "r") as parse:
    for line in parse: 
        text = json.loads(line) #converts to dict type

        if (text['label'] != "NOT ENOUGH INFO"):
            fullFileTest.append(text)

            testSentence.append(text["claim"])
            
            #true claims are labelled as 1, false are labelled as 0
            if text["label"] == "SUPPORTS":
                testLabel.append(1)
            else: 
                testLabel.append(0)


#FEATURE 1: tf-idf analyzer
#limit size to only the top 10,000 most important words for speed
vectorizer = TfidfVectorizer(max_features=10000)

X_train = vectorizer.fit_transform(trainSentence)
y_train = trainLabel

#make test data into different matrix
X_test = vectorizer.transform(testSentence)
y_test = testLabel

#model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

#predictions
y_pred = clf.predict(X_test)

#print out accuracies
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

