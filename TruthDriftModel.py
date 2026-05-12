#TruthDrift: An ML model designed to detect hallucinations within AI generated text!
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


fullFileTrain = []
fullFileTest = []
trainSentence = []
testSentence = []
trainLabel = []
testLabel = []

#TRAINING DATA
with open("Datasets/train (1).jsonl", "r") as parse:
    for line in parse: 
        text = json.loads(line)

        if (text['label'] != "NOT ENOUGH INFO"):
            fullFileTrain.append(text)
            trainSentence.append(text["claim"])
            
            if text["label"] == "SUPPORTS":
                trainLabel.append(1)
            else: 
                trainLabel.append(0)

#TESTING DATA
with open("Datasets/shared_task_dev.jsonl", "r") as parse:
    for line in parse: 
        text = json.loads(line)

        if (text['label'] != "NOT ENOUGH INFO"):
            fullFileTest.append(text)
            testSentence.append(text["claim"])
            
            if text["label"] == "SUPPORTS":
                testLabel.append(1)
            else: 
                testLabel.append(0)


#FEATURE 1: TF-IDF DATA
#features have been reduced to 3000 
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

X_train = vectorizer.fit_transform(trainSentence)
X_test = vectorizer.transform(testSentence)

#make the indexing faster with numpy 
y_train = np.array(trainLabel)
y_test = np.array(testLabel)


#FEATURE 2: JACCARD SIMILARITY also known as WORD OVERLAP
supports_tokens = set()
refutes_tokens = set()

for i in range(len(trainSentence)):
    tokens = trainSentence[i].lower().split()
    if trainLabel[i] == 1:
        supports_tokens.update(tokens)
    else:
        refutes_tokens.update(tokens)


def compute_jaccard_features(sentences):
    features = []
    
    len_supports = len(supports_tokens)
    len_refutes = len(refutes_tokens)
    
    for sentence in sentences:
        tokens = set(sentence.lower().split())
        len_tokens = len(tokens)
        
        inter_support = len(tokens & supports_tokens)
        inter_refute = len(tokens & refutes_tokens)
        
        union_support = len_tokens + len_supports - inter_support
        union_refute = len_tokens + len_refutes - inter_refute
        
        j_support = inter_support / union_support if union_support != 0 else 0
        j_refute = inter_refute / union_refute if union_refute != 0 else 0
        
        diff = j_support - j_refute
        
        features.append([j_support, j_refute, diff])
    
    return np.array(features)


jaccard_train = compute_jaccard_features(trainSentence)
jaccard_test = compute_jaccard_features(testSentence)

scaler_j = StandardScaler()
jaccard_train = scaler_j.fit_transform(jaccard_train)
jaccard_test = scaler_j.transform(jaccard_test)


#FEATURE 3: COSINE SIMILARITY
supports_indices = [i for i in range(len(y_train)) if y_train[i] == 1]
refutes_indices = [i for i in range(len(y_train)) if y_train[i] == 0]

supports_centroid = X_train[supports_indices].mean(axis=0)
refutes_centroid = X_train[refutes_indices].mean(axis=0)

supports_centroid = np.asarray(supports_centroid)
refutes_centroid = np.asarray(refutes_centroid)


def compute_cosine_features(X):
    sim_support = cosine_similarity(X, supports_centroid)
    sim_refute = cosine_similarity(X, refutes_centroid)
    
    diff = sim_support - sim_refute
    
    return np.hstack([sim_support, sim_refute, diff])


cosine_train = compute_cosine_features(X_train)
cosine_test = compute_cosine_features(X_test)

scaler_c = StandardScaler()
cosine_train = scaler_c.fit_transform(cosine_train)
cosine_test = scaler_c.transform(cosine_test)


#combine following features 
X_train_combined = hstack([X_train, jaccard_train, cosine_train])
X_test_combined = hstack([X_test, jaccard_test, cosine_test])

#train model utilizing logistic regression 
clf = LogisticRegression(max_iter=10000, solver='saga', C=1.0)
clf.fit(X_train_combined, y_train)

y_pred = clf.predict(X_test_combined)

#threshold to also optimize f1 score accuracy
y_pred = (y_pred > 0.4).astype(int)


#print out statisitcs/results to compare best features
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

#save for UI
pickle.dump(clf, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

pickle.dump(scaler_j, open("scaler_j.pkl", "wb"))
pickle.dump(scaler_c, open("scaler_c.pkl", "wb"))

pickle.dump(supports_tokens, open("supports_tokens.pkl", "wb"))
pickle.dump(refutes_tokens, open("refutes_tokens.pkl", "wb"))

pickle.dump(supports_centroid, open("supports_centroid.pkl", "wb"))
pickle.dump(refutes_centroid, open("refutes_centroid.pkl", "wb"))


#PREDICTION FUNCTION FOR UI TO USE
def predict_text(sentences):
    X = vectorizer.transform(sentences)

    j = compute_jaccard_features(sentences)
    j = scaler_j.transform(j)

    c = compute_cosine_features(X)
    c = scaler_c.transform(c)

    X_final = hstack([X, j, c])

    probs = clf.predict_proba(X_final)[:,1]

    return probs
