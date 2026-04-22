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
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))

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

#FEATURE 4: SEMANTIC SIMILARITY 
# #other features lack semantic similarity, so might be useful to import and incorporate to improve accuracy
def load_glove_filtered(path, sentences):
    vocab = set()
    for s in sentences:
        vocab.update(s.lower().split())
    
    embeddings = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            
            if word in vocab:   # ONLY keep needed words
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
    
    return embeddings

glove = load_glove_filtered("Datasets/glove.6B.50d.txt", trainSentence)
EMBED_DIM = 50


def sentence_to_vec(sentence):
    words = sentence.lower().split()
    vectors = [glove[w] for w in words if w in glove]
    
    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)
    
    return np.mean(vectors, axis=0)


train_emb = np.array([sentence_to_vec(s) for s in trainSentence])
test_emb = np.array([sentence_to_vec(s) for s in testSentence])


supports_emb = train_emb[supports_indices]
refutes_emb = train_emb[refutes_indices]

supports_centroid_g = np.mean(supports_emb, axis=0).reshape(1, -1)
refutes_centroid_g = np.mean(refutes_emb, axis=0).reshape(1, -1)


def compute_glove_features(embeddings):
    sim_support = cosine_similarity(embeddings, supports_centroid_g)
    sim_refute = cosine_similarity(embeddings, refutes_centroid_g)
    
    diff = sim_support - sim_refute
    
    return np.hstack([sim_support, sim_refute, diff])


glove_train = compute_glove_features(train_emb)
glove_test = compute_glove_features(test_emb)

scaler_g = StandardScaler()
glove_train = scaler_g.fit_transform(glove_train)
glove_test = scaler_g.transform(glove_test)


#combine following features 
X_train_combined = hstack([X_train, jaccard_train, cosine_train, glove_train])
X_test_combined = hstack([X_test, jaccard_test, cosine_test, glove_test])

#train model utilizing logistic regression 
clf = LogisticRegression(max_iter=500, solver='saga', C=1.0)
clf.fit(X_train_combined, y_train)

#use probabilities instead of predicition - runtime's faster
y_probs = clf.predict_proba(X_test_combined)[:,1]

#threshold to also optimize f1 score accuracy
y_pred = (y_probs > 0.4).astype(int)


#print out statisitcs/results to compare best features
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

#save for UI
pickle.dump(clf, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(scaler_j, open("scaler_j.pkl", "wb"))
pickle.dump(scaler_c, open("scaler_c.pkl", "wb"))
pickle.dump(scaler_g, open("scaler_g.pkl", "wb"))

pickle.dump(supports_tokens, open("supports_tokens.pkl", "wb"))
pickle.dump(refutes_tokens, open("refutes_tokens.pkl", "wb"))

pickle.dump(supports_centroid, open("supports_centroid.pkl", "wb"))
pickle.dump(refutes_centroid, open("refutes_centroid.pkl", "wb"))

pickle.dump(supports_centroid_g, open("supports_centroid_g.pkl", "wb"))
pickle.dump(refutes_centroid_g, open("refutes_centroid_g.pkl", "wb"))

pickle.dump(glove, open("glove.pkl", "wb"))

print("Saved everything for UI")


#PREDICTION FUNCTION FOR UI TO USE
def predict_text(sentences):
    X = vectorizer.transform(sentences)

    j = compute_jaccard_features(sentences)
    j = scaler_j.transform(j)

    c = compute_cosine_features(X)
    c = scaler_c.transform(c)

    emb = np.array([sentence_to_vec(s) for s in sentences])
    g = compute_glove_features(emb)
    g = scaler_g.transform(g)

    X_final = hstack([X, j, c, g])

    probs = clf.predict_proba(X_final)[:,1]

    return probs
