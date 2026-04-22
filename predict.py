import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


#LOAD SAVED COMPONENTS VIA PICKLE FUNCTION 
#where the trained model comes into effect in order to predict whether AI generated text is hallucinated or not
clf = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

scaler_j = pickle.load(open("scaler_j.pkl", "rb"))
scaler_c = pickle.load(open("scaler_c.pkl", "rb"))
scaler_g = pickle.load(open("scaler_g.pkl", "rb"))

supports_tokens = pickle.load(open("supports_tokens.pkl", "rb"))
refutes_tokens = pickle.load(open("refutes_tokens.pkl", "rb"))

supports_centroid = pickle.load(open("supports_centroid.pkl", "rb"))
refutes_centroid = pickle.load(open("refutes_centroid.pkl", "rb"))

supports_centroid_g = pickle.load(open("supports_centroid_g.pkl", "rb"))
refutes_centroid_g = pickle.load(open("refutes_centroid_g.pkl", "rb"))

glove = pickle.load(open("glove.pkl", "rb"))

#embed dimensions of the glove data w/i datasets folder of code
EMBED_DIM = 50


#FEATURE 4: SEMANTIC SIMILARITY
def sentence_to_vec(sentence):
    words = sentence.lower().split()
    vectors = [glove[w] for w in words if w in glove]
    
    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)
    return np.mean(vectors, axis=0)


#FEATURE 2: JACCARD SIMILARITY
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
        
        if union_support != 0:
            j_support = inter_support / union_support
        else:
            j_support = 0

        if union_refute != 0:
            j_refute = inter_refute / union_refute
        else:
            j_refute = 0
        
        diff = j_support - j_refute
        
        features.append([j_support, j_refute, diff])
    
    return np.array(features)


#FEATURE 3: COSINE SIMILARITY
def compute_cosine_features(X):
    sim_support = cosine_similarity(X, supports_centroid)
    sim_refute = cosine_similarity(X, refutes_centroid)
    
    diff = sim_support - sim_refute
    
    return np.hstack([sim_support, sim_refute, diff])


def compute_glove_features(embeddings):
    sim_support = cosine_similarity(embeddings, supports_centroid_g)
    sim_refute = cosine_similarity(embeddings, refutes_centroid_g)
    
    diff = sim_support - sim_refute
    
    return np.hstack([sim_support, sim_refute, diff])


#Ssplit stenences function as user can input multiple sentences from UI
def split_sentences(text):
    sentences = []
    parts = text.split('.')
    
    for s in parts:
        cleaned = s.strip()
        if cleaned != "":
            sentences.append(cleaned)
    return sentences


#PREDICTION FUNCTION - used to predict ai generated text using user generated text
def predict_text(sentences):

    #FEATURE 1: TF_IDF analyzer
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