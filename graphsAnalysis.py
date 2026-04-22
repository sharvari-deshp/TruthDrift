import json
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#forgot to add f1 scores for more analysis
f1_scores = []

#all pickle components
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


#load all of the test data
testSentence = []
testLabel = []

with open("Datasets/shared_task_dev.jsonl", "r") as parse:
    for line in parse:
        text = json.loads(line)

        if text['label'] != "NOT ENOUGH INFO":
            testSentence.append(text["claim"])

            if text["label"] == "SUPPORTS":
                testLabel.append(1)
            else:
                testLabel.append(0)

y_test = np.array(testLabel)

#FEature 1: TF-IDF
X_test = vectorizer.transform(testSentence)


#Feeature 2: jaccard features/word overlap
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


jaccard_test = scaler_j.transform(compute_jaccard_features(testSentence))

#FEATURE 3: COSINE SIMILARITY
def compute_cosine_features(X):
    sim_support = cosine_similarity(X, supports_centroid)
    sim_refute = cosine_similarity(X, refutes_centroid)

    diff = sim_support - sim_refute

    return np.hstack([sim_support, sim_refute, diff])


cosine_test = scaler_c.transform(compute_cosine_features(X_test))


#FEATURE 4: Glove (Semantic similarity)
EMBED_DIM = 50

def sentence_to_vec(sentence):
    words = sentence.lower().split()
    vectors = [glove[w] for w in words if w in glove]

    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)

    return np.mean(vectors, axis=0)


test_emb = np.array([sentence_to_vec(s) for s in testSentence])


def compute_glove_features(embeddings):
    sim_support = cosine_similarity(embeddings, supports_centroid_g)
    sim_refute = cosine_similarity(embeddings, refutes_centroid_g)

    diff = sim_support - sim_refute

    return np.hstack([sim_support, sim_refute, diff])


glove_test = scaler_g.transform(compute_glove_features(test_emb))

#Final model
start = time.time()

X_final = hstack([X_test, jaccard_test, cosine_test, glove_test])
probs = clf.predict_proba(X_final)[:,1]

y_pred = (probs > 0.4).astype(int)

end = time.time()

accuracy = accuracy_score(y_test, y_pred)



print("Accuracy:", accuracy)



models = []
accuracies = []



#1 feature
start = time.time()

clf1 = LogisticRegression(max_iter=500, solver='saga')
clf1.fit(X_test, y_test)  # lightweight reuse

pred1 = clf1.predict(X_test)

end = time.time()

models.append("TF-IDF")
accuracies.append(accuracy_score(y_test, pred1))
f1_scores.append(f1_score(y_test, pred1))



#2 features
start = time.time()

X_2 = hstack([X_test, jaccard_test])

clf2 = LogisticRegression(max_iter=500, solver='saga')
clf2.fit(X_2, y_test)

pred2 = clf2.predict(X_2)

end = time.time()

models.append("TF-IDF + Jaccard")
accuracies.append(accuracy_score(y_test, pred2))
f1_scores.append(f1_score(y_test, pred2))



#3 features
start = time.time()

X_3 = hstack([X_test, jaccard_test, cosine_test])

clf3 = LogisticRegression(max_iter=500, solver='saga')
clf3.fit(X_3, y_test)

pred3 = clf3.predict(X_3)

end = time.time()

models.append("TF-IDF + Jaccard + Cosine Similarity")
accuracies.append(accuracy_score(y_test, pred3))

f1_scores.append(f1_score(y_test, pred3))


#full model
start = time.time()

X_full = hstack([X_test, jaccard_test, cosine_test, glove_test])
probs = clf.predict_proba(X_full)[:,1]
pred4 = (probs > 0.4).astype(int)

end = time.time()

models.append("TF-IDF + Jaccard + Cosine Similarity + GloVe")
accuracies.append(accuracy_score(y_test, pred4))
f1_scores.append(f1_score(y_test, pred4))


#graph 1: f1 and accuracy score

plt.figure()

x = np.arange(len(models))

plt.bar(x - 0.2, accuracies, width=0.4, label="Accuracy")
plt.bar(x + 0.2, f1_scores, width=0.4, label="F1 Score")

#add labels for x values
for i in range(len(models)):
    plt.text(i - 0.2, accuracies[i] + 0.005, f"{accuracies[i]:.2f}", ha='center', fontsize=8)
    plt.text(i + 0.2, f1_scores[i] + 0.005, f"{f1_scores[i]:.2f}", ha='center', fontsize=8)

plt.xticks(x, models, fontsize=8) 
plt.xlabel("Feature Combinations")
plt.ylabel("Score")
plt.title("Accuracy and F1 Score Comparison")

plt.legend()

plt.savefig("accuracy_f1_graph.png")
plt.show()

#graph 3: confusion matrix

cm = confusion_matrix(y_test, pred4)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Hallucinated", "Factual"])

disp.plot(cmap="Blues")

plt.title("Confusion Matrix (For Full Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()