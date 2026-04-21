import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

#dataset
#true claims
supported_claims = [
"The capital of France is Paris.",
"The Earth revolves around the Sun.",
"Water freezes at 0 degrees Celsius.",
"Python is a programming language.",
"The Pacific Ocean is the largest ocean.",
"The human body has 206 bones.",
"The speed of light is approximately 3x10^8 m/s.",
"Mount Everest is the tallest mountain on Earth.",
"The boiling point of water is 100 degrees Celsius.",
"The Great Wall of China was built over centuries.",
"The Sun is a star.",
"The Moon orbits the Earth.",
"Humans breathe oxygen.",
"The Atlantic Ocean is smaller than the Pacific Ocean.",
"Shakespeare wrote Hamlet.",
"Light travels faster than sound.",
"The human heart pumps blood.",
"Jupiter is the largest planet in the solar system.",
"Saturn has rings.",
"Plants use photosynthesis to produce energy.",
"The Nile is the longest river in the world.",
"Venus is hotter than Mercury.",
"The Earth has one moon.",
"Gold is a metal.",
"Ice is solid water.",
"Sound requires a medium to travel.",
"The brain controls the body.",
"Electricity flows through conductors.",
"The Sahara is a desert.",
"Antarctica is the coldest continent.",
"The Amazon rainforest is in South America.",
"Birds have feathers.",
"Fish live in water.",
"Gravity pulls objects toward Earth.",
"The sky appears blue due to scattering.",
"DNA carries genetic information.",
"Atoms are made of protons, neutrons, and electrons.",
"The liver is an organ.",
"The lungs help with breathing.",
"The human body has skin.",
"Rain comes from clouds.",
"The Earth rotates on its axis.",
"Day and night are caused by Earth's rotation.",
"Seasons are caused by Earth's tilt.",
"The Sun rises in the east.",
"The Sun sets in the west.",
"Water is made of hydrogen and oxygen.",
"The Milky Way is a galaxy.",
"Mercury is the closest planet to the Sun.",
"Pluto is classified as a dwarf planet."
]

#fake claims
hallucinated_claims = [
"The capital of France is Berlin.",
"The Earth revolves around Mars.",
"Water freezes at 10 degrees Celsius.",
"Python was invented in 1890.",
"The Pacific Ocean is the smallest ocean.",
"The human body has 500 bones.",
"The speed of light is 1000 m/s.",
"Mount Everest is underwater.",
"Water boils at 50 degrees Celsius.",
"The Great Wall of China was built in 2005.",
"The Sun is a planet.",
"The Moon produces its own light.",
"Humans breathe carbon dioxide.",
"The Atlantic Ocean is larger than the Pacific Ocean.",
"Shakespeare wrote Harry Potter.",
"Sound travels faster than light.",
"The heart pumps air.",
"Jupiter is the smallest planet.",
"Saturn has no rings.",
"Plants do not need sunlight.",
"The Nile is the shortest river.",
"Venus is colder than Earth.",
"The Earth has three moons.",
"Gold is a liquid at room temperature.",
"Ice is a gas.",
"Sound travels in a vacuum.",
"The brain is located in the stomach.",
"Electricity cannot flow through metals.",
"The Sahara is an ocean.",
"Antarctica is the hottest continent.",
"The Amazon rainforest is in Europe.",
"Birds have scales instead of feathers.",
"Fish live on land.",
"Gravity pushes objects away from Earth.",
"The sky is green due to reflection.",
"DNA is made of metal.",
"Atoms do not exist.",
"The liver is a bone.",
"The lungs pump blood.",
"The human body is made of plastic.",
"Rain comes from underground.",
"The Earth does not rotate.",
"Day and night are random.",
"Seasons do not exist.",
"The Sun rises in the west.",
"The Sun sets in the east.",
"Water is made of carbon and nitrogen.",
"The Milky Way is a planet.",
"Mercury is the farthest planet from the Sun.",
"Pluto is a star."
]

#accurate text
accurate_text = [
"Paris is the capital city of France.",
"The Earth orbits the Sun once every 365 days.",
"Water freezes at zero degrees Celsius under normal conditions.",
"Python is a widely used high-level programming language.",
"The Great Wall of China was built over centuries."
]

claims = supported_claims + hallucinated_claims
labels = ([0] * len(supported_claims)) + ([1] * len(hallucinated_claims))

df = pd.DataFrame({"claim": claims, "label": labels})


#feature 1: tf-idf analyzer
vectorizer = TfidfVectorizer()
trusted_vectors = vectorizer.fit_transform(accurate_text)
claim_vectors = vectorizer.transform(df["claim"])

similarity_scores = []
for i, vec in enumerate(claim_vectors):
    similarity_matrix = cosine_similarity(vec, trusted_vectors)
    max_score = similarity_matrix.max()
    similarity_scores.append(max_score)

df["similarity"] = similarity_scores


#feature 2: word overlap
def word_overlap(claim, reference):
    claim_words = set(claim.lower().split())
    ref_words = set(reference.lower().split())
    return len(claim_words & ref_words) / len((claim_words | ref_words))

overlap_scores = []
for claim in df["claim"]:
    scores = [word_overlap(claim, ref) for ref in accurate_text]
    overlap_scores.append(max(scores))

df["overlap"] = overlap_scores


#trains model
X = df[["similarity", "overlap"]]
y = df["label"]


#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#cross validation
clf = LogisticRegression()
scores = cross_val_score(clf, X_scaled, y, cv = 5)

print("Cross-Val Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())


#train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))