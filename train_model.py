import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# load intents
with open('intents.json') as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        cleaned = " ".join(clean_text(pattern))
        patterns.append(cleaned)
        tags.append(intent['tag'])

# vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
X = vectorizer.fit_transform(patterns)

# model
model = LogisticRegression(max_iter=300)
model.fit(X, tags)

# save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
