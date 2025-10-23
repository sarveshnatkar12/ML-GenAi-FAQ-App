import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("faq_data.csv")

X = data["question"]
y = data["category"]

model = Pipeline([
    ("tfidf" , TfidfVectorizer()),
    ("clf" , MultinomialNB())
])

model.fit(X,y)

joblib.dump(model , "faq_model.pkl")
print("Model Trained and Saved as faq_model.pkl")