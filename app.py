from fastapi import FastAPI, Query
import joblib
import pandas as pd
from semantic_search import build_embeddings, semantic_search

app = FastAPI(title="ML + GenAI FAQ App")

# Load data and model
data = build_embeddings("faq_data.csv")
model = joblib.load("faq_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the ML + GenAI FAQ API"}

@app.get("/ask")
def ask_question(query: str = Query(..., description="Enter your question")):
    pred = model.predict([query])[0]
    probs = model.predict_proba([query])[0]
    confidence = probs.max()

    if confidence < 0.5:
        ans, sim = semantic_search(query, data)
        return {
            "method": "semantic_search",
            "similarity": round(sim, 3),
            "answer": ans
        }

    ans = data[data["category"] == pred].iloc[0]["answer"]
    return {
        "method": "ml_classification",
        "category": pred,
        "confidence": round(confidence, 3),
        "answer": ans
    }
