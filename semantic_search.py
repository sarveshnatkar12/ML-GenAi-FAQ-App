from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

embeded_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_embeddings(data_path = "faq_data.csv"):
    data = pd.read_csv(data_path)
    data["embeddings"] = data["question"].apply(lambda x:embeded_model.encode(x))
    return data

def semantic_search(query, data, top_K=1):
    q_emb = embeded_model.encode(query)
    doc_emb = np.vstack(data["embeddings"].to_numpy())
    sim = np.dot(doc_emb , q_emb) / (np.linalg.norm(doc_emb , axis=1)*np.linalg.norm(q_emb))
    # cosine_similarity(A,B)=A.B / ∣∣A∣∣×∣∣B∣∣
    
    idx = np.argsort(sim)[::-1][:top_K]

    return data.iloc[idx[0]]["answer"], float(sim[idx[0]])
print('done')