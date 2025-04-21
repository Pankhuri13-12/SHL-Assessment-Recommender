# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Load SBERT model, FAISS index, and data
@st.cache_resource
def load_artifacts():
    model = SentenceTransformer('sbert_model/')
    index = faiss.read_index('faiss_index.index')
    df = pd.read_csv('assessment_data.csv')
    return model, index, df

# Recommendation Function
def recommend_assessments(profile_text, model, index, df, top_n=10):
    profile_embedding = model.encode([profile_text]).astype('float32')
    _, indices = index.search(profile_embedding, top_n)
    return df.iloc[indices[0]]

# Streamlit UI
st.title("🔍 SHL Assessment Recommender")

profile = st.text_area("✍️ Enter your job role or career aspiration:", 
                       "Looking for a leadership role in financial planning and client management")

if st.button("Get Recommendations"):
    model, index, df = load_artifacts()
    results = recommend_assessments(profile, model, index, df, top_n=10)
    st.subheader("🧠 Top 10 Matching Assessments")
    st.dataframe(results[['Assesment Name', 'cleaned_text', 'Duration', 
                          'Remote Testing Support', 'URL', 'Adaptive/IRT', 'Job Type']].reset_index(drop=True))

# -


