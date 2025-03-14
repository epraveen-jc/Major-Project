# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# # Load the IPC sections CSV file
# df = pd.read_csv('chatbot_project/data/ipc_sections.csv')

# # Preprocessing function to clean text
# def preprocess_text(text):
#     if isinstance(text, str):  # Check if text is a string
#         text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#         return text
#     return ""  # Return an empty string if text is NaN or not a string

# # Drop rows where 'Offense' is completely missing (optional)
# df = df.dropna(subset=['Offense'])

# # Preprocess the "Offense" column
# df['Offense_cleaned'] = df['Offense'].apply(preprocess_text)

# # Initialize TF-IDF Vectorizer
# tfidf_vectorizer = TfidfVectorizer()

# # Fit and transform the "Offense_cleaned" column into TF-IDF vectors
# tfidf_matrix = tfidf_vectorizer.fit_transform(df['Offense_cleaned'])

# # Function to retrieve IPC information based on user query
# def retrieve_ipc_info(user_query):
#     # Preprocess the user query
#     user_query_cleaned = preprocess_text(user_query)
    
#     # Transform the user query into a TF-IDF vector
#     user_query_vector = tfidf_vectorizer.transform([user_query_cleaned])
    
#     # Calculate cosine similarity between the user query and the "Offense" column
#     cosine_similarities = cosine_similarity(user_query_vector, tfidf_matrix).flatten()
    
#     # Find the index of the most similar offense
#     most_similar_index = cosine_similarities.argmax()
    
#     # Retrieve the corresponding IPC section and other details
#     ipc_section = df.iloc[most_similar_index]['IPC Sections']
#     offense = df.iloc[most_similar_index]['Offense']
#     punishment = df.iloc[most_similar_index]['Punishment']
#     cognizable = df.iloc[most_similar_index]['Cognizable']
#     bailable = df.iloc[most_similar_index]['Bailable']
#     court = df.iloc[most_similar_index]['Court']
    
#     # Generate a user-friendly response
#     response = (
#         f"According to IPC Section {ipc_section}, the offense '{offense}' is punishable with {punishment}. "
#         f"It is {cognizable} and {bailable}. The case can be tried in {court}."
#     )
    
#     return response

# # Example usage
# test_queries = [
#     "What is the punishment for rioting?",
#     "If someone causes hurt with a dangerous weapon, what is the punishment?",
#     "Tell me about the law for wearing a military uniform without being a soldier."
# ]

# for query in test_queries:
#     print(retrieve_ipc_info(query))

# # Run chatbot loop
# while True:
#     user_query = input("Ask about an IPC law (or type 'bye' to exit): ")
#     if user_query.lower() == 'bye':
#         print("Terminating...")
#         break
#     print(retrieve_ipc_info(user_query))
# 
# import subprocess
# import pandas as pd
# import re
# import torch
# from transformers import BertTokenizer, BertModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import random
# # Load the IPC dataset
# df = pd.read_csv('C:/Users/E Praveen Kumar/Desktop/chatbot_project/chatbot_project/data/ipc_sections.csv')

# # Load BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# # Function to preprocess text (cleaning + tokenization)
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
    
#     # Convert to lowercase, remove special characters
#     text = text.lower().strip()
#     text = re.sub(r'[^\w\s]', '', text)
    
#     return text

# # Preprocess the "Offense" column
# df['Offense_cleaned'] = df['Offense'].apply(preprocess_text)

# # TF-IDF Vectorizer with bigrams & trigrams
# tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
# tfidf_matrix = tfidf_vectorizer.fit_transform(df['Offense_cleaned'])

# # Function to generate BERT embeddings
# def get_bert_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# # Compute BERT embeddings for all offenses
# df['BERT_Embeddings'] = df['Offense_cleaned'].apply(get_bert_embedding)

# # Function to match user query using BERT & TF-IDF
# def retrieve_ipc_info(user_query):
#     user_query_cleaned = preprocess_text(user_query)

#     # Compute BERT embedding for the query
#     query_embedding = get_bert_embedding(user_query_cleaned)

#     # Compute cosine similarity with BERT
#     df['BERT_Similarity'] = df['BERT_Embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])

#     # Compute cosine similarity with TF-IDF
#     user_query_vector = tfidf_vectorizer.transform([user_query_cleaned])
#     tfidf_similarities = cosine_similarity(user_query_vector, tfidf_matrix).flatten()

#     # Combine BERT and TF-IDF scores (weighted)
#     df['Final_Similarity'] = (df['BERT_Similarity'] * 0.7) + (tfidf_similarities * 0.3)

#     # Get the best match
#     best_match = df.loc[df['Final_Similarity'].idxmax()]

#     if best_match['Final_Similarity'] < 0.2:
#         return "Sorry, I couldn't find a clear match. Can you rephrase your question?"

    

#     responses = [
#         lambda best_match: f"🚔 **IPC {best_match['IPC Sections']}** - {best_match['Offense']}\n⚖️ Punishment: {best_match['Punishment']}\n🔍 Cognizable: {best_match['Cognizable']} | 🏛️ Bailable: {best_match['Bailable']}\n📌 Handled by: {best_match['Court']}",
        
#         lambda best_match: f"📖 **Section {best_match['IPC Sections']}**: {best_match['Offense']}\n⛓️ Penalty: {best_match['Punishment']}\n🔎 Cognizable? {best_match['Cognizable']} | 🎗️ Bailable? {best_match['Bailable']}\n⚖️ Jurisdiction: {best_match['Court']}",

#         lambda best_match: f"⚠️ {best_match['Offense']} falls under **IPC {best_match['IPC Sections']}**\n⏳ Consequences: {best_match['Punishment']}\n👨‍⚖️ Case Type - Cognizable: {best_match['Cognizable']}, Bailable: {best_match['Bailable']}\n🏢 Heard at: {best_match['Court']}",

#         lambda best_match: f"🛑 Violation: {best_match['Offense']} (IPC {best_match['IPC Sections']})\n🔨 Sentence: {best_match['Punishment']}\n⚡ Cognizable: {best_match['Cognizable']} | 🎗️ Bailable: {best_match['Bailable']}\n🏛️ Court: {best_match['Court']}",

#         lambda best_match: f"📌 **IPC Code {best_match['IPC Sections']}**\n🛡️ Offense: {best_match['Offense']}\n⚖️ Legal Action: {best_match['Punishment']}\n🔍 Cognizable: {best_match['Cognizable']} | 🎗️ Bailable: {best_match['Bailable']}\n🏢 Court: {best_match['Court']}"
#     ]

#     response = random.choice(responses)(best_match)


#     return response

# # Chatbot loop
# while True:
#     user_query = input("\n💬 Ask about an IPC law (or type 'bye' to exit): ")
#     if user_query.lower() == 'bye':
#         print("👋 Goodbye!")
#         break
#     print(retrieve_ipc_info(user_query))

import pandas as pd
import re
import torch
import os
from pathlib import Path
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
import pyttsx3
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as c
BASE_DIR = c.BASE_DIR  # Gets the project root dynamically
print(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "datasets", "ipc_sections.csv")
# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Create project directories
MODEL_DIR = Path('./model')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Load the IPC dataset
try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    try:
        df = pd.read_csv('datasets/ipc_sections.csv')
        print("Dataset loaded from alternative location.")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find ipc_sections.csv. Please ensure it's in the correct directory.")

# Load BERT tokenizer and model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load Sentence Transformer for semantic search
print("Loading SentenceTransformer model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def speak_success():
    engine = pyttsx3.init()

    # Get available voices
    voices = engine.getProperty('voices')

    # Select a female voice (varies by system)
    for voice in voices:
        if "female" in voice.name.lower():  # Adjust based on system
            engine.setProperty('voice', voice.id)
            break
    else:
        engine.setProperty('voice', voices[1].id)  # Default to another voice

    engine.setProperty('rate', 150)  # Adjust speed
    engine.say("Your ipc model is running successfully!")
    engine.runAndWait()

# Run the function


# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    return " ".join(filtered_text)

# Preprocess the "Offense" column
print("Preprocessing dataset...")
df['Offense_cleaned'] = df['Offense'].apply(preprocess_text)

# TF-IDF Vectorizer with bigrams & trigrams
print("Building TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Offense_cleaned'])

# Compute sentence transformer embeddings for all offenses
print("Computing sentence embeddings...")
df['Sentence_Embeddings'] = list(sentence_model.encode(df['Offense_cleaned'].tolist()))

# Build BM25 index for efficient keyword search
print("Building BM25 index...")
tokenized_corpus = [doc.split(" ") for doc in df['Offense_cleaned']]
bm25 = BM25Okapi(tokenized_corpus)

# Enhanced search function
def retrieve_ipc_info(user_query):
    user_query_cleaned = preprocess_text(user_query)
    
    # 1. BM25 Keyword Search
    tokenized_query = user_query_cleaned.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 2. Semantic Search with Sentence Transformers
    query_embedding = sentence_model.encode(user_query_cleaned)
    semantic_similarities = [cosine_similarity([query_embedding], [emb])[0][0] for emb in df['Sentence_Embeddings']]
    
    # 3. TF-IDF Vector Space Search
    user_query_vector = tfidf_vectorizer.transform([user_query_cleaned])
    tfidf_similarities = cosine_similarity(user_query_vector, tfidf_matrix).flatten()
    
    # Normalize scores
    normalized_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores)) if np.max(bm25_scores) != np.min(bm25_scores) else np.zeros_like(bm25_scores)
    
    # Combine all search methods with different weights
    final_scores = (
        normalized_bm25 * 0.3 + 
        np.array(semantic_similarities) * 0.5 + 
        tfidf_similarities * 0.2
    )
    
    # Get top 3 matches
    top_indices = np.argsort(final_scores)[-3:][::-1]
    top_matches = df.iloc[top_indices].copy()
    top_matches['Final_Similarity'] = final_scores[top_indices]
    
    best_match = top_matches.iloc[0]
    
    # If no good match found
    if best_match['Final_Similarity'] < 0.2:
        return {"error": "Sorry, I couldn't find a clear match. Can you rephrase your question?"}
    
    # Prepare response
    response = {
        "best_match": {
            "ipc_section": best_match['IPC Sections'],
            "offense": best_match['Offense'],
            "punishment": best_match['Punishment'],
            "cognizable": best_match['Cognizable'],
            "bailable": best_match['Bailable'],
            "court": best_match['Court'],
            "similarity_score": float(best_match['Final_Similarity'])
        },
        "similar_sections": [2]
    }
    
    # Add similar sections if there are good matches
    if len(top_matches) > 1 and top_matches.iloc[1]['Final_Similarity'] > 0.6:
        for i in range(1, len(top_matches)):
            match = top_matches.iloc[i]
            response["similar_sections"].append({
                "ipc_section": match['IPC Sections'],
                "offense": match['Offense'],
                "similarity_score": float(match['Final_Similarity'])
            })
    
    return response

# Flask API endpoint
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        result = retrieve_ipc_info(user_query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=8085)