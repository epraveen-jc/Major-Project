import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Allows cross-origin requests
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Download NLTK dependencies if not available
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as c
BASE_DIR = c.BASE_DIR  # Gets the project root dynamically
print(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "model", "chatbot_model.pkl")
# Load the trained model
MODEL_PATH = DATA_PATH

def load_model():
    """Loads the trained model, label encoder, and responses."""
    with open(MODEL_PATH, "rb") as f:
        nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, threshold = pickle.load(f)
    return nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, threshold

nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, threshold = load_model()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load Sentence-BERT model

def preprocess_text(text: str) -> str:
    """Tokenizes, removes stopwords, and converts text to lowercase."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

def get_response(user_message: str) -> str:
    """Processes the user's message and returns a chatbot response."""
    if not user_message:
        return "I didn't understand that. Can you please repeat?"

    # Preprocess the query
    processed_message = preprocess_text(user_message)
    query_embedding = sbert_model.encode([processed_message])[0]

    # Find the nearest neighbors using the NearestNeighbors model
    distances, indices = nn_model.kneighbors([query_embedding])
    nearest_distance = distances[0][0]
    nearest_index = indices[0][0]

    # Check if the nearest match is above the threshold
    if nearest_distance > threshold:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    # Get the predicted label
    predicted_label = label_encoder.inverse_transform([labels_encoded[nearest_index]])[0]
    return np.random.choice(responses[predicted_label])

@app.route('/')
def home():
    """Render the chatbot interface."""
    return "chatbot is running"

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from the web interface."""
    user_message = request.json.get('message')
    if("how are you " == user_message):
        response = "i am fine ...thanks for asking..."
    elif("where are you" == user_message):
        response = "i am in your local disk , and running in a ram...."
    elif("who are you" == user_message):
        response = "i am a chatbot... i am here to help you "
    elif("who are you ?" == user_message):
        response = "i am a chatbot... i am here to help you"
    elif("who are you?" == user_message):
        response = "i am a chatbot... i am here to help you"
    elif("where are you?" == user_message):
        response = "i am in your local disk , and running in your pc's ram...."
    elif("where are you ?" == user_message):
        response = "i am fine ...thanks for asking..."
    elif("how are you" == user_message):
        response = "i am fine ...thanks for asking..."
    elif("how are you ?" == user_message):
        response = "i am fine ...thanks for asking..."
    else:
        response = get_response(user_message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)