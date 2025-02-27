import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
import glob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK dependencies if not available
nltk.download('punkt')
nltk.download('stopwords')

# Define paths for training data
DATA_PATHS = [
    "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_1.json",
    "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_2.json",
    "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_3.json",
    "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_4.json",
    "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_5.json",
]

def load_data():
    """Loads and merges multiple training data JSON files."""
    patterns, labels, responses = [], [], {}

    for path in DATA_PATHS:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        for intent in data["intents"]:
            tag = intent["tag"]
            if tag not in responses:
                responses[tag] = intent["responses"]
            else:
                responses[tag].extend(intent["responses"])  # Merge responses if the same tag exists
            
            for pattern in intent["patterns"]:
                patterns.append(pattern)
                labels.append(tag)

    return patterns, labels, responses

# Preprocessing function
def preprocess_text(text):
    """Tokenizes, removes stopwords, and converts text to lowercase."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Train and save model
def train_and_save_model():
    """Trains an SVM model using multiple training data files and saves it."""
    print("Loading data...")
    patterns, labels, responses = load_data()

    print("Preprocessing text...")
    patterns = [preprocess_text(text) for text in patterns]

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Create TF-IDF vectorizer and SVM model pipeline
    vectorizer = TfidfVectorizer()
    model = make_pipeline(vectorizer, SVC(kernel="linear", probability=True))
    
    print("Training model...")
    model.fit(patterns, labels_encoded)

    # Save the trained model, vectorizer, and label encoder
    MODEL_PATH = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model/chatbot_model.pkl"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, label_encoder, responses), f)

    print(f"Model training completed and saved at '{MODEL_PATH}'.")

if __name__ == "__main__":
    train_and_save_model()
