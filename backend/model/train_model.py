# import json
# import pickle
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sentence_transformers import SentenceTransformer
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import os

# # Download NLTK dependencies if not available
# nltk.download('punkt')
# nltk.download('stopwords')

# # Define paths for training data
# DATA_PATHS = [
#     "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_1.json",
#     "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_2.json",
#     "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_3.json",
#     "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_4.json",
#     "C:/Users/E Praveen Kumar/Desktop/chatbot_project/data/training_data_5.json",
# ]

# # Load pre-trained Sentence-BERT model for semantic similarity
# sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# def load_data():
#     """Loads and merges multiple training data JSON files."""
#     patterns, labels, responses = [], [], {}

#     for path in DATA_PATHS:
#         if not os.path.exists(path):
#             print(f"Warning: File {path} does not exist. Skipping...")
#             continue

#         with open(path, "r", encoding="utf-8") as file:
#             data = json.load(file)
        
#         for intent in data["intents"]:
#             tag = intent["tag"]
#             if tag not in responses:
#                 responses[tag] = intent["responses"]
#             else:
#                 responses[tag].extend(intent["responses"])  # Merge responses if the same tag exists
            
#             for pattern in intent["patterns"]:
#                 patterns.append(pattern)
#                 labels.append(tag)

#     return patterns, labels, responses

# def preprocess_text(text):
#     """Tokenizes, removes stopwords, and converts text to lowercase."""
#     stop_words = set(stopwords.words("english"))
#     tokens = word_tokenize(text.lower())
#     filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#     return " ".join(filtered_tokens)

# def train_and_save_model():
#     """Trains a semantic similarity-based model and saves it."""
#     print("Loading data...")
#     patterns, labels, responses = load_data()

#     if not patterns or not labels:
#         raise ValueError("No training data found. Please check your data files.")

#     print("Preprocessing text...")
#     patterns = [preprocess_text(text) for text in patterns]

#     # Encode labels
#     label_encoder = LabelEncoder()
#     labels_encoded = label_encoder.fit_transform(labels)

#     # Generate embeddings for all patterns
#     print("Generating embeddings for training data...")
#     pattern_embeddings = sbert_model.encode(patterns)

#     # Save the trained model, embeddings, label encoder, and responses
#     MODEL_DIR = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model"
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl")

#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump((pattern_embeddings, labels_encoded, label_encoder, responses, 0.5), f)  # Default threshold = 0.7

#     print(f"Model training completed and saved at '{MODEL_PATH}'.")

# if __name__ == "__main__":
#     train_and_save_model()

import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors

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

# Load pre-trained Sentence-BERT model for semantic similarity
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextManager:
    """Manages context for multi-turn conversations."""
    def __init__(self):
        self.context: Dict[str, Optional[str]] = {}

    def set_context(self, user_id: str, context: Optional[str]):
        self.context[user_id] = context

    def get_context(self, user_id: str) -> Optional[str]:
        return self.context.get(user_id, None)

    def clear_context(self, user_id: str):
        if user_id in self.context:
            del self.context[user_id]

def load_data() -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Loads and merges multiple training data JSON files."""
    patterns, labels, responses = [], [], {}

    for path in DATA_PATHS:
        if not os.path.exists(path):
            logger.warning(f"File {path} does not exist. Skipping...")
            continue

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

def preprocess_text(text: str) -> str:
    """Tokenizes, removes stopwords, and converts text to lowercase."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

def augment_data(patterns: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Augments training data with paraphrased sentences."""
    # Placeholder for data augmentation logic (e.g., using a paraphrasing model)
    return patterns, labels

def train_and_save_model():
    """Trains a semantic similarity-based model and saves it."""
    logger.info("Loading data...")
    patterns, labels, responses = load_data()

    if not patterns or not labels:
        raise ValueError("No training data found. Please check your data files.")

    logger.info("Preprocessing text...")
    patterns = [preprocess_text(text) for text in patterns]

    # Augment data with paraphrased sentences
    logger.info("Augmenting data...")
    patterns, labels = augment_data(patterns, labels)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Generate embeddings for all patterns
    logger.info("Generating embeddings for training data...")
    pattern_embeddings = sbert_model.encode(patterns)

    # Train a NearestNeighbors model for similarity-based matching
    logger.info("Training NearestNeighbors model...")
    nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn_model.fit(pattern_embeddings)

    # Save the trained model, embeddings, label encoder, and responses
    MODEL_DIR = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, 0.5), f)  # Default threshold = 0.5

    logger.info(f"Model training completed and saved at '{MODEL_PATH}'.")

if __name__ == "__main__":
    train_and_save_model()