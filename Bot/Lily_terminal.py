print("\033[32m ")
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows cross-origin requests
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import pygame  # Import pygame for background music
import time
# Download NLTK dependencies if not available
# nltk.download('punkt')
# nltk.download('stopwords')
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication
# Load the trained model
MODEL_PATH = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model/chatbot_model.pkl"
#INTRO
AUDIO_PATH = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/utils/background.mp3"  
def type_effect(text, delay=0.1, erase=False):
    """Prints text with a typing effect, optionally erasing it character by character."""
    if erase:
        for _ in text:
            sys.stdout.write("\b \b")  
            sys.stdout.flush()
            time.sleep(delay)
    else:
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
def play_background_music():
    """Plays background music while the model is loading."""
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_PATH)
    pygame.mixer.music.set_volume(0.3)  # Set initial volume
    pygame.mixer.music.play(0)  # -1 makes it loop indefinitely
def stop_background_music_fadeout():
    """Gradually decreases the volume and stops the music."""
    pygame.mixer.music.fadeout(2000)  # Fade out over 2 seconds



def load_model():
    """Loads the trained model, label encoder, and responses."""
    play_background_music()
    type_effect("the magic", delay=0.1)
    time.sleep(0.5)  
    type_effect("magic", delay=0.1, erase=True)
    type_effect("model you created is running successfully.\n", delay=0.1)
    stop_background_music_fadeout()
    print("\nLoading the trained model....")
    with open(MODEL_PATH, "rb") as f:
        nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, threshold = pickle.load(f)
    return nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, threshold
nn_model, pattern_embeddings, labels_encoded, label_encoder, responses, threshold = load_model()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load Sentence-BERT model
print("Loading Sentence-BERT model.....")
print("Preprocessing text : Tokenizes, removes stopwords,removing unrelevent data, and converts text to lowercase.")
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
def terminal_chat():
    """Runs the chatbot in the terminal in a loop until the user exits."""
    print("\nTerminal Chatbot Mode Activated! (Type 'exit' to stop)\n")
    while True:
        print("\033[31m")
        user_message = input("\nYou: ")
        if user_message.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye!")
            break
        response = get_response(user_message)
        print("\033[32m")
        print("Chatbot: ", end="")  # Print "Chatbot: " without a newline
        for char in response:
         print(char, end="", flush=True)  # Print each character one by one
         time.sleep(0.1)  # Delay to simulate typing
    print("\033[32m\nTerminating..... \n")
if __name__ == "__main__":
    terminal_chat()