from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Allows cross-origin requests
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK dependencies if not available
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model
MODEL_PATH = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model/chatbot_model.pkl"

def load_model():
    """Loads the trained model, label encoder, and responses."""
    with open(MODEL_PATH, "rb") as f:
        model, label_encoder, responses = pickle.load(f)
    return model, label_encoder, responses

model, label_encoder, responses = load_model()

def preprocess_text(text):
    """Tokenizes, removes stopwords, and converts text to lowercase."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

def get_response(user_message):
    """Processes the user's message and returns a chatbot response."""
    if not user_message:
        return "I didn't understand that. Can you please repeat?"

    processed_message = preprocess_text(user_message)
    prediction = model.predict([processed_message])[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    return np.random.choice(responses[predicted_label])

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the chatbot response based on user input."""
    data = request.get_json() or request.form  # Support JSON and form-data
    user_message = data.get("message", "")

    response = get_response(user_message)
    return jsonify({"response": response, "status": "success"})

@app.route("/")
def home():
    """Renders the web-based chatbot UI."""
    return render_template("index.html")

def terminal_chat():
    """Runs the chatbot in the terminal in a loop until the user exits."""
    print("\nTerminal Chatbot Mode Activated! (Type 'exit' to stop)\n")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_message)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    mode = input("Choose mode: 'web' or 'terminal': ").strip().lower()
    if mode == "terminal":
        terminal_chat()
    else:
        print("\nStarting Web Server... Access at http://127.0.0.1:5055/\n")
        app.run(host="0.0.0.0", port=5055, debug=True)
