import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import time
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import winsound  # For sound effects

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

MODEL_PATH = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model/chatbot_model.pkl"

def load_model():
    """Loads the trained model, label encoder, and responses."""
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", f"Model file not found at:\n{MODEL_PATH}")
        return None, None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            model, label_encoder, responses = pickle.load(f)
        return model, label_encoder, responses
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return None, None, None

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
    try:
        prediction = model.predict([processed_message])[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        return np.random.choice(responses[predicted_label])
    except Exception as e:
        return f"Error: {str(e)}"

# GUI using Tkinter
class HackerChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot - Lets Have Friendly Chat....")
        self.root.geometry("600x700")
        self.root.configure(bg="black")

        # Header Frame
        self.header = tk.Frame(root, bg="black")
        self.header.pack(fill=tk.X, pady=5)

        # Welcome Label
        self.welcome_label = tk.Label(self.header, text="Welcome, I am Lily..!", fg="green", bg="black", font=("Courier", 14, "bold"))
        self.welcome_label.pack(side=tk.LEFT, padx=10)
        self.root.after(3000, self.welcome_label.destroy)  # Remove welcome text after 3 seconds

        # LED Status Indicators
        self.led_green = tk.Label(self.header, text="🟢", font=("Arial", 16), bg="black", fg="green")
     
        self.led_red = tk.Label(self.header, text="🔴", font=("Arial", 16), bg="black", fg="red")  # Hidden initially

        self.led_green.pack(side=tk.LEFT, padx=4)
        
        self.led_red.pack(side=tk.LEFT, padx=4)

        self.animate_leds()

        # Chat Display Area
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, bg="black", fg="green", font=("Courier", 12))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # User Input
        self.user_input = tk.Entry(root, font=("Courier", 14), bg="black", fg="green", insertbackground="green")
        self.user_input.pack(padx=10, pady=5, fill=tk.X)
        self.user_input.insert(0, "Type here…")
        self.user_input.bind("<FocusIn>", self.clear_placeholder)
        self.user_input.bind("<FocusOut>", self.restore_placeholder)
        self.user_input.bind("<Return>", self.send_message)

        # Send Button
        self.send_button = tk.Button(root, text="Send", command=self.send_message, font=("Courier", 12), bg="green", fg="black")
        self.send_button.pack(pady=5)

        # Exit Button
        self.exit_button = tk.Button(root, text="Exit", command=self.exit_application, font=("Courier", 12), bg="red", fg="black")
        self.exit_button.pack(pady=5)

        # Alert Sound if Model Fails
        if model is None or label_encoder is None or responses is None:
            self.led_red.config(fg="red")
            self.play_alert_sound()

    def animate_leds(self):
        """Animates the blinking of the LED indicators."""
        def blink():
            while True:
               
                if model is not None:
                    self.led_green.config(fg="green" if self.led_green.cget("fg") == "black" else "black")
                    self.led_red.config(fg="red" if self.led_red.cget("fg") == "black" else "black")
                else:
                    self.led_green.config(fg="green" if self.led_green.cget("fg") == "black" else "black")
                time.sleep(0.5)

        threading.Thread(target=blink, daemon=True).start()

    def play_alert_sound(self):
        """Plays an alert sound when the model fails."""
        def beep():
            while True:
                winsound.Beep(1000, 300)  # Frequency, Duration
                time.sleep(1)

        threading.Thread(target=beep, daemon=True).start()

    def clear_placeholder(self, event):
        """Removes placeholder text when user focuses on input field."""
        if self.user_input.get() == "Type here…":
            self.user_input.delete(0, tk.END)

    def restore_placeholder(self, event):
        """Restores placeholder text when user leaves input field."""
        if not self.user_input.get():
            self.user_input.insert(0, "Type here…")

    def send_message(self, event=None):
        """Handles sending user input and displaying chatbot response."""
        user_message = self.user_input.get().strip()
        if user_message and user_message != "Type here…":
            self.display_message(f"You: {user_message}\n", "cyan")
            self.user_input.delete(0, tk.END)
            threading.Thread(target=self.process_response, args=(user_message,)).start()

    def process_response(self, user_message):
        """Runs chatbot response processing with a loading effect."""
        self.display_message("Chatbot: 🔄 Processing...\n", "green")
        time.sleep(2)
        response = get_response(user_message)
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete("end-2l", tk.END)  # Remove "Processing..." message
        self.display_message(f"\nChatbot: {response}\n", "green")

    def display_message(self, message, color):
        """Displays a message in the chat area."""
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message, color)
        self.chat_area.yview(tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.tag_configure("cyan", foreground="cyan")
        self.chat_area.tag_configure("green", foreground="lightgreen")

    def exit_application(self):
        """Handles exiting the application gracefully."""
        self.root.destroy()

# Run GUI
if __name__ == "__main__":
    if model:
        root = tk.Tk()
        HackerChatbotGUI(root)
        root.mainloop()
