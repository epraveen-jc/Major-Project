document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const errorBox = document.getElementById("error-box");
  
    // Function to add a message to the chat box
    const addMessage = (message, isUser) => {
      const messageElement = document.createElement("div");
      messageElement.classList.add("message");
      messageElement.classList.add(isUser ? "user-message" : "bot-message");
      messageElement.textContent = message;
      chatBox.appendChild(messageElement);
      chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
    };
  
    // Function to display errors
    const showError = (error) => {
      errorBox.textContent = error;
      errorBox.style.display = "block";
      setTimeout(() => {
        errorBox.style.display = "none";
      }, 5000); // Hide error after 5 seconds
    };
  
    // Function to send user message to the backend
    const sendMessage = async () => {
      const message = userInput.value.trim();
      if (!message) {
        showError("Please enter a message.");
        return;
      }
  
      addMessage(message, true); // Add user message to the chat box
      userInput.value = ""; // Clear input field
  
      try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message }),
        });
  
        if (!response.ok) {
          throw new Error("Failed to fetch response from the server.");
        }
  
        const data = await response.json();
        if (data.error) {
          showError(data.error);
        } else {
          addMessage(data.response, false); // Add bot response to the chat box
        }
      } catch (error) {
        showError("An error occurred. Please try again.");
        console.error(error);
      }
    };
  
    // Event listeners
    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        sendMessage();
      }
    });
  });