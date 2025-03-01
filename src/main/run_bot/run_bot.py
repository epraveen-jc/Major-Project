import subprocess
import sys
import os
import time

def main():
    print("\n💬 Welcome to the Chatbot...................! 💬\n")
    time.sleep(1)

    choice = input("Do you want to: \n(1) Train the model \n(2) Run the bot \nEnter 1 or 2: ").strip()

    if choice == "1":
        print("\n📚 Training the model...")
        model_path = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/backend/model/train_model.py"

        if os.path.exists(model_path):
            subprocess.run([sys.executable, model_path])
            print("\nDo you want to run the model...?")
            choice = input("type run").strip()
            print("\n🤖 Running the bot...")
            chatbot_choice = input("\nChoose mode:\n(1) Terminal Chatbot \n(2) GUI Chatbot \nEnter 1 or 2: ").strip()

            if chatbot_choice == "1":
                script_path = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/Bot/Lily_terminal.py"
            elif chatbot_choice == "2":
                script_path = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/Bot/Lily_gui.py"
            else:
                print("\n❌ Invalid choice. Exiting.\n")
                return

            if os.path.exists(script_path):
                print("\n🔄 Launching Chatbot...")
                subprocess.run([sys.executable, script_path])
            else:
                print("\n❌ Error: Script not found!\n")             
 

        else:
            print("\n❌ Error: Training script not found!\n")

    elif choice == "2":
        print("\n🤖 Running the bot...")
        chatbot_choice = input("\nChoose mode:\n(1) Terminal Chatbot \n(2) GUI Chatbot \nEnter 1 or 2: ").strip()

        if chatbot_choice == "1":
            script_path = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/Bot/Lily_terminal.py"
        elif chatbot_choice == "2":
            script_path = "C:/Users/E Praveen Kumar/Desktop/chatbot_project/Bot/Lily_gui.py"
        else:
            print("\n❌ Invalid choice. Exiting.\n")
            return

        if os.path.exists(script_path):
            print("\n🔄 Launching Chatbot...")
            subprocess.run([sys.executable, script_path])
        else:
            print("\n❌ Error: Script not found!\n")

    else:
        print("\n❌ Invalid choice. Exiting.\n")

if __name__ == "__main__":
    main()
