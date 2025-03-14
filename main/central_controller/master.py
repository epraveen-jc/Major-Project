import os
import sys
import time
import subprocess
import webbrowser
import subprocess
import importlib
# List of scripts to run
print("\033[93m ")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import config as c
BASE_DIR = c.BASE_DIR  # Gets the project root dynamically
print(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "model", "chatbot_model.pkl")
def ensure_pip():
    try:
        import pip
    except ImportError:
        print("pip is not installed. Installing pip...")
        try:
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
            print("pip installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install pip. Error: {e}")
            sys.exit(1)

# List of required packages
required_packages = [
    "pandas", "torch", "transformers", "scikit-learn", "sentence-transformers",
    "rank-bm25", "nltk", "flask", "flask-cors", "pyttsx3", "numpy", "pygame",
    "jsonpickle", "wordnet", "pyyaml"
]
ensure_pip()
# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Check which packages are missing
missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]

# Install missing packages
if missing_packages:
    print(f"The following packages are missing: {', '.join(missing_packages)}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
        print("All missing packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages. Error: {e}")
else:
    print("All required packages are already installed.")



    
scripts = [
    r""+os.path.join(BASE_DIR, "train_model", "train_model.py"),
    r""+os.path.join(BASE_DIR, "bots", "ipc_nlp.py"),
    r""+os.path.join(BASE_DIR, "bots", "web_server.py"),
    r""+os.path.join(BASE_DIR, "bots", "terminal_based_bot.py"),
]

# Different animations for each script
animations = [
    ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],  
]

# Function to show loading animation
def loading_animation(animation, duration=3):
    start_time = time.time()
    while time.time() - start_time < duration:
        for symbol in animation:
            print(f"\rRunning {symbol} ", end="")
            time.sleep(0.20)

# Function to minimize the window (Windows-specific)
def minimize_window(process_id):
    try:
        # Use PowerShell to minimize the window
        subprocess.run(
            ["powershell", "-Command", f"(Get-Process -Id {process_id}).MainWindowHandle | ForEach-Object {{ (New-Object -ComObject Shell.Application).MinimizeAll() }}"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to minimize window: {e}")

# Function to run scripts sequentially
def run_script(script_path, animation):
    print(f"\nStarting script: {os.path.basename(script_path)}")
    process = subprocess.Popen(["python", script_path], creationflags=subprocess.CREATE_NO_WINDOW)
    loading_animation(animation, duration=8)  # Show animation for 3 seconds
    # minimize_window(process.pid)  # Minimize the window
      # Wait for 1 second before moving to the next script

# Run all scripts sequentially with different animations
for i, script in enumerate(scripts):
    run_script(script, animations[i % len(animations)])  # Cycle through animations

# Open the index.html file in the default browser
print("\nAll scripts have been started. Opening the web page...")
print("Crafted By Praveen....")
time.sleep(3)
print("Created By Praneeth....")
time.sleep(4)
print("All scripts running successfully....interface is created 2 seconds ago....")
html_file = r""+os.path.join(BASE_DIR, "templates", "index.html")
webbrowser.open(html_file)

print("Web page opened. All tasks completed.")