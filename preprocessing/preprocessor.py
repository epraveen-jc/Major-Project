#preprocessing
import json
import os
import string
import re
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as c
BASE_DIR = c.BASE_DIR  # Gets the project root dynamically
print(BASE_DIR)
class Preprocessor:
    def __init__(self, data_files):
        """
        Initialize the preprocessor with data files.
        
        Args:
            data_files (list): List of data file paths
        """
        self.data_files = data_files
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.intents_data = self.load_data()
        self.combined_intents = self.combine_intents()
        
    def load_data(self):
        """
        Load and validate all data files.
        
        Returns:
            list: List of loaded data from each file
        """
        all_data = []
        
        for file_path in self.data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # Basic validation
                    if 'intents' not in data:
                        print(f"Warning: 'intents' key not found in {file_path}")
                        continue
                    all_data.append(data)
                    print(f"Successfully loaded data from {file_path}")
            except FileNotFoundError:
                print(f"Error: File not found - {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                
        if not all_data:
            raise ValueError("No valid data files were loaded. Please check file paths and formats.")
            
        return all_data
    
    def combine_intents(self):
        """
        Combine intents from all data files, merging duplicate tags.
        
        Returns:
            dict: Combined intents dictionary
        """
        combined = {"intents": []}
        intent_map = {}
        
        # Process each data file
        for data in self.intents_data:
            for intent in data["intents"]:
                tag = intent["tag"].lower()  # Convert to lowercase for consistency
                
                if tag in intent_map:
                    # Merge with existing intent
                    existing_intent = intent_map[tag]
                    # Add patterns if they don't already exist
                    for pattern in intent["patterns"]:
                        if pattern not in existing_intent["patterns"]:
                            existing_intent["patterns"].append(pattern)
                    # Add responses if they don't already exist
                    for response in intent["responses"]:
                        if response not in existing_intent["responses"]:
                            existing_intent["responses"].append(response)
                else:
                    # Create a new intent
                    intent_map[tag] = {
                        "tag": tag,
                        "patterns": intent["patterns"].copy(),
                        "responses": intent["responses"].copy()
                    }
        
        # Convert the map back to a list
        combined["intents"] = list(intent_map.values())
        return combined
    
    def preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, converting to lowercase,
        tokenizing, removing stop words, and lemmatizing.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return tokens
    
    def generate_training_data(self):
        """
        Generate training data from the combined intents.
        
        Returns:
            tuple: (patterns, tags, words, classes)
        """
        patterns = []
        tags = []
        words = set()
        classes = set()
        
        # Process each intent
        for intent in self.combined_intents["intents"]:
            tag = intent["tag"]
            classes.add(tag)
            
            # Process each pattern in the intent
            for pattern in intent["patterns"]:
                processed_pattern = self.preprocess_text(pattern)
                patterns.append(processed_pattern)
                tags.append(tag)
                words.update(processed_pattern)
        
        # Convert sets to sorted lists for deterministic behavior
        words = sorted(list(words))
        classes = sorted(list(classes))
        
        return patterns, tags, words, classes
    
    def save_preprocessed_data(self, output_file="processed_data.json"):
        """
        Save the combined and preprocessed data.
        
        Args:
            output_file (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        patterns, tags, words, classes = self.generate_training_data()
        
        try:
            processed_data = {
                "words": words,
                "classes": classes,
                "patterns": [' '.join(pattern) for pattern in patterns],
                "tags": tags,
                "intents": self.combined_intents["intents"]
            }
            
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(processed_data, file, indent=2)
            
            print(f"Preprocessed data saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving preprocessed data: {str(e)}")
            return False
    
    def extend_patterns_with_synonyms(self):
        """
        Extend patterns with synonyms to improve matching.
        Uses WordNet for synonym generation.
        
        Note: This can significantly increase the training data size.
        """
        try:
            from nltk.corpus import wordnet
            nltk.download('wordnet')
        except:
            print("WordNet not available, skipping synonym extension")
            return
        
        for intent in self.combined_intents["intents"]:
            new_patterns = []
            
            for pattern in intent["patterns"]:
                tokens = self.preprocess_text(pattern)
                
                # Only consider patterns with a small number of words to avoid explosion
                if len(tokens) <= 5:
                    # Generate variants using synonyms (up to 3 per pattern to avoid explosion)
                    variants_count = 0
                    for i, word in enumerate(tokens):
                        synsets = wordnet.synsets(word)
                        for synset in synsets[:2]:  # Limit to 2 synsets per word
                            for lemma in synset.lemmas()[:2]:  # Limit to 2 synonyms per synset
                                synonym = lemma.name().replace('_', ' ')
                                if synonym != word and len(synonym) > 2:
                                    new_tokens = tokens.copy()
                                    new_tokens[i] = synonym
                                    new_pattern = ' '.join(new_tokens)
                                    if new_pattern not in intent["patterns"] and new_pattern not in new_patterns:
                                        new_patterns.append(new_pattern)
                                        variants_count += 1
                                        
                                        if variants_count >= 3:  # Limit to 3 variants per pattern
                                            break
                            if variants_count >= 3:
                                break
                        if variants_count >= 3:
                            break
            
            # Add the new patterns
            intent["patterns"].extend(new_patterns)
            
        print(f"Extended patterns with synonyms. New pattern count: {sum(len(intent['patterns']) for intent in self.combined_intents['intents'])}")

    def add_misspellings(self):
        """
        Add common misspellings for patterns to make the model more robust.
        """
        # Define common character replacements for misspellings
        replacements = {
            'a': ['e', 'i'],
            'e': ['a', 'i'],
            'i': ['e', 'a', 'y'],
            'o': ['u', 'a'],
            'u': ['o', 'a'],
            's': ['c', 'z'],
            'c': ['k', 's'],
            't': ['d'],
            'd': ['t'],
            'm': ['n'],
            'n': ['m'],
        }
        
        for intent in self.combined_intents["intents"]:
            new_patterns = []
            
            for pattern in intent["patterns"]:
                # Only process short patterns to avoid combinatorial explosion
                if len(pattern.split()) <= 3 and len(pattern) <= 20:
                    words = pattern.split()
                    
                    for word_idx, word in enumerate(words):
                        if len(word) >= 4:  # Only modify longer words
                            for char_idx, char in enumerate(word):
                                if char in replacements:
                                    for replacement in replacements[char]:
                                        misspelled_word = word[:char_idx] + replacement + word[char_idx+1:]
                                        misspelled_words = words.copy()
                                        misspelled_words[word_idx] = misspelled_word
                                        misspelled_pattern = ' '.join(misspelled_words)
                                        
                                        if misspelled_pattern not in intent["patterns"] and misspelled_pattern not in new_patterns:
                                            new_patterns.append(misspelled_pattern)
            
            # Limit the number of misspellings to add (avoid explosion)
            max_misspellings = 10
            if len(new_patterns) > max_misspellings:
                new_patterns = new_patterns[:max_misspellings]
                
            intent["patterns"].extend(new_patterns)
            
        print(f"Added misspelled patterns. New pattern count: {sum(len(intent['patterns']) for intent in self.combined_intents['intents'])}")

if __name__ == "__main__":
    # Define data files
    data_files = [
         os.path.join(BASE_DIR, "datasets", "training_data_1.json"),
         os.path.join(BASE_DIR, "datasets", "training_data_2.json"),
         os.path.join(BASE_DIR, "datasets", "training_data_3.json"),
         os.path.join(BASE_DIR, "datasets", "training_data_4.json"),
         os.path.join(BASE_DIR, "datasets", "training_data_5.json"),

    ]
    
    # Create and use the preprocessor
    try:
        preprocessor = Preprocessor(data_files)
        
        # Optional: Extend patterns with synonyms and misspellings
        # Uncomment these if you want more robust pattern matching
        # preprocessor.extend_patterns_with_synonyms()
        # preprocessor.add_misspellings()
        
        # Save the preprocessed data
        preprocessor.save_preprocessed_data()
        
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")