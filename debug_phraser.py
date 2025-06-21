# debug_phraser.py (Corrected)
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.phrases import Phraser
import re

print("Loading the phraser model from data/phraser.model...")
try:
    phraser = Phraser.load("data/phraser.model")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: data/phraser.model not found!")
    exit()

lemmatizer = WordNetLemmatizer()
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text).strip()
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word in tokens]
    return lemmas

test_phrases = ["cream cheese", "black pepper", "chicken breast", "olive oil", "soy sauce", "all purpose flour"]
print("\n--- Running Tests ---")

for phrase in test_phrases:
    tokens = clean_and_tokenize(phrase)
    result = phraser[tokens]
    print(f"Input: '{phrase}'")
    print(f"  - Tokens given to model: {tokens}")
    print(f"  - Result from phraser:   {result}")
    print("-" * 20)

print("\n--- Inspecting the phraser model ---")
# This part is now fixed to correctly inspect the model's vocabulary
try:
    cheese_phrases = {tuple(k.decode('utf-8') for k in key): val for key, val in phraser.phrasegrams.items() if b'cheese' in key}
    print("Some learned phrases containing 'cheese':")
    # Print only the top 5 for brevity
    for i, (phrase, score) in enumerate(cheese_phrases.items()):
        if i >= 5: break
        print(f"  - Phrase: {phrase}, Score: {score}")
except Exception as e:
    print(f"Could not inspect model vocabulary. Error: {e}")