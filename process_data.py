# process_data.py (Re-tuned Version)

import pandas as pd
import ast
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm
import time

# --- Configuration ---
RAW_DATA_PATH = 'data/RAW_recipes.csv'
PROCESSED_DATA_PATH = 'data/recipes_cleaned.parquet'
PHRASER_MODEL_PATH = 'data/phraser.model'  # Path to save the trained phraser
PHRASE_MODEL_MIN_COUNT = 5
# LOWERED THRESHOLD TO BE MORE AGGRESSIVE WITH PHRASES
PHRASE_MODEL_THRESHOLD = 5

print("Script started: Re-tuned Data Processing.")


# (All functions like setup_nltk, stopwords, etc., remain the same)
# ... [Keeping the rest of the functions from the previous version] ...
def setup_nltk():
    print("Setting up NLTK resources...")
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    print("NLTK setup complete.")


STOPWORDS = set(
    ['cup', 'cups', 'oz', 'ounce', 'ounces', 'teaspoon', 'teaspoons', 'tsp', 'tablespoon', 'tablespoons', 'tbsp',
     'pound', 'pounds', 'lb', 'lbs', 'pinch', 'dash', 'clove', 'cloves', 'can', 'cans', 'package', 'packages', 'bunch',
     'slices', 'slice', 'head', 'heads', 'gallon', 'gallons', 'quart', 'quarts', 'pint', 'pints', 'jar', 'jars',
     'container', 'containers', 'box', 'boxes', 'stick', 'sticks', 'bag', 'bags', 'chopped', 'diced', 'sliced',
     'minced', 'fresh', 'dried', 'freshly', 'ground', 'crushed', 'peeled', 'seeded', 'halved', 'quartered', 'cubed',
     'finely', 'coarsely', 'thinly', 'thickly', 'optional', 'to', 'taste', 'and', 'or', 'of', 'a', 'an', 'the', 'with',
     'for', 'in', 'at', 'on', 'plus', 'into', 'as', 'needed', 'about', 'more', 'less', 'large', 'medium', 'small',
     'jumbo', 'miniature', 'ripe', 'cold', 'hot', 'warm', 'room', 'temperature', 'boneless', 'skinless', 'cooked',
     'uncooked', 'softened', 'melted', 'frozen', 'thawed', 'drained', 'rinsed', 'divided', 'inch', 'inches'])
lemmatizer = WordNetLemmatizer()


def clean_ingredient_text(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_lemmatize(text):
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word in tokens if word not in STOPWORDS]
    return lemmas


if __name__ == "__main__":
    setup_nltk()

    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Parsing and cleaning ingredient lists...")
    tqdm.pandas(desc="Parsing ingredients")
    df['ingredients_parsed'] = df['ingredients'].progress_apply(ast.literal_eval)

    tqdm.pandas(desc="Cleaning text")
    corpus = df['ingredients_parsed'].progress_apply(
        lambda ingredients: [clean_ingredient_text(ing) for ing in ingredients])

    tqdm.pandas(desc="Tokenizing & Lemmatizing")
    all_lemmatized_ingredients = corpus.progress_apply(
        lambda ingredients: [token for ing_text in ingredients for token in tokenize_and_lemmatize(ing_text)])

    print("\nBuilding a more aggressive phrase model...")
    start_time = time.time()
    phrases = Phrases(all_lemmatized_ingredients, min_count=PHRASE_MODEL_MIN_COUNT, threshold=PHRASE_MODEL_THRESHOLD,
                      delimiter='_')
    phraser = Phraser(phrases)
    print(f"Phrase model built in {time.time() - start_time:.2f} seconds.")

    # --- NEW: SAVE THE TRAINED PHRASER MODEL ---
    print(f"Saving phrase model to {PHRASER_MODEL_PATH}...")
    phraser.save(PHRASER_MODEL_PATH)

    print("Applying phrase model to ingredients...")
    tqdm.pandas(desc="Applying phrases")
    df['ingredients_cleaned'] = all_lemmatized_ingredients.progress_apply(
        lambda ingredients: sorted(list(set(phraser[ingredients]))))

    print("\nFinalizing and saving data...")
    final_df = df[['name', 'id', 'minutes', 'tags', 'n_steps', 'steps', 'ingredients', 'ingredients_cleaned']]
    final_df = final_df.rename(columns={'name': 'title'})
    final_df = final_df[final_df['ingredients_cleaned'].apply(lambda x: len(x) > 0)]
    final_df.to_parquet(PROCESSED_DATA_PATH)

    print("\n--- Script Finished Successfully! ---")