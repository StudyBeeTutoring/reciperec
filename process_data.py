# process_data.py (Super App - Final Build, Corrected)

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
RAW_RECIPES_PATH = 'data/RAW_recipes.csv'
RAW_INTERACTIONS_PATH = 'data/RAW_interactions.csv'
FINAL_DATA_PATH = 'data/recipes_final.parquet'
PHRASER_MODEL_PATH = 'data/phraser.model'
COLLAB_DATA_PATH = 'data/collaborative_filtering_data.parquet'

MIN_RATINGS = 10
MIN_AVG_RATING = 4.0
PHRASE_MODEL_MIN_COUNT = 3
PHRASE_MODEL_SCORING = 'npmi'
PHRASE_MODEL_THRESHOLD = 0.2

print("--- Super App Data Engine: Final Build ---")


# (Helper functions remain the same)
def setup_nltk(): nltk.data.find('corpora/wordnet.zip'); nltk.data.find('corpora/omw-1.4.zip')


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


def clean_ingredient_text(text): text = text.lower(); text = re.sub(r'\([^)]*\)', '', text); text = re.sub(r'[^a-z\s]',
                                                                                                           ' ',
                                                                                                           text); text = re.sub(
    r'\s+', ' ', text).strip(); return text


def tokenize_and_lemmatize(text): tokens = text.split(); lemmas = [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word
                                                                   in tokens if word not in STOPWORDS]; return lemmas


if __name__ == "__main__":
    setup_nltk()

    print("\n[1/6] Loading raw data...")
    df_recipes = pd.read_csv(RAW_RECIPES_PATH)
    df_interactions = pd.read_csv(RAW_INTERACTIONS_PATH)

    print("[2/6] Calculating and filtering for high-quality recipes...")
    ratings_summary = df_interactions.groupby('recipe_id')['rating'].agg(['mean', 'count']).reset_index()
    ratings_summary.columns = ['recipe_id', 'avg_rating', 'num_ratings']
    quality_recipes_ids = ratings_summary[
        (ratings_summary['num_ratings'] >= MIN_RATINGS) & (ratings_summary['avg_rating'] >= MIN_AVG_RATING)]
    df_recipes.rename(columns={'id': 'recipe_id'}, inplace=True)
    df_hq = pd.merge(df_recipes, quality_recipes_ids, on='recipe_id', how='inner')

    print("[3/6] Processing ingredients for phrase modeling...")
    df_hq['ingredients_parsed'] = df_hq['ingredients'].apply(ast.literal_eval)

    master_ingredient_list = []
    for recipe_ingredients in tqdm(df_hq['ingredients_parsed'], desc="     - Compiling master list"):
        for ingredient_text in recipe_ingredients:
            tokens = tokenize_and_lemmatize(clean_ingredient_text(ingredient_text))
            if tokens:
                master_ingredient_list.append(tokens)

    print("[4/6] Training phrase model...")
    phrases = Phrases(master_ingredient_list, min_count=PHRASE_MODEL_MIN_COUNT, threshold=PHRASE_MODEL_THRESHOLD,
                      scoring=PHRASE_MODEL_SCORING, delimiter='_')
    phraser = Phraser(phrases)
    phraser.save(PHRASER_MODEL_PATH)


    def apply_phraser_to_recipe(ingredient_list):
        cleaned_set = set()
        for text in ingredient_list:
            tokens = tokenize_and_lemmatize(clean_ingredient_text(text))
            if tokens: cleaned_set.update(phraser[tokens])
        return sorted(list(cleaned_set))


    # --- THIS IS THE FIX ---
    tqdm.pandas(desc="     - Applying new phraser to all recipes")
    df_hq['ingredients_cleaned'] = df_hq['ingredients_parsed'].progress_apply(apply_phraser_to_recipe)

    print("[5/6] Pre-calculating discovery data...")
    user_rating_counts = df_interactions['user_id'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= 5].index
    df_5star = df_interactions[(df_interactions['rating'] == 5) & (df_interactions['user_id'].isin(active_users))]
    merged = pd.merge(df_5star, df_5star, on='user_id')
    co_ratings = merged[merged['recipe_id_x'] != merged['recipe_id_y']]
    co_rating_counts = co_ratings.groupby(['recipe_id_x', 'recipe_id_y']).size().reset_index(name='co_rating_count')
    co_rating_counts = co_rating_counts[co_rating_counts['co_rating_count'] >= 2]
    co_rating_counts.to_parquet(COLLAB_DATA_PATH, index=False)
    print(f"     ... Saved pre-calculated discovery data to {COLLAB_DATA_PATH}")

    print("[6/6] Finalizing and saving main dataset...")
    # I have simplified the column selection and renaming for clarity
    df_final = df_hq.rename(columns={'name': 'title'})
    final_columns = ['recipe_id', 'title', 'minutes', 'tags', 'steps', 'ingredients', 'ingredients_cleaned',
                     'avg_rating', 'num_ratings']
    df_final = df_final[final_columns].copy()
    df_final['avg_rating'] = df_final['avg_rating'].round(2)
    df_final.to_parquet(FINAL_DATA_PATH, index=False)

    print("\n--- ✅ SUCCESS! All data for the Super App is built and optimized. ---")