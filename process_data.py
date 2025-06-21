# process_data.py (Final Intelligent & Memory-Optimized Version - Corrected)
import pandas as pd
import numpy as np  # Import numpy directly
import ast
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from tqdm import tqdm

# --- Configuration ---
RAW_RECIPES_PATH = 'data/RAW_recipes.csv'
RAW_INTERACTIONS_PATH = 'data/RAW_interactions.csv'
FULL_DATA_PATH = 'data/recipes_full_with_idf.parquet'
HQ_DATA_PATH = 'data/recipes_high_quality.parquet'
COLLAB_DATA_PATH = 'data/collaborative_filtering_data.parquet'
PHRASER_MODEL_PATH = 'data/phraser.model'
IDF_SCORES_PATH = 'data/idf_scores.json'
# --- All helper functions and settings remain the same ---
MIN_RATINGS = 10;
MIN_AVG_RATING = 4.0;
PHRASE_MODEL_MIN_COUNT = 3;
PHRASE_MODEL_SCORING = 'npmi';
PHRASE_MODEL_THRESHOLD = 0.2


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


# --- The crucial Memory Optimization Function ---
def optimize_memory(df, df_name=""):
    start_mem = df.memory_usage().sum() / 1024 ** 2;
    print(f'Optimizing {df_name}... Initial memory usage: {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min();
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2;
    print(f'Final memory usage: {end_mem:.2f} MB. Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%');
    return df


if __name__ == "__main__":
    setup_nltk();
    print("\n--- Final Intelligent & Memory-Optimized Data Build ---")
    print("[1/5] Loading all recipes and training phrase model...");
    df_recipes = pd.read_csv(RAW_RECIPES_PATH);
    df_recipes.rename(columns={'id': 'recipe_id'}, inplace=True);
    df_recipes['ingredients_parsed'] = df_recipes['ingredients'].apply(ast.literal_eval)
    master_ingredient_list = [token for recipe_ingredients in
                              tqdm(df_recipes['ingredients_parsed'], desc="     - Compiling master list") for
                              ingredient_text in recipe_ingredients for token in
                              tokenize_and_lemmatize(clean_ingredient_text(ingredient_text))]
    phrases = Phrases(master_ingredient_list, min_count=PHRASE_MODEL_MIN_COUNT, threshold=PHRASE_MODEL_THRESHOLD,
                      scoring=PHRASE_MODEL_SCORING, delimiter='_');
    phraser = Phraser(phrases);
    phraser.save(PHRASER_MODEL_PATH)
    print("[2/5] Applying phraser and calculating TF-IDF scores...")


    def apply_phraser_to_recipe(ingredient_list): cleaned_set = set(); [cleaned_set.update(phraser[tokens]) for text in
                                                                        ingredient_list if (
                                                                            tokens := tokenize_and_lemmatize(
                                                                                clean_ingredient_text(
                                                                                    text)))]; return sorted(
        list(cleaned_set))


    tqdm.pandas(desc="     - Applying phraser");
    df_recipes['ingredients_cleaned'] = df_recipes['ingredients_parsed'].progress_apply(apply_phraser_to_recipe)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x);
    tfidf_matrix = vectorizer.fit_transform(df_recipes['ingredients_cleaned'])
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_));
    with open(IDF_SCORES_PATH, 'w') as f: json.dump(idf_scores, f)
    print("[3/5] Creating full and high-quality datasets...")
    df_interactions = pd.read_csv(RAW_INTERACTIONS_PATH);
    ratings_summary = df_interactions.groupby('recipe_id')['rating'].agg(['mean', 'count']).reset_index();
    ratings_summary.columns = ['recipe_id', 'avg_rating', 'num_ratings']

    # --- THIS IS THE FIX ---
    # Merge first, THEN fill missing values only in specific columns.
    df_full = pd.merge(df_recipes, ratings_summary, on='recipe_id', how='left')
    df_full[['avg_rating', 'num_ratings']] = df_full[['avg_rating', 'num_ratings']].fillna(0)

    df_full_final = df_full.rename(columns={'name': 'title'})[
        ['recipe_id', 'title', 'minutes', 'tags', 'steps', 'ingredients', 'ingredients_cleaned', 'avg_rating',
         'num_ratings']].copy()
    df_hq = df_full[(df_full['num_ratings'] >= MIN_RATINGS) & (df_full['avg_rating'] >= MIN_AVG_RATING)]
    df_hq_final = df_hq.rename(columns={'name': 'title'})[
        ['recipe_id', 'title', 'minutes', 'tags', 'steps', 'ingredients', 'ingredients_cleaned', 'avg_rating',
         'num_ratings']].copy()
    print("[4/5] Pre-calculating discovery data...");
    user_rating_counts = df_interactions['user_id'].value_counts();
    active_users = user_rating_counts[user_rating_counts >= 5].index
    df_5star = df_interactions[(df_interactions['rating'] == 5) & (df_interactions['user_id'].isin(active_users))]
    merged = pd.merge(df_5star, df_5star, on='user_id');
    co_ratings = merged[merged['recipe_id_x'] != merged['recipe_id_y']]
    co_rating_counts = co_ratings.groupby(['recipe_id_x', 'recipe_id_y']).size().reset_index(name='co_rating_count')
    co_rating_counts = co_rating_counts[co_rating_counts['co_rating_count'] >= 2]
    print("\n[5/5] Optimizing and saving final data files...")
    df_full_optimized = optimize_memory(df_full_final, "Full Dataset")
    df_full_optimized.to_parquet(FULL_DATA_PATH, index=False)
    df_hq_optimized = optimize_memory(df_hq_final, "High-Quality Dataset")
    df_hq_optimized.to_parquet(HQ_DATA_PATH, index=False)
    co_rating_counts_optimized = optimize_memory(co_rating_counts, "Discovery Dataset")
    co_rating_counts_optimized.to_parquet(COLLAB_DATA_PATH, index=False)
    print("\n--- ✅ SUCCESS! All intelligent and memory-optimized data is built. ---")