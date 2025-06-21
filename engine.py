# engine.py (Final Intelligent Version)
import pandas as pd
import json
from gensim.models.phrases import Phraser
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# --- Load All Models and Data at Startup ---
try:
    phraser = Phraser.load("data/phraser.model")
    with open("data/idf_scores.json", 'r') as f:
        idf_scores = json.load(f)
except FileNotFoundError:
    phraser = None
    idf_scores = {}

# (Helper functions and settings remain the same)
lemmatizer = WordNetLemmatizer()
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


def clean_user_input(text: str) -> set:
    if phraser is None: print("Warning: Phraser model not found.")
    ingredients = text.split(',')
    final_ingredients = set()
    for ing in ingredients:
        cleaned_text = ing.lower();
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text);
        cleaned_text = re.sub(r'[^a-z\s]', ' ', cleaned_text);
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        tokens = cleaned_text.split()
        lemmas = [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word in tokens if word not in STOPWORDS]
        if phraser and lemmas:
            final_ingredients.update(phraser[lemmas])
        else:
            final_ingredients.update(lemmas)
    return final_ingredients


# --- The Final, Smartest Recommendation Function ---
def recommend(user_ingredients: set, recipes_full_df: pd.DataFrame, recipes_hq_df: pd.DataFrame, top_n=10,
              sort_by='best_match'):
    if sort_by == 'best_match':
        # Use the FULL dataset and the intelligent TF-IDF score
        target_df = recipes_full_df
        all_scores = []
        for recipe in target_df.itertuples():
            recipe_ingredients = set(recipe.ingredients_cleaned)
            matched = user_ingredients.intersection(recipe_ingredients)

            if not matched: continue

            # --- TF-IDF Weighted Score ---
            # Sum the "importance" of the ingredients you have
            match_score = sum(idf_scores.get(ing, 0) for ing in matched)

            # Penalty for missing ingredients remains crucial
            missing_count = len(recipe_ingredients.difference(user_ingredients))
            # Heavier penalty to prioritize recipes you can actually make
            final_score = match_score - (missing_count * 2)

            all_scores.append({'recipe_id': recipe.recipe_id, 'score': final_score, 'title': recipe.title,
                               'avg_rating': recipe.avg_rating, 'num_ratings': recipe.num_ratings,
                               'matched_count': len(matched), 'missing_count': missing_count,
                               'matched_ingredients': sorted(list(matched)),
                               'missing_ingredients': sorted(list(recipe_ingredients.difference(user_ingredients))),
                               'ingredients': recipe.ingredients, 'steps': recipe.steps})

        if not all_scores: return pd.DataFrame()
        recs_df = pd.DataFrame(all_scores).sort_values(by='score', ascending=False)

    else:
        # For Highest Rated/Popular, search within the smaller, high-quality dataset
        target_df = recipes_hq_df
        candidates = []
        for recipe in target_df.itertuples():
            # Find potential candidates by checking for at least one match
            if user_ingredients.intersection(set(recipe.ingredients_cleaned)):
                candidates.append(recipe)

        if not candidates: return pd.DataFrame()

        recs_df = pd.DataFrame(candidates)
        # Now sort these candidates by the user's choice
        if sort_by == 'highest_rated':
            recs_df = recs_df.sort_values(by='avg_rating', ascending=False)
        elif sort_by == 'most_popular':
            recs_df = recs_df.sort_values(by='num_ratings', ascending=False)

    return recs_df.head(top_n)


def get_collaborative_recs(target_recipe_id: int, collab_df: pd.DataFrame, recipes_df: pd.DataFrame, top_n=5):
    recs = collab_df[collab_df['recipe_id_x'] == target_recipe_id].sort_values(by='co_rating_count', ascending=False)
    recs = pd.merge(recs, recipes_df[['recipe_id', 'title', 'avg_rating', 'ingredients', 'steps']],
                    left_on='recipe_id_y', right_on='recipe_id')
    return recs.head(top_n)