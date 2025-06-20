# engine.py (Super App - Phase 2: Hybrid Engine)

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.phrases import Phraser
import re

# --- Load the trained phraser model ---
try:
    phraser = Phraser.load("data/phraser.model")
except FileNotFoundError:
    phraser = None

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
    """Correctly cleans user input by processing each comma-separated ingredient individually."""
    if phraser is None:
        print("Warning: Phraser model not found.")

    ingredients = text.split(',')
    final_ingredients = set()

    for ing in ingredients:
        cleaned_text = ing.lower()
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text)
        cleaned_text = re.sub(r'[^a-z\s]', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        tokens = cleaned_text.split()
        lemmas = [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word in tokens if word not in STOPWORDS]

        if phraser and lemmas:
            phrased_lemmas = phraser[lemmas]
            final_ingredients.update(phrased_lemmas)
        else:
            final_ingredients.update(lemmas)

    return final_ingredients


# --- THE NEW HYBRID RECOMMENDATION FUNCTION ---
def recommend(user_ingredients: set, recipes_df: pd.DataFrame, top_n=10, sort_by='hybrid_score'):
    """
    Recommends recipes using a hybrid score of ingredient match and community rating.
    Allows for different sorting methods.
    """

    # --- Scoring Weights: The "Dials" of our engine ---
    # W1 is for the ingredient match, W2 is for the community rating.
    W1_INGREDIENTS = 0.7  # 70% of the score is how well it matches ingredients
    W2_RATING = 0.3  # 30% is how much the community liked it

    all_scores = []
    for recipe in recipes_df.itertuples():
        recipe_ingredients = set(recipe.ingredients_cleaned)
        matched_ingredients = user_ingredients.intersection(recipe_ingredients)
        missing_ingredients = recipe_ingredients.difference(user_ingredients)

        if not matched_ingredients:
            continue

        # Score 1: Ingredient Match Score (0 to 1)
        ingredient_score = (len(matched_ingredients) / len(recipe_ingredients)) ** 2

        # Score 2: Quality Score (normalized from 4.0-5.0 to 0-1)
        # Our data is already filtered for avg_rating >= 4.0
        quality_score = (recipe.avg_rating - 4.0) / (5.0 - 4.0)

        # Final Hybrid Score
        hybrid_score = (W1_INGREDIENTS * ingredient_score) + (W2_RATING * quality_score)

        # Penalty for missing ingredients. This is applied after scoring.
        # It ensures that even a highly-rated recipe is penalized if you're missing a lot.
        final_score = hybrid_score - (len(missing_ingredients) * 0.1)

        all_scores.append({
            'recipe_id': recipe.recipe_id,
            'title': recipe.title,
            'hybrid_score': final_score,
            'avg_rating': recipe.avg_rating,
            'num_ratings': recipe.num_ratings,
            'matched_count': len(matched_ingredients),
            'missing_count': len(missing_ingredients),
            'matched_ingredients': sorted(list(matched_ingredients)),
            'missing_ingredients': sorted(list(missing_ingredients)),
            'ingredients': recipe.ingredients,
            'steps': recipe.steps
        })

    if not all_scores:
        return pd.DataFrame()

    recs_df = pd.DataFrame(all_scores)

    # --- Sorting Logic ---
    if sort_by == 'highest_rated':
        recs_df = recs_df.sort_values(by='avg_rating', ascending=False)
    elif sort_by == 'most_popular':
        recs_df = recs_df.sort_values(by='num_ratings', ascending=False)
    else:  # Default to our hybrid score
        recs_df = recs_df.sort_values(by='hybrid_score', ascending=False)

    return recs_df.head(top_n)


# Add this function to the end of engine.py

def get_collaborative_recs(target_recipe_id: int, recipes_df: pd.DataFrame, interactions_df: pd.DataFrame, top_n=5):
    """
    Finds recipes that are frequently co-rated with the target recipe.
    This is a simple item-based collaborative filtering approach.
    """
    # 1. Find all users who gave the target recipe a 5-star rating.
    target_raters = interactions_df[
        (interactions_df['recipe_id'] == target_recipe_id) & (interactions_df['rating'] == 5)
        ]['user_id']

    if target_raters.empty:
        return pd.DataFrame()  # No 5-star raters found for this recipe

    # 2. Find all OTHER recipes that these "taste twin" users also rated 5 stars.
    similar_recipes = interactions_df[
        interactions_df['user_id'].isin(target_raters) & (interactions_df['rating'] == 5)
        ]

    # 3. Count which recipes appear most frequently among this group.
    recommendation_counts = similar_recipes['recipe_id'].value_counts().reset_index()
    recommendation_counts.columns = ['recipe_id', 'co_rating_count']

    # 4. Merge with the main recipe data to get titles and filter out the original recipe.
    recs = pd.merge(recommendation_counts, recipes_df[['recipe_id', 'title', 'avg_rating', 'num_ratings']],
                    on='recipe_id')
    recs = recs[recs['recipe_id'] != target_recipe_id]

    # Sort by how many of our "taste twins" also loved this recipe.
    recs = recs.sort_values(by='co_rating_count', ascending=False)

    return recs.head(top_n)