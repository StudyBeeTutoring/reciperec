# engine.py (Super App - Final Corrected Version)
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.phrases import Phraser
import re

# (All functions up to 'recommend' remain the same as the last version)
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


def recommend(user_ingredients: set, recipes_df: pd.DataFrame, top_n=10, sort_by='hybrid_score'):
    W1_INGREDIENTS = 0.7;
    W2_RATING = 0.3
    all_scores = []
    for recipe in recipes_df.itertuples():
        recipe_ingredients = set(recipe.ingredients_cleaned)
        matched_ingredients = user_ingredients.intersection(recipe_ingredients)
        if not matched_ingredients: continue
        ingredient_score = (len(matched_ingredients) / len(recipe_ingredients)) ** 2
        quality_score = (recipe.avg_rating - 4.0)
        hybrid_score = (W1_INGREDIENTS * ingredient_score) + (W2_RATING * quality_score)
        final_score = hybrid_score - (len(recipe_ingredients.difference(user_ingredients)) * 0.1)
        all_scores.append({'recipe_id': recipe.recipe_id, 'title': recipe.title, 'hybrid_score': final_score,
                           'avg_rating': recipe.avg_rating, 'num_ratings': recipe.num_ratings,
                           'matched_count': len(matched_ingredients),
                           'missing_count': len(recipe_ingredients.difference(user_ingredients)),
                           'matched_ingredients': sorted(list(matched_ingredients)),
                           'missing_ingredients': sorted(list(recipe_ingredients.difference(user_ingredients))),
                           'ingredients': recipe.ingredients, 'steps': recipe.steps})
    if not all_scores: return pd.DataFrame()
    recs_df = pd.DataFrame(all_scores)
    if sort_by == 'highest_rated':
        recs_df = recs_df.sort_values(by='avg_rating', ascending=False)
    elif sort_by == 'most_popular':
        recs_df = recs_df.sort_values(by='num_ratings', ascending=False)
    else:
        recs_df = recs_df.sort_values(by='hybrid_score', ascending=False)
    return recs_df.head(top_n)


# --- THIS IS THE UPGRADED FUNCTION ---
def get_collaborative_recs(target_recipe_id: int, collab_df: pd.DataFrame, recipes_df: pd.DataFrame, top_n=5):
    """Finds similar recipes and now includes their full details."""
    recs = collab_df[collab_df['recipe_id_x'] == target_recipe_id].sort_values(by='co_rating_count', ascending=False)

    # --- THIS IS THE CORRECTED LINE ---
    recs = pd.merge(recs, recipes_df[['recipe_id', 'title', 'avg_rating', 'ingredients', 'steps']],
                    left_on='recipe_id_y', right_on='recipe_id')

    if recs.empty:
        return pd.DataFrame()

    return recs.head(top_n)