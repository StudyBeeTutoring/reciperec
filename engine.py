# engine.py (Final, Corrected Version)
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.phrases import Phraser
import re

# --- LOAD THE TRAINED PHRASER MODEL ---
try:
    phraser = Phraser.load("data/phraser.model")
except FileNotFoundError:
    phraser = None

lemmatizer = WordNetLemmatizer()
# (Stopwords list remains the same)
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


# --- THIS IS THE CORRECTED FUNCTION ---
def clean_user_input(text: str) -> set:
    """
    Correctly cleans user input by processing each comma-separated ingredient individually.
    """
    if phraser is None:
        print("Warning: Phraser model not found. Phrase detection will not work.")

    # Split the user input string into a list of individual ingredients
    ingredients = text.split(',')

    final_ingredients = set()

    for ing in ingredients:
        # Apply the same text cleaning as the original processing script
        cleaned_text = ing.lower()
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text)
        cleaned_text = re.sub(r'[^a-z\s]', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Tokenize and lemmatize
        tokens = cleaned_text.split()
        lemmas = [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word in tokens if word not in STOPWORDS]

        # Apply the phraser to the small list of tokens for THIS ingredient
        if phraser and lemmas:
            phrased_lemmas = phraser[lemmas]
            final_ingredients.update(phrased_lemmas)
        else:
            final_ingredients.update(lemmas)

    return final_ingredients


def recommend(user_ingredients: set, recipes_df: pd.DataFrame, top_n=10):
    """Recommends recipes based on user ingredients."""
    # This function remains the same as it was already correct
    all_scores = []
    for recipe in recipes_df.itertuples():
        recipe_ingredients = set(recipe.ingredients_cleaned)
        matched_ingredients = user_ingredients.intersection(recipe_ingredients)
        missing_ingredients = recipe_ingredients.difference(user_ingredients)

        if not matched_ingredients:
            continue

        match_score = (len(matched_ingredients) / len(recipe_ingredients)) ** 2
        missing_penalty = len(missing_ingredients) * 2
        final_score = match_score - missing_penalty

        all_scores.append({
            'id': recipe.id,
            'title': recipe.title,
            'score': final_score,
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
    recs_df = recs_df.sort_values(by='score', ascending=False)

    return recs_df.head(top_n)