# app.py (Super App - Final UI)
import streamlit as st
import pandas as pd
import ast
import nltk
# Import ALL our engine functions
from engine import clean_user_input, recommend, get_collaborative_recs


# --- NLTK Data Download (Essential for deployment) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        nltk.download('omw-1.4')


download_nltk_data()

# --- Page Configuration ---
st.set_page_config(
    page_title="Pantry-to-Plate Pro",
    page_icon="🧑‍🍳",
    layout="wide"
)


# --- Data Loading (Loading our new, final data files) ---
@st.cache_data
def load_data():
    """Loads the final pre-processed recipe data and interaction data."""
    try:
        recipes_df = pd.read_parquet("data/recipes_final.parquet")
        # Also load the raw interactions for the discovery engine
        interactions_df = pd.read_csv("data/RAW_interactions.csv")
        return recipes_df, interactions_df
    except FileNotFoundError as e:
        st.error(
            f"Data file not found. Please ensure 'recipes_final.parquet' and 'RAW_interactions.csv' are in the 'data' folder. Error: {e}")
        return None, None


recipes_df, interactions_df = load_data()

# --- UI ---
st.title("🧑‍🍳 Pantry-to-Plate Pro")
st.markdown("Your intelligent recipe assistant. Find high-quality recipes with the ingredients you have!")

if recipes_df is not None:
    # --- SEARCH AND SORTING ---
    with st.form(key="ingredient_form"):
        user_input = st.text_area(
            "Enter your ingredients, separated by commas",
            "cream cheese, chicken breast, black pepper, onion",
            height=100
        )
        # Add the sorting options to the form
        sort_option = st.selectbox(
            "Sort results by:",
            ('Best Match', 'Highest Rated', 'Most Popular')
        )
        submit_button = st.form_submit_button(label="Find Recipes!")

    # Map user-friendly sort options to the engine's expected values
    sort_map = {
        'Best Match': 'hybrid_score',
        'Highest Rated': 'highest_rated',
        'Most Popular': 'most_popular'
    }
    engine_sort_by = sort_map[sort_option]

    if submit_button and user_input:
        cleaned_input = clean_user_input(user_input)

        if not cleaned_input:
            st.warning("Please enter some valid ingredients.")
        else:
            parsed_input_str = ", ".join(f"`{item.replace('_', ' ')}`" for item in cleaned_input)
            st.info(f"Searching for recipes with: {parsed_input_str}")

            with st.spinner("Analyzing millions of ratings to find the best recipes..."):
                recommendations_df = recommend(cleaned_input, recipes_df, top_n=10, sort_by=engine_sort_by)

            # --- DISPLAY RESULTS ---
            if not recommendations_df.empty:
                st.success(f"Found {len(recommendations_df)} top-tier recipes for you!")

                for _, recipe in recommendations_df.iterrows():
                    title = recipe['title']
                    # Display the new quality metrics
                    expander_title = f"**{title}** | ⭐ {recipe['avg_rating']}/5 ({recipe['num_ratings']} ratings)"

                    with st.expander(expander_title):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.subheader("Ingredients You Have")
                            st.success(", ".join(recipe['matched_ingredients']).replace('_', ' '))

                            st.subheader("Missing Ingredients")
                            if recipe['missing_count'] > 0:
                                st.warning(", ".join(recipe['missing_ingredients']).replace('_', ' '))
                            else:
                                st.success("You have everything!")

                            st.divider()

                            # --- DISCOVERY ENGINE UI ---
                            st.subheader("Discover More")
                            # Use a unique key for each button based on recipe_id
                            if st.button("Find similar popular recipes", key=f"discover_{recipe['recipe_id']}"):
                                with st.spinner("Searching for what other foodies loved..."):
                                    collab_recs = get_collaborative_recs(recipe['recipe_id'], recipes_df,
                                                                         interactions_df)
                                if not collab_recs.empty:
                                    st.write("People who loved this recipe also enjoyed:")
                                    for _, rec in collab_recs.iterrows():
                                        st.write(f"- **{rec['title']}** (Rated ⭐ {rec['avg_rating']})")
                                else:
                                    st.write("Couldn't find any similar popular recipes.")

                        with col2:
                            st.subheader("Instructions")
                            try:
                                steps = ast.literal_eval(recipe['steps'])
                                for i, step in enumerate(steps, 1):
                                    st.write(f"{i}. {step}")
                            except:
                                st.write(recipe['steps'])
            else:
                st.error(
                    "Sorry, couldn't find any high-quality recipes with those ingredients. Try a different combination!")
else:
    st.stop()