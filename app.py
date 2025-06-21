# app.py (Super App - Final Version with Full Discovery UI)
import streamlit as st
import pandas as pd
import ast
import nltk
from engine import clean_user_input, recommend, get_collaborative_recs

# --- Configuration and Initial Setup ---
st.set_page_config(page_title="Pantry-to-Plate Pro", page_icon="🧑‍🍳", layout="wide")


@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip'); nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        nltk.download('wordnet'); nltk.download('omw-1.4')


download_nltk_data()


@st.cache_data
def load_data():
    try:
        recipes_df = pd.read_parquet("data/recipes_final.parquet")
        collab_df = pd.read_parquet("data/collaborative_filtering_data.parquet")
        return recipes_df, collab_df
    except FileNotFoundError as e:
        st.error(
            f"FATAL ERROR: A required data file was not found. Please re-run the processing script and ensure all .parquet files are in the 'data' folder. Details: {e}")
        return None, None


recipes_df, collab_df = load_data()
if 'recommendations' not in st.session_state: st.session_state.recommendations = None
if 'discovery_results' not in st.session_state: st.session_state.discovery_results = {}


# --- Callback Functions ---
def run_main_search(user_input, sort_by):
    st.session_state.discovery_results = {}
    cleaned_input = clean_user_input(user_input)
    if cleaned_input:
        with st.spinner("Analyzing millions of ratings to find the best recipes..."):
            st.session_state.recommendations = recommend(cleaned_input, recipes_df, top_n=10, sort_by=sort_by)
    else:
        st.warning("Please enter some valid ingredients.");
        st.session_state.recommendations = pd.DataFrame()


def run_discovery_search(recipe_id):
    with st.spinner("Searching for what other foodies loved..."):
        st.session_state.discovery_results[recipe_id] = get_collaborative_recs(recipe_id, collab_df, recipes_df)


# --- Main UI Rendering ---
st.title("🧑‍🍳 Pantry-to-Plate Pro");
st.markdown("Your intelligent recipe assistant. Find high-quality recipes with the ingredients you have!")
if recipes_df is not None:
    search_col, sort_col = st.columns([3, 1])
    with search_col:
        user_input = st.text_area("Enter your ingredients, separated by commas",
                                  "cream cheese, chicken breast, black pepper, onion", height=120)
    with sort_col:
        sort_option = st.selectbox("Sort results by:", ('Best Match', 'Highest Rated', 'Most Popular'))
        sort_map = {'Best Match': 'hybrid_score', 'Highest Rated': 'highest_rated', 'Most Popular': 'most_popular'}
        st.button("Find Recipes!", on_click=run_main_search, args=(user_input, sort_map[sort_option]), type="primary",
                  use_container_width=True)
    st.divider()

    if st.session_state.recommendations is not None:
        if not st.session_state.recommendations.empty:
            st.success(f"Found {len(st.session_state.recommendations)} top-tier recipes for you!")
            for _, recipe in st.session_state.recommendations.iterrows():
                recipe_id = recipe['recipe_id']
                with st.expander(
                        f"**{recipe['title']}** | ⭐ {recipe['avg_rating']}/5 ({recipe['num_ratings']} ratings)"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("Ingredients You Have");
                        st.success(", ".join(recipe['matched_ingredients']).replace('_', ' '))
                        st.subheader("Missing Ingredients")
                        if recipe['missing_count'] > 0:
                            st.warning(", ".join(recipe['missing_ingredients']).replace('_', ' '))
                        else:
                            st.success("You have everything!")
                        st.divider()
                        st.subheader("Discover More")
                        st.button("Find similar popular recipes", key=f"discover_{recipe_id}",
                                  on_click=run_discovery_search, args=(recipe_id,))

                        # --- THIS IS THE UPGRADED DISCOVERY DISPLAY ---
                        if recipe_id in st.session_state.discovery_results:
                            discovery_df = st.session_state.discovery_results[recipe_id]
                            if not discovery_df.empty:
                                st.markdown("--- \n ##### People who loved this recipe also enjoyed:")
                                for _, rec in discovery_df.iterrows():
                                    with st.expander(f"**{rec['title']}** (Rated ⭐ {rec['avg_rating']})"):
                                        st.subheader("Full Ingredients")
                                        try:
                                            ingredients = ast.literal_eval(rec['ingredients'])
                                            st.info(", ".join(ingredients))
                                        except:
                                            st.write("Could not parse ingredients.")
                                        st.subheader("Instructions")
                                        try:
                                            steps = ast.literal_eval(rec['steps'])
                                            for i, step in enumerate(steps, 1): st.write(f"{i}. {step}")
                                        except:
                                            st.write(rec['steps'])
                            else:
                                st.write("Couldn't find any similar popular recipes for this one.")
                    with col2:
                        st.subheader("Instructions")
                        try:
                            steps = ast.literal_eval(recipe['steps'])
                            for i, step in enumerate(steps, 1): st.write(f"{i}. {step}")
                        except:
                            st.write(recipe['steps'])
        else:
            st.error("Sorry, couldn't find any high-quality recipes with those ingredients.")
else:
    st.stop()