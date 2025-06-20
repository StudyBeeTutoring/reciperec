# app.py (Final Deployment-Ready Version)
import streamlit as st
import pandas as pd
import ast
import nltk  # Import NLTK
from engine import clean_user_input, recommend


# --- NLTK Data Download ---
# This is a one-time setup for the Streamlit server.
# It checks if the data is present and downloads it if not.
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
# --- End of NLTK Setup ---


# --- Page Configuration ---
st.set_page_config(
    page_title="Pantry-to-Plate",
    page_icon="🍳",
    layout="wide"
)


# --- Data Loading ---
@st.cache_data
def load_data(path):
    """Loads the pre-processed recipe data."""
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {path}. Please check your GitHub repository.")
        return None


# Load the data
recipes_df = load_data("data/recipes_cleaned.parquet")

# --- UI ---
st.title("🍳 Pantry-to-Plate")
st.markdown("Got a bunch of ingredients and no ideas? I'm here to help! \n"
            "Enter the ingredients you have, and I'll suggest some recipes.")

if recipes_df is not None:
    with st.form(key="ingredient_form"):
        user_input = st.text_area(
            "Enter your ingredients, separated by commas",
            "cream cheese, chicken breast, black pepper, onion",
            height=100
        )
        submit_button = st.form_submit_button(label="Find Recipes!")

    if submit_button and user_input:
        cleaned_input = clean_user_input(user_input)

        if not cleaned_input:
            st.warning("Please enter some valid ingredients.")
        else:
            parsed_input_str = ", ".join(f"`{item.replace('_', ' ')}`" for item in cleaned_input)
            st.info(f"Searching for recipes with: {parsed_input_str}")

            with st.spinner("Finding the best recipes for you... 🧑‍🍳"):
                recommendations_df = recommend(cleaned_input, recipes_df, top_n=10)

            if not recommendations_df.empty:
                st.success(f"Found {len(recommendations_df)} great recipes for you!")

                for _, recipe in recommendations_df.iterrows():
                    title = recipe['title']
                    matched = recipe['matched_count']
                    missing = recipe['missing_count']

                    with st.expander(f"**{title}** - ({matched} ingredients match, {missing} missing)"):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.subheader("Ingredients You Have")
                            st.success(", ".join(recipe['matched_ingredients']).replace('_', ' '))

                            st.subheader("Missing Ingredients")
                            if recipe['missing_count'] > 0:
                                st.warning(", ".join(recipe['missing_ingredients']).replace('_', ' '))
                            else:
                                st.success("You have everything!")

                        with col2:
                            st.subheader("Instructions")
                            try:
                                steps = ast.literal_eval(recipe['steps'])
                                for i, step in enumerate(steps, 1):
                                    st.write(f"{i}. {step}")
                            except:
                                st.write(recipe['steps'])
            else:
                st.error("Sorry, I couldn't find any recipes with those ingredients. Try adding a few more!")
else:
    st.stop()