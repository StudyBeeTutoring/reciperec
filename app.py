# app.py
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Pantry-to-Plate",
    page_icon="🍳"
)

st.title("🍳 Pantry-to-Plate")
st.header("Find recipes with the ingredients you have on hand!")

st.write("This app is under construction. Soon, you'll be able to enter your ingredients and get recipe recommendations.")

if st.button("Test the App"):
    st.success("Success! The connection is working! 🎉")