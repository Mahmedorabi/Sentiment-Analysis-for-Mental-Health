import streamlit as st
from textblob import TextBlob

st.title("🔤 Simple Spell Checker")

text = st.text_area("Enter text:", "I havv goood speling!")

if st.button("Correct"):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    st.write("**Corrected text:**", corrected_text)
