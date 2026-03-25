import streamlit as st
from transformers import pipeline

st.title("📰 News Topic Classifier (BERT)")

classifier = pipeline(
    "text-classification",
    model="bert_news_model",
    tokenizer="bert_news_model"
)

text = st.text_area("Enter News Headline:")

if st.button("Classify"):
    result = classifier(text)
    st.write("Prediction:", result)