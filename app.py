import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from textblob import TextBlob
import pandas as pd

# ---------------- Load Environment ----------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- Sentiment Function ----------------
def basic_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# ---------------- Groq Emotion Analysis ----------------
def analyze_with_groq(text):
    prompt = f"""
Analyze the following text and give:
1. Sentiment (Positive / Negative / Neutral)
2. Emotion (Joy, Anger, Sadness, Fear, Surprise, Neutral)

Text:
{text}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("💬 Sentiment Analysis App")

# --------- TYPE SELECTION (as requested) ----------
data_type = st.selectbox(
    "Select Data Source Type",
    [
        "Amazon Reviews",
        "Social Media",
        "News Sites"
    ]
)

# ---------------- Amazon Reviews ----------------
if data_type == "Amazon Reviews":
    file = st.file_uploader("Upload Amazon Reviews CSV (column name: review)", type="csv")

    if file:
        df = pd.read_csv(file)
        df["Sentiment"] = df["review"].apply(basic_sentiment)

        st.subheader("📄 Review Analysis")
        st.dataframe(df)

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(df["Sentiment"].value_counts())

# ---------------- Social Media ----------------
elif data_type == "Social Media":
    text = st.text_area("Paste social media content (tweets, comments, posts)")

    if st.button("Analyze Social Media"):
        if text.strip():
            st.subheader("📊 Basic Sentiment")
            st.write(basic_sentiment(text))

            st.subheader("🤖 Emotion Analysis")
            st.write(analyze_with_groq(text))
        else:
            st.warning("Please enter text")

# ---------------- News Sites ----------------
elif data_type == "News Sites":
    text = st.text_area("Paste news article or headline")

    if st.button("Analyze News"):
        if text.strip():
            st.subheader("📊 Basic Sentiment")
            st.write(basic_sentiment(text))

            st.subheader("🤖 Emotion Analysis")
            st.write(analyze_with_groq(text))
        else:
            st.warning("Please enter text")
