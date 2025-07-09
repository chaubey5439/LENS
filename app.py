import streamlit as st
import joblib
import random
import time
import pandas as pd
from poster_down import download_posters

# Load models
lang_model = joblib.load("Language_detection.pkl")
review_model = joblib.load("Restaurant_reviews.pkl")
spam_model = joblib.load("Spam_detection.pkl")
vectorizer = joblib.load("vectorizer.pkl")
nn_model = joblib.load("nearest_neighbors.pkl")
movie_df = pd.read_csv("final_movies_data.csv")

# Fun facts (updated with movie facts)
fun_facts = [
    "🌍 Over 7,000 languages are spoken worldwide!",
    "🤖 ML is just advanced pattern recognition!",
    "💬 Emojis are technically part of modern language!",
    "🍽️ Reviews can make or break a restaurant!",
    "📧 85% of all emails are spam globally!",
    "📰 News classification helps filter fake news!",
    "🎬 The first movie ever made was 'Roundhay Garden Scene' in 1888!",
    "🍿 The most expensive movie ever made is 'Pirates of the Caribbean: On Stranger Tides'!",
    "🎞️ The longest film ever made is over 85 hours long!",
    "🏆 The Oscars were first held in 1929, with just 12 categories.",
    "🎥 The word 'movie' comes from 'moving pictures'.",
    "🎬 Bollywood produces over 1,500 films per year—more than Hollywood!",
    "👀 IMDB stands for Internet Movie Database!",
    "💡 AI is now used in film editing, scriptwriting, and casting decisions!"
]

# Page setup
st.set_page_config(page_title="LENS eXpert (NLP Suites)", page_icon="🧠", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #121212;
        color: #f0f0f0;
    }

    .header {
        background: linear-gradient(90deg, #8e2de2, #4a00e0);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 34px;
        font-weight: bold;
        margin-bottom: 30px;
    }

    .section {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 15px;
        margin-top: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: #ffffff;
    }

    .result {
        background-color: #00f2c3;
        padding: 15px;
        border-radius: 12px;
        font-size: 19px;
        color: #000;
        margin-top: 20px;
        font-weight: bold;
    }

    .fun-fact {
        background: linear-gradient(to right, #43e97b, #38f9d7);
        color: #121212;
        padding: 12px 20px;
        border-radius: 50px;
        text-align: center;
        font-size: 16px;
        width: fit-content;
        margin: 15px auto 10px auto;
        font-weight: bold;
    }

    .poster-img {
        width: 120px;
        height: auto;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    .movie-block {
        text-align: center;
        margin-bottom: 30px;
        background-color: #1f1f1f;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }

    .movie-title {
        font-weight: 700;
        margin-top: 8px;
        font-size: 15px;
        color: #00f2c3;
    }

    .movie-info {
        font-size: 13px;
        color: #ccc;
        line-height: 1.4;
    }

    .footer {
        margin-top: 30px;
        text-align: center;
        font-size: 14px;
        color: #888;
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🧠 LENS eXpert (NLP Suites)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("robot_img.png", width=180)
    tool = st.radio("📂 Choose a Tool", [
        "🌍 Language Detection",
        "⭐ Restaurant Reviews",
        "📨 Spam Classifier",
        "📰 News Classifier",
        "🎬 Movie Recommender"
    ])

    with st.expander("ℹ️ More Info"):
        st.subheader("About Us")
        st.write("""
            This project uses Machine Learning, Natural Language Processing (NLP), 
            Scikit-learn, NLTK, and many other powerful libraries to provide smart text analysis tools.
        """)
        st.subheader("Contact")
        st.write("📧 chaubey5439@gmail.com")

# Fun fact rotation
if tool == "🎬 Movie Recommender":
    # Pick movie-related fun fact only
    movie_facts = [fact for fact in fun_facts if "🎬" in fact or "🎥" in fact or "🍿" in fact or "🏆" in fact]
    st.session_state.current_fact = random.choice(movie_facts)

if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
    st.session_state.current_fact = random.choice(fun_facts)

if time.time() - st.session_state.last_update > 5:
    st.session_state.current_fact = random.choice(fun_facts)
    st.session_state.last_update = time.time()

st.markdown(f'<div class="fun-fact">💡 {st.session_state.current_fact}</div>', unsafe_allow_html=True)

# --------- Tool Logic ---------

if tool == "🌍 Language Detection":
    st.markdown('<div class="section">🌐 <strong>Language Detection</strong></div>', unsafe_allow_html=True)
    text = st.text_input("✍️ Enter a sentence:")
    if st.button("🚀 Detect Language"):
        if text.strip():
            pred = lang_model.predict([text])[0]
            flag = {"English": "🇬🇧", "French": "🇫🇷", "Hindi": "🇮🇳", "Spanish": "🇪🇸", "German": "🇩🇪"}.get(pred, "🌍")
            st.markdown(f'<div class="result">🗣️ {flag} {pred}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a sentence.")

elif tool == "⭐ Restaurant Reviews":
    st.markdown('<div class="section">⭐ <strong>Review Sentiment Analyzer</strong></div>', unsafe_allow_html=True)
    text = st.text_area("📝 Enter review text:")
    if st.button("🔍 Analyze Review"):
        if text.strip():
            pred = review_model.predict([text])[0]
            if str(pred).lower() in ["1", "positive"]:
                st.markdown(f'<div class="result">😄 Positive Review!</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="result">😞 Negative Review</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a review.")

elif tool == "📨 Spam Classifier":
    st.markdown('<div class="section">📬 <strong>Spam Detection</strong></div>', unsafe_allow_html=True)
    text = st.text_area("📨 Enter message:")
    if st.button("⚡ Detect Spam"):
        if text.strip():
            pred = spam_model.predict([text])[0]
            if pred == "spam":
                st.markdown('<div class="result">⚠️ SPAM</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result">✅ Not Spam</div>', unsafe_allow_html=True)
                st.balloons()
        else:
            st.warning("Please enter a message.")

elif tool == "📰 News Classifier":
    st.markdown('<div class="section">📰 <strong>News Classifier</strong></div>', unsafe_allow_html=True)
    st.info("🚧 This feature is under construction. Stay tuned!")

elif tool == "🎬 Movie Recommender":
    st.markdown('<div class="section">🎬 <strong>Movie Recommender</strong></div>', unsafe_allow_html=True)
    movie_name = st.selectbox("🎥 Choose a movie", sorted(movie_df['name'].unique()))
    api_key = "a6185303"

    if st.button("Recommend"):
        idx = movie_df[movie_df['name'].str.lower() == movie_name.lower()].index[0]
        tag_vec = vectorizer.transform([movie_df.loc[idx, 'tag']]).toarray()
        _, inds = nn_model.kneighbors(tag_vec, n_neighbors=6)
        recommended_idx = inds[0][1:]
        posters = download_posters(recommended_idx, movie_df, api_key)

        for i in range(0, len(posters), 3):
            row = st.columns(3)
            for j, col in enumerate(row):
                if i + j < len(posters):
                    title, path, info = posters[i + j]
                    with col:
                        st.markdown('<div class="movie-block">', unsafe_allow_html=True)
                        if path:
                            st.markdown(f'<img src="{path}" class="poster-img">', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="color:#aaa;">🎞️ Poster not available</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-title">🎬 {title}</div>', unsafe_allow_html=True)
                        if info:
                            st.markdown(f'<div class="movie-info">⭐ {info.get("imdbRating", "N/A")}<br>🎭 {info.get("Genre", "N/A")}<br>📅 {info.get("Year", "N/A")}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('<div class="footer">🚀 Built with Streamlit, ML, and Good Vibes!</div>', unsafe_allow_html=True)
