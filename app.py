import streamlit as st
import joblib
import random
import time
import pandas as pd

# Load models
lang_model = joblib.load("Language_detection.pkl")
review_model = joblib.load("Restaurant_reviews.pkl")
spam_model = joblib.load("Spam_detection.pkl")
# News classifier under construction

# Fun facts
fun_facts = [
    "🌍 Over 7,000 languages are spoken worldwide!",
    "🤖 ML is just advanced pattern recognition!",
    "💬 Emojis are technically part of modern language!",
    "🍽️ Reviews can make or break a restaurant!",
    "📧 85% of all emails are spam globally!",
    "📰 News classification helps filter fake news!",
]

# Page setup
st.set_page_config(page_title="LENS eXpert (NLP Suites)", page_icon="🧠", layout="centered")

# CSS for design and floating fun facts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8f9fa; }
    .header { background: linear-gradient(90deg, #2c3e50, #3498db); padding: 25px; border-radius: 15px; text-align: center; color: white; font-size: 34px; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 30px; }
    .section { background-color: white; padding: 25px; border-radius: 15px; margin-top: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); color: #2c3e50; }
    .result { background-color: #e8f5e9; padding: 15px; border-radius: 12px; font-size: 19px; color: #1b5e20; margin-top: 20px; font-weight: bold; }
    .fun-fact { background-color: #d1ecf1; color: #0c5460; padding: 12px 20px; border-radius: 50px; text-align: center; font-size: 16px; width: fit-content; margin: 15px auto 10px auto; animation: float 3s ease-in-out infinite; }
    .stTextInput > div > input, .stTextArea > div > textarea { font-size: 17px; border-radius: 8px; border: 1px solid #ddd; color: black !important; background-color: white; }
    .file-upload-text { font-size: 18px; font-weight: 600; margin-top: 20px; margin-bottom: 8px; color: #333; }
    @keyframes float { 0% { transform: translateY(0); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0); } }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">🧠 LENS eXpert (NLP Suites)</div>', unsafe_allow_html=True)

# Sidebar with Image
with st.sidebar:
    st.image("robot_img.png", width=180)
    tool = st.radio("📂 Choose a Tool", ["🌍 Language Detection", "⭐ Restaurant Reviews", "📨 Spam Classifier", "📰 News Classifier"])

    st.write("")  
    st.write("")  
    st.write("")  

    with st.expander("ℹ️ More Info"):
        st.subheader("About Us")
        st.write("""
            This project uses Machine Learning, Natural Language Processing (NLP), 
            Scikit-learn, NLTK, and many other powerful libraries to provide smart text analysis tools.
        """)
        
        st.subheader("Contact")
        st.write("📧 chaubey5439@gmail.com")
        st.info("Feel free to reach out for collaborations, queries, or feedback!")

# Reset bulk file/results if tool changed
previous_tool = st.session_state.get("previous_tool", None)
if tool != previous_tool:
    st.session_state["bulk_file"] = None
    st.session_state["bulk_df"] = None
    st.session_state["previous_tool"] = tool

# Floating Fun Fact
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
    st.session_state.current_fact = random.choice(fun_facts)

if time.time() - st.session_state.last_update > 5:
    st.session_state.current_fact = random.choice(fun_facts)
    st.session_state.last_update = time.time()

st.markdown(f'<div class="fun-fact">💡 {st.session_state.current_fact}</div>', unsafe_allow_html=True)

# ------------------------- Tool Logic -------------------------

# Language Detection
if tool == "🌍 Language Detection":
    st.markdown('<div class="section">🌐 <strong>Language Detection</strong> – Guess the language of any sentence!</div>', unsafe_allow_html=True)
    text = st.text_input("✍️ Enter a sentence:", placeholder="e.g., Bonjour tout le monde")

    if st.button("🚀 Detect Language"):
        if text.strip() == "":
            st.warning("Please enter a sentence.")
        else:
            prediction = lang_model.predict([text])[0]
            emoji_flag = { "English": "🇬🇧", "French": "🇫🇷", "Hindi": "🇮🇳", "Spanish": "🇪🇸", "German": "🇩🇪" }.get(prediction, "🌍")
            st.markdown(f'<div class="result">🗣️ Predicted Language: {emoji_flag} {prediction}</div>', unsafe_allow_html=True)

# Restaurant Reviews
elif tool == "⭐ Restaurant Reviews":
    st.markdown('<div class="section">⭐ <strong>Review Sentiment Analyzer</strong> – Check if a review is positive or negative.</div>', unsafe_allow_html=True)
    text = st.text_area("📝 Enter review text:", placeholder="e.g., The food was amazing and service was great!")

    if st.button("🔍 Analyze Review"):
        if text.strip() == "":
            st.warning("Please enter a review.")
        else:
            prediction = review_model.predict([text])[0]
            if prediction == 1 or str(prediction).lower() == "positive":
                st.markdown(f'<div class="result">😄 Positive Review! Glad they loved it!</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="result">😞 Negative Review! We need to do better.</div>', unsafe_allow_html=True)

# Spam Classifier
elif tool == "📨 Spam Classifier":
    st.markdown('<div class="section">📬 <strong>Spam Detection</strong> – Detect if a message is spam or safe.</div>', unsafe_allow_html=True)
    text = st.text_area("📨 Enter message:", placeholder="e.g., You won $1,000,000! Click here!")

    if st.button("⚡ Detect Spam"):
        if text.strip() == "":
            st.warning("Please enter a message.")
        else:
            prediction = spam_model.predict([text])[0]
            if prediction == "spam":
                st.markdown(f'<div class="result">⚠️ This message is <span style="color:#c0392b">SPAM</span>!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result">✅ This message is <span style="color:#27ae60">NOT Spam</span>! 🎉</div>', unsafe_allow_html=True)
                st.balloons()

# News Classifier
elif tool == "📰 News Classifier":
    st.markdown('<div class="section">📰 <strong>News Classifier</strong></div>', unsafe_allow_html=True)
    st.info("🚧 Our News Classifier model is under construction. We are working hard to make it available soon!")
    st.markdown("""
    <div style="background-color:#fff3cd; color:#856404; padding:15px; border-radius:12px; margin-top:15px; font-size:17px;">
        ✅ We have already built the foundation for News Classification.<br>
        🛠️ Our model is currently being improved and will be launched shortly.<br>
        💡 Stay tuned for updates!
    </div>
    """, unsafe_allow_html=True)

# ---------------- BULK FILE UPLOAD AT BOTTOM ------------------

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="file-upload-text">📁 Want to analyze a bulk file? Upload your `.txt` or `.csv` file below:</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose your file", type=["txt", "csv"], key=f"uploader_{tool}")

def process_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, delimiter=None)
            if df.shape[1] > 1:
                st.warning("⚠️ Your CSV has multiple columns. Only the first column will be analyzed.")
            df = df.iloc[:, [0]].rename(columns={df.columns[0]: "Text"})
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        try:
            content = uploaded_file.read().decode('utf-8').splitlines()
            df = pd.DataFrame({"Text": content})
            return df
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return None

if uploaded_file:
    df = process_file(uploaded_file)
    st.session_state["bulk_file"] = uploaded_file
    st.session_state["bulk_df"] = df

if st.session_state.get("bulk_df") is not None:
    df = st.session_state["bulk_df"]
    
    if tool == "🌍 Language Detection":
        df["Prediction"] = lang_model.predict(df["Text"])
    
    elif tool == "⭐ Restaurant Reviews":
        df["Prediction"] = review_model.predict(df["Text"])
        df["Prediction"] = df["Prediction"].map({0: "Disliked", 1: "Liked"})
    
    elif tool == "📨 Spam Classifier":
        df["Prediction"] = spam_model.predict(df["Text"])
        df["Prediction"] = df["Prediction"].replace({"ham": "Not Spam"})
        df["Prediction"] = df["Prediction"].replace({"spam": "Spam"})
    
    else:
        st.warning("Bulk upload not available for News Classifier.")
        df = None

    if df is not None:
        st.success("Bulk Prediction Complete!")
        with st.expander("🔍 See Bulk Predictions"):
            st.dataframe(df)

# ---------------- Footer ----------------
st.markdown('<div class="footer">🚀 Built with Streamlit, Machine Learning, and Positive Energy!</div>', unsafe_allow_html=True)
