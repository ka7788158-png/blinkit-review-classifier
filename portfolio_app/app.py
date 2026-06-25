import streamlit as st
import pickle
import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set page config for a wider, cleaner layout
st.set_page_config(page_title="Blinkit Review Analyzer", page_icon="🛒", layout="centered")

# --- LOAD MODELS & NLP ---
@st.cache_resource
def load_models():

# Get the directory where app.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Build the absolute paths to the model files
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    svm_path = os.path.join(BASE_DIR, 'svm_classifier.pkl')
    
    with open(tfidf_path, 'rb') as f: 
        vectorizer = pickle.load(f)
    with open(svm_path, 'rb') as f:    
        classifier = pickle.load(f)
        
    nlp_model = spacy.load("en_core_web_sm")
    vader_analyzer = SentimentIntensityAnalyzer()
    return vectorizer, classifier, nlp_model, vader_analyzer

tfidf, svm_model, nlp, sia = load_models()

# --- PREPROCESSING FUNCTIONS (From Notebook) ---
url_pattern = re.compile(r"https?://\S+|www\.\S+")
emoji_pattern = re.compile(
    "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE
)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = url_pattern.sub("", text)
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

def get_sentiment(text):
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05: return 'Positive 🟢'
    elif compound <= -0.05: return 'Negative 🔴'
    else: return 'Neutral ⚪'

def get_urgency(text, rating):
    t = text.lower()
    score = 0
    
    tier3 = ["consumer forum", "consumer court", "legal action", "fir", "police", "lawyer", "fraud", "cheating", "scam", "poisoned", "food poisoning", "hospitalized", "sick", "injured"]
    tier2 = ["refund", "money deducted", "charged twice", "amount deducted", "not delivered", "never arrived", "order missing", "wrong address", "expired", "rotten", "cockroach", "insect", "hair", "stone"]
    tier1 = ["late", "delayed", "waiting", "no response", "support not helping", "pathetic", "worst", "terrible", "disgusting", "horrible"]
    
    if any(kw in t for kw in tier3): score = 3
    elif any(kw in t for kw in tier2): score = 2
    elif any(kw in t for kw in tier1): score = 1
    
    label_map = {0: 'Low', 1: 'Medium', 2: 'High ⚠️', 3: 'Critical 🚨'}
    return label_map[score]

# --- UI DESIGN ---
st.title("🛒 Blinkit / Swiggy Complaint Analyzer")
st.write("This tool analyzes customer reviews to predict the **issue category**, assess **sentiment**, and flag **urgency** for support teams.")

# User Input
review_text = st.text_area("Enter a customer review:", placeholder="e.g., Money got deducted but order was never placed. Where is my refund?", height=150)
rating = st.slider("Customer Rating (Stars)", min_value=1, max_value=5, value=3)

if st.button("Analyze Review", type="primary"):
    if review_text.strip():
        with st.spinner("Analyzing text..."):
            # 1. Clean Text
            cleaned_text = clean_text(review_text)
            
            # 2. Predict Category using your PKL files
            vectorized_text = tfidf.transform([cleaned_text])
            predicted_category = svm_model.predict(vectorized_text)[0]
            
            # Make the category name look nicer (e.g., 'payment_issue' -> 'Payment Issue')
            display_category = predicted_category.replace('_', ' ').title()
            
            # 3. Sentiment & Urgency
            sentiment = get_sentiment(review_text)
            urgency = get_urgency(review_text, rating)
            
            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("Analysis Results")
            
            # Use columns to display metrics cleanly
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Issue Category", display_category)
            with col2:
                st.metric("Sentiment", sentiment)
            with col3:
                st.metric("Urgency Level", urgency)
                
            st.success("Analysis complete! Ready for the next review.")
    else:
        st.warning("Please enter a review to analyze.")
