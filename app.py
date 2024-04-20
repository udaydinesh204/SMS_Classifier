import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError  # Import NotFittedError
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

# Initialize stemmer
ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]

    return " ".join(filtered_words)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Set up Streamlit interface
st.title("Email/SMS Spam Classifier")

# Input message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    try:
        vector_input = tfidf.transform([transformed_sms])
    except ValueError as e:
        st.error(f"Error vectorizing input: {e}")
        st.stop()
    
    # Predict
    try:
        result = model.predict(vector_input)[0]
        # Display results
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except NotFittedError as e:
        st.error(f"Model is not fitted: {e}")