import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from joblib import load
import nltk
nltk.download('stopwords')
nltk.download('punkt')

best_model = load("artifact/best_model.joblib")
vectorizer = load("artifact/vectorizer.joblib")

def text_processing(text):
    new_text = text.lower()  # Lowercase
    token = word_tokenize(new_text)  # Tokenize all words
    filter_words = [word for word in token if
                    word.isalnum() and word not in string.punctuation and word not in stopwords.words("english")]
    stemmer = PorterStemmer()
    word_stemming = [stemmer.stem(word) for word in filter_words]
    return " ".join(word_stemming)