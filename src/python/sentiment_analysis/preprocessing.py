"""
Text Preprocessing for Sentiment Analysis

Purpose:
This module preprocesses raw textual data (e.g., news articles, tweets) for sentiment analysis.
It performs tasks such as tokenization, stopword removal, and normalization to prepare the data
for input into NLP models.

Role in FORESIGHT:
- Cleans and prepares textual data for sentiment analysis.
- Ensures compatibility with NLP models like FinBERT.
- Supports the broader sentiment analysis pipeline.

Key Features:
- Text cleaning (e.g., removing special characters, lowercasing).
- Tokenization and stopword removal.
- Normalization (e.g., stemming, lemmatization).
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_text(text: str) -> str:
    """
    Preprocess raw text for sentiment analysis.

    Args:
        text (str): Raw text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Example usage
if __name__ == "__main__":
    raw_text = "Apple Inc. reported record-breaking earnings this quarter! #AAPL $AAPL"
    preprocessed_text = preprocess_text(raw_text)
    print("Preprocessed text:", preprocessed_text)