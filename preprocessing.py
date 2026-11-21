
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
for resource in [
    'stopwords',
    'wordnet',
    'omw-1.4',
    'punkt',
    'punkt_tab'
]:
    nltk.download(resource, quiet=True)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Preserve negation words
negation_words = {'no', 'nor', 'not', 'never'}
stop_words = stop_words - negation_words

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return ' '.join(tokens)
