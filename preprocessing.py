import re
import spacy
from langdetect import detect, LangDetectException

# Load spaCy English model for tokenization, stop-word removal, and lemmatization
try:
    nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    nlp_en = spacy.blank("en")

# Preserve negation words in English stop-word removal
NEGATION_WORDS = {"no", "nor", "not", "never"}
for neg in NEGATION_WORDS:
    nlp_en.Defaults.stop_words.discard(neg)
    if neg in nlp_en.vocab:
        nlp_en.vocab[neg].is_stop = False

# Blank spaCy pipeline for Hindi / Devanagari tokenization
nlp_hi = spacy.blank("hi")

# Common Hindi stopwords (preserving negation words like नहीं, न, मत)
HINDI_STOPWORDS = {
    "के", "का", "की", "को", "से", "में", "पर", "है", "हैं", "था", "थी", "थे",
    "और", "एक", "यह", "वह", "कि", "ने", "हो", "गया", "गई", "गए", "कर",
    "रहा", "रही", "रहे", "इस", "उस", "भी", "तो", "ही", "जो", "तक", "लिए",
    "साथ", "बाद", "द्वारा", "अपनी", "अपने", "किया", "दिया", "लिया",
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}


def detect_language(text: str) -> tuple[str, str]:
    """Detect the primary language of the input text."""
    if not text or not text.strip():
        return "en", "English"

    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    alpha_chars = len(re.findall(r"[a-zA-Z]", text))
    if devanagari_chars > 5 and devanagari_chars >= alpha_chars * 0.3:
        return "hi", "Hindi"

    try:
        lang_code = detect(text)
    except LangDetectException:
        lang_code = "en"

    return lang_code, LANGUAGE_NAMES.get(lang_code, lang_code.upper())


def preprocess_text(text: str) -> str:
    """Clean, tokenize, lowercase, remove stopwords, and lemmatize English text using spaCy."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    doc = nlp_en(text)
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        word = token.text.lower()
        if word in NEGATION_WORDS:
            tokens.append(word)
            continue
        if token.is_stop:
            continue
        lemma = token.lemma_.lower().strip() if token.lemma_ else word
        if lemma == "-pron-":
            lemma = word
        if lemma:
            tokens.append(lemma)

    return " ".join(tokens)


def preprocess_multilingual(text: str, lang_code: str | None = None) -> dict:
    """Detect language and apply spaCy language-appropriate preprocessing."""
    if lang_code is None:
        detected_code, detected_name = detect_language(text)
    else:
        detected_code = lang_code
        detected_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())

    if detected_code == "en":
        cleaned = preprocess_text(text)
        return {
            "language": detected_code,
            "language_name": detected_name,
            "cleaned_text": cleaned,
            "model_input_text": cleaned,
            "token_count": len(cleaned.split()) if cleaned else 0,
        }

    # Multilingual / Hindi spaCy tokenization + stop-word removal
    cleaned_unicode = re.sub(r"https?://\S+|www\.\S+", "", text.lower())
    cleaned_unicode = re.sub(r"[^\w\s\u0900-\u097F]", " ", cleaned_unicode)
    cleaned_unicode = re.sub(r"\s+", " ", cleaned_unicode).strip()

    doc_multi = nlp_hi(cleaned_unicode)
    tokens = [
        tok.text
        for tok in doc_multi
        if not tok.is_space
        and tok.text not in HINDI_STOPWORDS
        and tok.text not in nlp_en.Defaults.stop_words
    ]
    cleaned_multi = " ".join(tokens)

    # Extract any English tokens for the English-trained TF-IDF model
    english_cleaned = preprocess_text(text)

    return {
        "language": detected_code,
        "language_name": detected_name,
        "cleaned_text": cleaned_multi,
        "model_input_text": english_cleaned,
        "token_count": len(tokens),
    }
