import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

DATA_PATH = "dataset/apple_support.csv"

# Load dataset (aman walau CSV kotor)
data = pd.read_csv(
    DATA_PATH,
    engine="python",
    on_bad_lines="skip"
)

data.columns = [c.strip().lower() for c in data.columns]

QUESTION_COL = "question"
ANSWER_COL = "answer"
if QUESTION_COL not in data.columns or ANSWER_COL not in data.columns:
    raise ValueError(f"Kolom CSV harus ada: {data.columns}")

# rapikan nama kolom
data.columns = [c.strip().lower() for c in data.columns]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

data["clean_question"] = data[QUESTION_COL].astype(str).apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

def chatbot_response(user_input):
    # translate input ke English
    translated_input = GoogleTranslator(source="auto", target="en").translate(user_input)
    clean_input = clean_text(translated_input)

    vec_input = vectorizer.transform([clean_input])
    similarities = cosine_similarity(vec_input, X)
    best_idx = similarities.argmax()

    # threshold biar gak random
    if similarities[0][best_idx] < 0.2:
        return GoogleTranslator(source="en", target="id").translate(
            "Sorry, this question is not available in my dataset."
        )

    answer_en = data.iloc[best_idx][ANSWER_COL]

    # kembalikan ke bahasa user
    return GoogleTranslator(source="en", target="auto").translate(answer_en)


