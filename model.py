import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "dataset/apple_support.csv"
QUESTION_COL = "question"
ANSWER_COL = "answer"

# === Load CSV dengan aman ===
data = pd.read_csv(
    DATA_PATH,
    encoding="utf-8",
    on_bad_lines="skip"
)

# Validasi kolom
required_cols = {QUESTION_COL, ANSWER_COL}
if not required_cols.issubset(data.columns):
    raise ValueError(f"Kolom CSV harus ada: {data.columns}")

# === Text cleaning ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()

data["clean_question"] = data[QUESTION_COL].apply(clean_text)

# === Vectorizer ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

# === Greeting handler ===
def greeting_response(text):
    greetings = {
        "halo": "Halo! Ada yang bisa saya bantu terkait perangkat Apple?",
        "hai": "Hai! Silakan tanyakan masalah perangkat Apple kamu.",
        "hello": "Hello! How can I help you with your Apple device?"
    }
    return greetings.get(text.lower())

# === Chatbot function ===
def chatbot_response(user_input):
    user_input = user_input.strip()

    greet = greeting_response(user_input)
    if greet:
        return greet

    cleaned = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned])
    similarity = cosine_similarity(user_vec, X)
    best_idx = similarity.argmax()

    if similarity[0][best_idx] < 0.2:
        return "Maaf, pertanyaan tersebut belum tersedia di data saya."

    return data.iloc[best_idx][ANSWER_COL]





