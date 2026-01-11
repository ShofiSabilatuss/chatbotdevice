import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== PATH =====
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "apple_support.csv")

# ===== LOAD CSV (ANTI ERROR) =====
data = pd.read_csv(
    DATA_PATH,
    engine="python",
    on_bad_lines="skip"
)

# ===== NORMALISASI NAMA KOLOM =====
data.columns = data.columns.str.strip().str.lower()

# GANTI sesuai CSV kamu
QUESTION_COL = "question"
ANSWER_COL = "answer"

if QUESTION_COL not in data.columns or ANSWER_COL not in data.columns:
    raise ValueError(
        f"CSV harus punya kolom '{QUESTION_COL}' dan '{ANSWER_COL}'. "
        f"Sekarang kolomnya: {list(data.columns)}"
    )

# ===== CLEAN TEXT =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

data["clean_question"] = data[QUESTION_COL].apply(clean_text)

# ===== TF-IDF =====
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

# ===== SAPAAN =====
GREETINGS = {
    "halo": "Halo! Saya siap membantu dukungan perangkat Apple 😊",
    "hai": "Hai! Ada yang bisa saya bantu tentang perangkat Apple?",
    "hello": "Hello! How can I help you with Apple devices?"
}

# ===== CHATBOT =====
def chatbot_response(user_input):
    user_input = user_input.strip()

    if user_input.lower() in GREETINGS:
        return GREETINGS[user_input.lower()]

    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    similarity = cosine_similarity(input_vec, X)

    best_idx = similarity.argmax()

    if similarity[0][best_idx] < 0.2:
        return "Maaf, pertanyaan tersebut belum tersedia dalam data saya."

    return str(data.iloc[best_idx][ANSWER_COL])
