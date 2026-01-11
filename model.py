import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset 
DATA_PATH = "apple_support.csv"

data = pd.read_csv(DATA_PATH, sep=",")

# Validasi kolom
if not {"question", "answer"}.issubset(data.columns):
    raise ValueError(f"Kolom CSV tidak sesuai: {data.columns}")

# Preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

data["clean_question"] = data["question"].apply(clean_text)

# TF-IDF Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

# Chatbot Logic
def chatbot_response(user_input: str) -> str:
    user_input = user_input.strip()

    # Greeting (Indonesia & English)
    if user_input.lower() in ["halo", "hai", "hi", "hello", "selamat pagi", "selamat siang", "selamat sore", "selamat malam"]:
        return (
            "Halo! Saya siap membantu Anda terkait perangkat Apple."
            if user_input.lower() in ["halo", "hai"]
            else "Hello! I can help you with Apple device support."
        )

    user_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_clean])
    similarity = cosine_similarity(user_vec, X)

    best_idx = similarity.argmax()
    best_score = similarity[0][best_idx]

    # Threshold (strict mode default)
    if best_score < 0.2:
        return "Maaf, saya tidak mengerti maksud anda."

    return data.iloc[best_idx]["answer"]



