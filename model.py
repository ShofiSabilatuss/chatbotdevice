import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

DATA_PATH = "apple_support.csv"
QUESTION_COL = "question"
ANSWER_COL = "answer"

translator = Translator()

data = pd.read_csv(DATA_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

data["clean_question"] = data[QUESTION_COL].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

def chatbot_response(user_input):
    user_input = user_input.strip()

    # Sapaan
    if user_input.lower() in ["halo", "hai", "hi", "hello"]:
        return "Halo! 👋 Saya siap membantu Anda terkait perangkat Apple."

    detected = translator.detect(user_input).lang
    user_input_en = translator.translate(
        user_input, src=detected, dest="en"
    ).text

    user_vec = vectorizer.transform([clean_text(user_input_en)])
    similarity = cosine_similarity(user_vec, X)
    best_idx = similarity.argmax()

    if similarity[0][best_idx] < 0.2:
        response_en = "Sorry, this question is not available in my dataset."
    else:
        response_en = data.iloc[best_idx][ANSWER_COL]

    if detected != "en":
        return translator.translate(
            response_en, src="en", dest=detected
        ).text

    return response_en
