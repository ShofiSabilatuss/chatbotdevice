import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# =====================
# LOAD DATASET
# =====================
DATA_PATH = "dataset/apple_support_clean.csv"
data = pd.read_csv(
    DATA_PATH,
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip"
)

QUESTION_COL = "question"
ANSWER_COL = "answer"

# =====================
# TEXT CLEANING
# =====================
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["clean_question"] = data[QUESTION_COL].apply(clean_text)

# =====================
# TF-IDF
# =====================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_question"])

# =====================
# LANGUAGE CHECK (TANPA DETECT)
# =====================
def is_indonesian(text):
    keywords = ["bagaimana", "cara", "apa", "kenapa", "mengapa", "bisa", "tidak"]
    return any(word in text.lower() for word in keywords)

# =====================
# CHATBOT RESPONSE
# =====================
def chatbot_response(user_input):
    user_lang = "id" if is_indonesian(user_input) else "en"

    # Greeting handling
    greetings_id = ["halo", "hai", "hi"]
    greetings_en = ["hello", "hi"]

    if user_input.lower().strip() in greetings_id:
        return "Halo! Saya siap membantu seputar dukungan perangkat Apple 😊"

    if user_input.lower().strip() in greetings_en:
        return "Hello! I can help you with Apple device support 😊"

    # Translate input to English if needed
    processed_input = user_input
    if user_lang == "id":
        processed_input = GoogleTranslator(source="id", target="en").translate(user_input)

    processed_input = clean_text(processed_input)

    user_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(user_vector, X)
    best_score = similarity.max()
    best_idx = similarity.argmax()

    THRESHOLD = 0.3

    if best_score < THRESHOLD:
        return (
            "Maaf, pertanyaan tersebut tidak tersedia dalam data yang saya miliki."
            if user_lang == "id"
            else "Sorry, this question is not available in my dataset."
        )

    answer_en = data.iloc[best_idx][ANSWER_COL]

    # Translate answer back to Indonesian if needed
    if user_lang == "id":
        return GoogleTranslator(source="en", target="id").translate(answer_en)

    return answer_en
