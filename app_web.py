from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ================= LOAD PHRASES =================
def load_phrases(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

safe_phrases = load_phrases("safe_phrases.txt")

# ================= CLEAN TEXT =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# ================= FAKE NEWS MODEL =================
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

news_data = pd.concat([fake, true])
news_data["content"] = news_data["title"] + " " + news_data["text"]

news_data = news_data.dropna(subset=["content"])
news_data["content"] = news_data["content"].astype(str).str.lower()

vectorizer_news = TfidfVectorizer(stop_words="english", max_df=0.7)
X_news = vectorizer_news.fit_transform(news_data["content"])
y_news = news_data["label"]

model_news = MultinomialNB()
model_news.fit(X_news, y_news)

# ================= HATE MODEL =================
hate = pd.read_csv("hate_speech.csv")
hate = hate[["tweet", "class"]]

hate["class"] = hate["class"].apply(lambda x: 0 if x in [0, 1] else 1)

hate = hate.dropna(subset=["tweet"])
hate["tweet"] = hate["tweet"].astype(str).str.lower()

vectorizer_hate = TfidfVectorizer(stop_words="english")
X_hate = vectorizer_hate.fit_transform(hate["tweet"])
y_hate = hate["class"]

model_hate = MultinomialNB()
model_hate.fit(X_hate, y_hate)

# ================= HATE PHRASES =================
hate_phrases = [
    "you are stupid", "you are dumb", "you are useless",
    "i hate you", "go to hell", "kill yourself",
    "you are trash", "idiot", "moron"
]

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""

    if request.method == "POST":
        raw_text = request.form["text"]
        text = clean_text(raw_text)
        action = request.form.get("action")

        # ===== FAKE NEWS =====
        if action == "fake":
            vec = vectorizer_news.transform([text])
            pred = model_news.predict(vec)
            prob = model_news.predict_proba(vec)[0]

            conf = max(prob) * 100

            if conf < 55:
                result = "⚠️ Uncertain (needs more context)"
            elif pred[0] == 1:
                result = "🟢 Likely Real News"
            else:
                result = "🔴 Likely Fake News"

            confidence = f"Confidence: {conf:.2f}%"

        # ===== HATE DETECTION =====
        elif action == "hate":

            # 🔥 SAFE CHECK (SMART MATCH)
            text_no_space = text.replace(" ", "")

            if any(
                phrase in text or phrase.replace(" ", "") in text_no_space
                for phrase in safe_phrases
            ):
                return render_template(
                    "index.html",
                    result="✅ Safe Comment",
                    confidence="Confidence: High"
                )

            # 🔥 STRONG HATE
            if any(phrase in text for phrase in hate_phrases):
                return render_template(
                    "index.html",
                    result="🚨 Hate Speech",
                    confidence="Confidence: High"
                )

            # 🔥 ML MODEL
            vec = vectorizer_hate.transform([text])
            pred = model_hate.predict(vec)
            prob = model_hate.predict_proba(vec)[0]

            conf = max(prob) * 100

            if pred[0] == 0 and conf > 70:
                result = "🚨 Hate Speech"
            else:
                result = "✅ Safe Comment"

            confidence = f"Confidence: {conf:.2f}%"

    return render_template("index.html", result=result, confidence=confidence)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)