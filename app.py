from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ===== LOAD MODELS =====
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

news_data = pd.concat([fake, true])
news_data = news_data.dropna(subset=["text"])

x_train_n, _, y_train_n, _ = train_test_split(
    news_data["text"], news_data["label"], test_size=0.2, random_state=42
)

vectorizer_news = TfidfVectorizer(stop_words="english")
x_train_n_vec = vectorizer_news.fit_transform(x_train_n)

model_news = LogisticRegression(max_iter=1000)
model_news.fit(x_train_n_vec, y_train_n)

hate = pd.read_csv("hate_speech.csv")
hate = hate[["tweet", "class"]]
hate["class"] = hate["class"].apply(lambda x: 0 if x in [0, 1] else 1)

x_train_h, _, y_train_h, _ = train_test_split(
    hate["tweet"], hate["class"], test_size=0.2, random_state=42
)

vectorizer_hate = TfidfVectorizer(stop_words="english")
x_train_h_vec = vectorizer_hate.fit_transform(x_train_h)

model_hate = LogisticRegression(max_iter=1000)
model_hate.fit(x_train_h_vec, y_train_h)

# ===== HTML =====
html = """
<h2>AI Text Detector 🔥</h2>
<form method="post">
<textarea name="text" rows="6" cols="50"></textarea><br><br>
<button name="action" value="fake">Check Fake News</button>
<button name="action" value="hate">Check Hate Speech</button>
</form>

{% if result %}
<h3>{{ result }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        action = request.form["action"]

        if action == "fake":
            vec = vectorizer_news.transform([text])
            pred = model_news.predict(vec)
            result = "🟢 Real News" if pred[0] == 1 else "🔴 Fake News"

        else:
            vec = vectorizer_hate.transform([text])
            pred = model_hate.predict(vec)
            result = "🚨 Hate Speech" if pred[0] == 0 else "✅ Safe Comment"

    return render_template_string(html, result=result)

# IMPORTANT
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)