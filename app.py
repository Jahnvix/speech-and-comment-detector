import pandas as pd
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= MODELS =================
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

news_data = pd.concat([fake, true])
news_data = news_data.dropna(subset=["text"])
news_data["text"] = news_data["text"].astype(str)

x_train_n, _, y_train_n, _ = train_test_split(
    news_data["text"], news_data["label"], test_size=0.2, random_state=42
)

vectorizer_news = TfidfVectorizer(stop_words="english")
x_train_n_vec = vectorizer_news.fit_transform(x_train_n)

model_news = LogisticRegression(max_iter=1000)
model_news.fit(x_train_n_vec, y_train_n)

# Hate model
hate = pd.read_csv("hate_speech.csv")
hate = hate[["tweet", "class"]]
hate["class"] = hate["class"].apply(lambda x: 0 if x in [0, 1] else 1)

hate = hate.dropna(subset=["tweet"])
hate["tweet"] = hate["tweet"].astype(str)

x_train_h, _, y_train_h, _ = train_test_split(
    hate["tweet"], hate["class"], test_size=0.2, random_state=42
)

vectorizer_hate = TfidfVectorizer(stop_words="english")
x_train_h_vec = vectorizer_hate.fit_transform(x_train_h)

model_hate = LogisticRegression(max_iter=1000)
model_hate.fit(x_train_h_vec, y_train_h)

# ================= FUNCTIONS =================
def check_fake():
    text = input_text.get("1.0", END)
    vec = vectorizer_news.transform([text])
    pred = model_news.predict(vec)
    prob = model_news.predict_proba(vec)[0]

    confidence = max(prob) * 100

    if pred[0] == 1:
        result_label.config(text=f"🟢 Real News ({confidence:.2f}%)", fg="gold")
    else:
        result_label.config(text=f"🔴 Fake News ({confidence:.2f}%)", fg="red")


def check_hate():
    text = input_text.get("1.0", END)
    vec = vectorizer_hate.transform([text])
    pred = model_hate.predict(vec)
    prob = model_hate.predict_proba(vec)[0]

    confidence = max(prob) * 100

    if pred[0] == 0:
        result_label.config(text=f"🚨 Hate Speech ({confidence:.2f}%)", fg="red")
    else:
        result_label.config(text=f"✅ Safe Comment ({confidence:.2f}%)", fg="gold")

# ================= UI =================
root = Tk()
root.title("AI Detector 🔥")
root.geometry("500x400")
root.configure(bg="black")

title = Label(root, text="AI Text Detector", font=("Arial", 18, "bold"), bg="black", fg="gold")
title.pack(pady=10)

input_text = Text(root, height=8, width=50, bg="black", fg="gold", insertbackground="gold")
input_text.pack(pady=10)

btn1 = Button(root, text="Check Fake News", command=check_fake, bg="gold", fg="black")
btn1.pack(pady=5)

btn2 = Button(root, text="Check Hate Speech", command=check_hate, bg="gold", fg="black")
btn2.pack(pady=5)

result_label = Label(root, text="", font=("Arial", 14), bg="black")
result_label.pack(pady=20)

root.mainloop()