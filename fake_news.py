import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# SAFE LOADING (FINAL FIX)
fake = pd.read_csv("Fake.csv", encoding="latin1", engine="python", quoting=3, on_bad_lines="skip")
true = pd.read_csv("True.csv", encoding="latin1", engine="python", quoting=3, on_bad_lines="skip")

print(fake.columns)
print(true.columns)

# Labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true])
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)


# Use text column
data = data[["text", "label"]]

# Split
x_train, x_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

# Accuracy
y_pred = model.predict(x_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Test
while True:
    news = input("Enter news: ")
    vec = vectorizer.transform([news])
    pred = model.predict(vec)

    print("🟢 Real" if pred[0] == 1 else "🔴 Fake")