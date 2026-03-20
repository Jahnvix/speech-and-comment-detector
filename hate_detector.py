import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset (make sure file is extracted properly)
data = pd.read_csv("hate_speech.csv")

# Check columns (optional but helpful)
print("Columns:", data.columns)

# Use correct columns (for this dataset)
data = data[["tweet", "class"]]

# Convert labels (0 = hate, 1 = safe)
data["class"] = data["class"].apply(lambda x: 0 if x in [0, 1] else 1)
# Clean data
data = data.dropna(subset=["tweet"])
data["tweet"] = data["tweet"].astype(str)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data["tweet"], data["class"], test_size=0.2, random_state=42
)

# Convert text → numbers
vectorizer = TfidfVectorizer(stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

# Test loop
while True:
    text = input("\nEnter comment (type 'exit' to quit): ")

    if text.lower() == "exit":
        print("Exiting...")
        break

    vec = vectorizer.transform([text])
    pred = model.predict(vec)

    if pred[0] == 0:
        print("🚨 Hate Speech")
    else:
        print("✅ Safe Comment")