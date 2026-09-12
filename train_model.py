import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def train():
    print("Loading dataset...")
    # এক্সেল ফাইল পড়তে openpyxl প্যাকেজ লাগবে (pip install openpyxl)
    file_path = "Merged_Final_Dataset1 (2).xlsx"
    df = pd.read_excel(file_path)

    # আপনার ডাটা সেটের কলাম অনুযায়ী নেওয়া হচ্ছে
    df.dropna(subset=['clean_review', 'Sentiments'], inplace=True)

    X = df['clean_review']
    y = df['Sentiments']

    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Machine Learning Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.5))
    ])

    print("Training model on dataset...")
    pipeline.fit(X_train, y_train)

    # ইভালুয়েশন রিপোর্ট
    y_pred = pipeline.predict(X_test)
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, y_pred))

    # সেভ করা
    joblib.dump(pipeline, 'sentiment_model.pkl')
    print("\nModel saved successfully as 'sentiment_model.pkl'")

if __name__ == "__main__":
    train()