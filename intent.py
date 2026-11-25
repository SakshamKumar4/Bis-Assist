"""Step 2: train the intent classifier (TF-IDF + Logistic Regression)."""
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("intent_data.csv")
X_tr, X_te, y_tr, y_te = train_test_split(
    df.text, df.label, test_size=0.2, stratify=df.label, random_state=42)

clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
                    LogisticRegression(max_iter=1000))
clf.fit(X_tr, y_tr)

pred = clf.predict(X_te)
print(classification_report(y_te, pred))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))

clf.fit(df.text, df.label)          # retrain on all data for the app
joblib.dump(clf, "data/intent.pkl")
print("Saved data/intent.pkl")
