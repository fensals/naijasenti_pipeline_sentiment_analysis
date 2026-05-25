import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

os.makedirs("data/gold", exist_ok=True)

# Load preprocessed test split from Silver layer
test_df = pd.read_parquet("data/silver/test")
test_df = test_df.fillna({"processed_text": ""})
print(f"Test records loaded: {len(test_df)}")

analyzer = SentimentIntensityAnalyzer()

def classify_vader(text):
    """
    VADER compound score thresholds:
       >= 0.05  → positive
       <= -0.05 → negative
       else     → neutral
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral"
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


test_df["vader_pred"] = test_df["processed_text"].apply(classify_vader)

y_true = test_df["label"]
y_pred = test_df["vader_pred"]

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=["negative","neutral","positive"]))
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print(confusion_matrix(y_true, y_pred,
      labels=["negative","neutral","positive"]))

test_df.to_csv("data/gold/vader_predictions.csv", index=False, encoding="utf-8-sig")
print("\n✓ Saved to data/gold/vader_predictions.csv")
