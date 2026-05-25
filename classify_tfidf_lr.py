import pandas as pd
import numpy as np
import os, joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

os.makedirs("models",    exist_ok=True)
os.makedirs("data/gold", exist_ok=True)

# Load Silver-layer Parquet splits 
train_df = pd.read_parquet("data/silver/train")
dev_df   = pd.read_parquet("data/silver/dev")
test_df  = pd.read_parquet("data/silver/test")

for df in [train_df, dev_df, test_df]:
    df.fillna({"processed_text": ""}, inplace=True)

X_train, y_train = train_df["processed_text"], train_df["label"]
X_dev,   y_dev   = dev_df["processed_text"],   dev_df["label"]
X_test,  y_test  = test_df["processed_text"],  test_df["label"]

print(f"Train: {len(X_train)} | Dev: {len(X_dev)} | Test: {len(X_test)}")

#Load PCM Sentiment Lexicon for Feature Augmentation 
# For each tweet, we compute two numeric features:
#   lexicon_pos_score — fraction of words found in the positive lexicon
#   lexicon_neg_score — fraction of words found in the negative lexicon
# These features complement TF-IDF by injecting curated linguistic knowledge.
try:
    lex_df = pd.read_csv("data/raw/lexicon_manual.tsv", sep="\t",
                         names=["word","sentiment"], header=None)
    POS_LEX = set(lex_df[lex_df["sentiment"]=="positive"]["word"].str.lower())
    NEG_LEX = set(lex_df[lex_df["sentiment"]=="negative"]["word"].str.lower())
    print(f"Lexicon: {len(POS_LEX)} positive | {len(NEG_LEX)} negative words")
    USE_LEXICON = True
except Exception:
    POS_LEX, NEG_LEX = set(), set()
    USE_LEXICON = False
    print("Lexicon not loaded. Proceeding with TF-IDF features only.")


def lex_features(texts):
    """Compute (pos_score, neg_score) pair for each text."""
    feats = []
    for t in texts:
        words = str(t).lower().split()
        n = max(len(words), 1)
        pos = sum(1 for w in words if w in POS_LEX) / n
        neg = sum(1 for w in words if w in NEG_LEX) / n
        feats.append([pos, neg])
    import scipy.sparse as sp
    return sp.csr_matrix(np.array(feats))


#TF-IDF Vectorisation 
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),     # Unigrams + bigrams capture Pidgin collocations
    sublinear_tf=True,
    min_df=2,
    analyzer="word"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf   = vectorizer.transform(X_dev)
X_test_tfidf  = vectorizer.transform(X_test)

if USE_LEXICON:
    X_train_feat = hstack([X_train_tfidf, lex_features(X_train)])
    X_dev_feat   = hstack([X_dev_tfidf,   lex_features(X_dev)])
    X_test_feat  = hstack([X_test_tfidf,  lex_features(X_test)])
else:
    X_train_feat, X_dev_feat, X_test_feat = X_train_tfidf, X_dev_tfidf, X_test_tfidf

# Train Logistic Regression 
print("Training Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",  # Compensate for class imbalance
    solver="lbfgs",
    multi_class="multinomial"
)
model.fit(X_train_feat, y_train)

#Dev Evaluation (tuning) 
y_dev_pred = model.predict(X_dev_feat)
dev_f1 = f1_score(y_dev, y_dev_pred, average="weighted")
print(f"Dev Weighted F1-Score : {dev_f1:.4f}")

#  Final Test Evaluation 
y_test_pred = model.predict(X_test_feat)

print("\n LR + TF-IDF Results on PCM Test Split ")
print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred,
      target_names=["negative","neutral","positive"]))
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print(confusion_matrix(y_test, y_test_pred,
      labels=["negative","neutral","positive"]))

# Save model and vectoriser joblib.dump({"model": model, "vectorizer": vectorizer}, "models/lr_tfidf_model.pkl")
print("✓ Model saved to models/lr_tfidf_model.pkl")

# Save predictions 
test_df["lr_pred"] = y_test_pred
test_df.to_csv("data/gold/lr_predictions.csv", index=False, encoding="utf-8-sig")
print("✓ Predictions saved to data/gold/lr_predictions.csv")
