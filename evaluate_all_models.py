import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import os
os.makedirs("outputs", exist_ok=True)

LABELS = ["negative", "neutral", "positive"]

MODELS = {
    "VADER"       : ("data/gold/vader_predictions.csv",     "vader_pred"),
    "LR + TF-IDF" : ("data/gold/lr_predictions.csv",        "lr_pred"),
    "AfriBERTa"   : ("data/gold/afriberta_predictions.csv", "afriberta_pred"),
}

summary = {}

for model_name, (csv_path, pred_col) in MODELS.items():
    df = pd.read_csv(csv_path).dropna(subset=[pred_col, "label"])
    y_true = df["label"]
    y_pred = df[pred_col]
    summary[model_name] = {
        "Accuracy"  : accuracy_score(y_true, y_pred),
        "Precision" : precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall"    : recall_score(y_true, y_pred,    average="weighted", zero_division=0),
        "F1-Score"  : f1_score(y_true, y_pred,        average="weighted", zero_division=0),
        "CM"        : confusion_matrix(y_true, y_pred, labels=LABELS)
    }

#  Print summary 
print("\n Model Performance Summary — NaijaSenti PCM Test Split)
for name, m in summary.items():
    print(f"\n  {name}")
    for k in ["Accuracy","Precision","Recall","F1-Score"]:
        print(f"    {k:10} : {m[k]:.4f}")

# Bar chart comparison 
metrics = ["Accuracy","Precision","Recall","F1-Score"]
model_names = list(summary.keys())
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metrics)); w = 0.25
colours = ["#3498DB","#27AE60","#E74C3C"]

for i, (mname, m) in enumerate(summary.items()):
    vals = [m[k] for k in metrics]
    bars = ax.bar(x + i*w, vals, w, label=mname, color=colours[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=8)

ax.set_title("Model Performance Comparison — NaijaSenti PCM Test Split",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + w); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()
print("\n✓ Comparison chart saved to outputs/model_comparison.png")

# Confusion matrices (side by side) 
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Confusion Matrices — NaijaSenti PCM Test Split", fontsize=13, fontweight="bold")

for ax, (mname, m) in zip(axes, summary.items()):
    cm_norm = m["CM"].astype(float) / m["CM"].sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False)
    ax.set_title(mname); ax.set_xlabel("Predicted"); ax.set_ylabel("True")

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)
plt.close()
print("✓ Confusion matrices saved to outputs/confusion_matrices.png")
