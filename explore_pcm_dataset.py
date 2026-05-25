import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

SPLITS = {
    "Train" : "data/raw/pcm_train.tsv",
    "Dev"   : "data/raw/pcm_dev.tsv",
    "Test"  : "data/raw/pcm_test.tsv",
}

frames = {}
for split_name, path in SPLITS.items():
    df = pd.read_csv(path, sep="\t", names=["tweet", "label"],
                     header=None, encoding="utf-8")
    df = df.dropna(subset=["tweet", "label"])
    df["split"] = split_name
    frames[split_name] = df

combined = pd.concat(frames.values(), ignore_index=True)

#Basic statistics 
print("=" * 60)
print("NaijaSenti PCM Dataset — Summary")
print("=" * 60)

for split_name, df in frames.items():
    print(f"\n── {split_name} Split ──")
    print(f"  Total records : {len(df)}")
    dist = df["label"].value_counts()
    for label, cnt in dist.items():
        print(f"  {label:>10} : {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

# Tweet length analysis 
combined["word_count"]  = combined["tweet"].str.split().str.len()
combined["char_count"]  = combined["tweet"].str.len()

print("\n── Tweet Length Statistics (Training Split) ──")
train_df = frames["Train"]
train_df = train_df.copy()
train_df["word_count"] = train_df["tweet"].str.split().str.len()
print(train_df["word_count"].describe().to_string())

# Null/duplicate audit 
print("\n── Data Quality Audit ──")
for split_name, df in frames.items():
    nulls = df.isnull().sum().sum()
    dupes = df.duplicated(subset=["tweet"]).sum()
    print(f"  {split_name:6} — Nulls: {nulls} | Duplicates: {dupes}")

#  Class distribution bar chart 
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("NaijaSenti PCM — Sentiment Class Distribution by Split",
             fontsize=14, fontweight="bold")

colours = {"positive": "#27AE60", "negative": "#E74C3C", "neutral": "#3498DB"}

for ax, (split_name, df) in zip(axes, frames.items()):
    counts = df["label"].value_counts().reindex(["positive","negative","neutral"])
    bars = ax.bar(counts.index, counts.values,
                  color=[colours.get(l, "grey") for l in counts.index])
    ax.set_title(f"{split_name} (n={len(df)})")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.close()
print("\n✓ Class distribution chart saved to outputs/class_distribution.png")

# Word count distribution (train) 
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.hist(train_df["word_count"], bins=30, color="#2E75B6", edgecolor="white")
ax2.set_title("Tweet Word Count Distribution — PCM Training Split")
ax2.set_xlabel("Word Count")
ax2.set_ylabel("Frequency")
ax2.axvline(train_df["word_count"].median(), color="red",
            linestyle="--", label=f"Median = {train_df["word_count"].median():.0f}")
ax2.legend()
plt.tight_layout()
plt.savefig("outputs/word_count_distribution.png", dpi=150)
plt.close()

print("✓ Word count distribution chart saved to outputs/word_count_distribution.png")
