import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

MODEL_NAME = "castorini/afriberta_large"
LABEL2ID   = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}

os.makedirs("models/afriberta_pcm", exist_ok=True)
os.makedirs("data/gold",            exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_df = pd.read_parquet("data/silver/train").fillna({"processed_text":""})
dev_df   = pd.read_parquet("data/silver/dev").fillna({"processed_text":""})
test_df  = pd.read_parquet("data/silver/test").fillna({"processed_text":""})

# Use cleaned_text (not processed_text) for AfriBERTa so the model sees
# richer context including stopwords — transformers handle stopwords internally.
for df in [train_df, dev_df, test_df]:
    df.fillna({"cleaned_text":""}, inplace=True)
    df["label_id"] = df["label"].map(LABEL2ID)

print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
class PCMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(
            texts.tolist(), truncation=True,
            padding="max_length", max_length=max_len
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

print(f"Loading tokeniser and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
).to(device)

train_dataset = PCMDataset(train_df["cleaned_text"], train_df["label_id"].values, tokenizer)
dev_dataset   = PCMDataset(dev_df["cleaned_text"],   dev_df["label_id"].values,   tokenizer)
test_dataset  = PCMDataset(test_df["cleaned_text"],  test_df["label_id"].values,  tokenizer)

# PCM training set (~10,000 samples) is smaller than other NaijaSenti subsets.
# 5 epochs with early stopping (patience=2) prevents overfitting.
training_args = TrainingArguments(
    output_dir            = "./results/afriberta_pcm",
    num_train_epochs      = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    warmup_ratio          = 0.1,
    weight_decay          = 0.01,
    learning_rate         = 2e-5,
    logging_dir           = "./logs",
    logging_steps         = 50,
    evaluation_strategy   = "epoch",
    save_strategy         = "epoch",
    load_best_model_at_end= True,
    metric_for_best_model = "accuracy",
    fp16                  = torch.cuda.is_available(),
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = dev_dataset,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
)

#  Fine-Tuning 
print("Starting fine-tuning on NaijaSenti PCM train split...")
trainer.train()

# Test Evaluation 
print("\nEvaluating on held-out test split...")
preds_out = trainer.predict(test_dataset)
y_pred = np.argmax(preds_out.predictions, axis=-1)
y_true = test_df["label_id"].values

print(f"\nAfriBERTa PCM Test Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=["negative","neutral","positive"]))
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print(confusion_matrix(y_true, y_pred))

# Save fine-tuned model 
trainer.save_model("models/afriberta_pcm")
tokenizer.save_pretrained("models/afriberta_pcm")
print("\n✓ Model saved to models/afriberta_pcm/")

# Save predictions 

test_df["afriberta_pred"]    = [ID2LABEL[p] for p in y_pred]
test_df["afriberta_pred_id"] = y_pred
test_df.to_csv("data/gold/afriberta_predictions.csv", index=False, encoding="utf-8-sig")
print("✓ Predictions saved to data/gold/afriberta_predictions.csv")
