# section3/train_classifier.py
# CPU-Friendly Correct Version

# Install first if needed:
# python -m pip install datasets transformers evaluate scikit-learn pandas torch "accelerate>=0.26.0"

import os
import json
import random
import numpy as np
import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "tickets_dataset.csv"
OUTPUT_DIR = "saved_model"

SEED = 42
MAX_LEN = 128

BATCH_SIZE = 8          # CPU friendly
EPOCHS = 3             # enough for assignment
LR = 2e-5


def ensure_output_dir(path: str):
    if os.path.isdir(path):
        return

    if os.path.isfile(path):
        if os.path.getsize(path) == 0:
            os.remove(path)
        else:
            raise RuntimeError(
                f"'{path}' exists as a file, not a directory."
            )

    os.makedirs(path, exist_ok=True)


ensure_output_dir(OUTPUT_DIR)

# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass

set_seed(SEED)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

assert "text" in df.columns
assert "label" in df.columns

df = df.dropna().reset_index(drop=True)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)
print(df.shape)
print(df["label"].value_counts())

# =====================================================
# LABEL ENCODING
# =====================================================

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

label_names = list(le.classes_)
num_labels = len(label_names)

id2label = {i: v for i, v in enumerate(label_names)}
label2id = {v: i for i, v in enumerate(label_names)}

with open(
    os.path.join(OUTPUT_DIR, "label_mapping.json"),
    "w"
) as f:
    json.dump(id2label, f, indent=2)

print("\nLABELS")
for k, v in id2label.items():
    print(k, "->", v)

# =====================================================
# SPLIT
# =====================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label_id"]
)

print("\nTrain:", train_df.shape)
print("Test :", test_df.shape)

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,
        max_length=MAX_LEN
    )

train_ds = Dataset.from_pandas(
    train_df[["text", "label_id"]]
).rename_column("label_id", "labels")

test_ds = Dataset.from_pandas(
    test_df[["text", "label_id"]]
).rename_column("label_id", "labels")

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "labels"]

train_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

data_collator = DataCollatorWithPadding(tokenizer)

# =====================================================
# MODEL
# =====================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# =====================================================
# METRICS
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        labels,
        preds,
        average="weighted"
    )

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4)
    }

# =====================================================
# TRAINING ARGS
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LR,
    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    weight_decay=0.01,

    logging_steps=20,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    save_total_limit=1,

    fp16=False,          # IMPORTANT CPU FIX

    report_to="none",

    seed=SEED
)

# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =====================================================
# TRAIN
# =====================================================

print("\n" + "=" * 60)
print("START TRAINING")
print("=" * 60)

trainer.train()

# =====================================================
# FINAL EVALUATION
# =====================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

metrics = trainer.evaluate()

print(metrics)

# =====================================================
# SAVE MODEL
# =====================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved to:", OUTPUT_DIR)
