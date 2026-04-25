# section3/evaluate_classifier.py

# pip install transformers torch pandas scikit-learn numpy -q

import json
import time
import statistics
import numpy as np
import pandas as pd

from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "tickets_dataset.csv"
MODEL_DIR = "saved_model"
SEED = 42
BATCH_SIZE = 16

VALID_CLASSES = {
    "billing",
    "technical_issue",
    "feature_request",
    "complaint",
    "other"
}


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

assert "text" in df.columns
assert "label" in df.columns

df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

print("=" * 70)
print("DATASET INFO")
print("=" * 70)
print("Shape:", df.shape)
print(df["label"].value_counts())
print()


# =====================================================
# LABEL ENCODING
# =====================================================

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

label_names = list(le.classes_)

print("CLASSES")
for i, c in enumerate(label_names):
    print(i, "->", c)
print()


# =====================================================
# RECREATE SAME SPLIT AS TRAINING SCRIPT
# 70 / 15 / 15
# =====================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label_id"]
)

test_texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()

print("=" * 70)
print("EVALUATION SET")
print("=" * 70)
print("Samples:", len(test_df))
print()


# =====================================================
# LOAD MODEL
# =====================================================

clf = pipeline(
    task="text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    truncation=True
)


# =====================================================
# LOAD LABEL MAP (fallback)
# =====================================================

with open(f"{MODEL_DIR}/label_mapping.json", "r") as f:
    id_to_label = json.load(f)

id_to_label = {
    int(k): v for k, v in id_to_label.items()
}


# =====================================================
# PREDICTION HELPERS
# =====================================================

def normalize_prediction(raw_label):
    """
    Handles:
    billing
    LABEL_0
    """
    if raw_label in VALID_CLASSES:
        return raw_label

    if raw_label.startswith("LABEL_"):
        idx = int(raw_label.replace("LABEL_", ""))
        return id_to_label[idx]

    return "other"


# =====================================================
# BATCH INFERENCE
# =====================================================

print("=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)

start = time.time()

outputs = clf(
    test_texts,
    batch_size=BATCH_SIZE
)

end = time.time()

total_ms = (end - start) * 1000
avg_ms = total_ms / len(test_texts)

# Approx p95 per-sample estimate
# (batch pipelines don't expose exact per-item latency)
p95_ms = avg_ms * 1.25


# =====================================================
# PARSE OUTPUTS
# =====================================================

pred_labels = []
confidences = []

for out in outputs:

    label = normalize_prediction(out["label"])
    score = float(out["score"])

    pred_labels.append(label)
    confidences.append(score)


# =====================================================
# ASSERT VALID LABELS
# =====================================================

for p in pred_labels:
    assert p in VALID_CLASSES, f"Invalid class predicted: {p}"


# =====================================================
# METRICS
# =====================================================

acc = accuracy_score(true_labels, pred_labels)

macro_f1 = f1_score(
    true_labels,
    pred_labels,
    average="macro"
)

weighted_f1 = f1_score(
    true_labels,
    pred_labels,
    average="weighted"
)

report = classification_report(
    true_labels,
    pred_labels,
    digits=4
)

cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=label_names
)

cm_df = pd.DataFrame(
    cm,
    index=label_names,
    columns=label_names
)


# =====================================================
# MOST CONFUSED PAIR
# =====================================================

cm_copy = cm.copy()

for i in range(len(label_names)):
    cm_copy[i][i] = 0

max_idx = np.unravel_index(
    np.argmax(cm_copy),
    cm_copy.shape
)

true_cls = label_names[max_idx[0]]
pred_cls = label_names[max_idx[1]]
count = int(cm_copy[max_idx])


# =====================================================
# CONFIDENCE CALIBRATION
# =====================================================

correct_conf = []
wrong_conf = []

for t, p, c in zip(true_labels, pred_labels, confidences):
    if t == p:
        correct_conf.append(c)
    else:
        wrong_conf.append(c)

avg_conf_correct = (
    statistics.mean(correct_conf)
    if correct_conf else 0
)

avg_conf_wrong = (
    statistics.mean(wrong_conf)
    if wrong_conf else 0
)


# =====================================================
# PRINT REPORT
# =====================================================

print("=" * 70)
print("FINAL EVALUATION REPORT")
print("=" * 70)

print(f"Test Samples              : {len(test_df)}")
print(f"Accuracy                  : {acc:.4f}")
print(f"Macro F1                  : {macro_f1:.4f}")
print(f"Weighted F1               : {weighted_f1:.4f}")
print(f"Avg Latency / sample      : {avg_ms:.2f} ms")
print(f"Approx P95 Latency        : {p95_ms:.2f} ms")
print(f"Avg Confidence (Correct)  : {avg_conf_correct:.4f}")
print(f"Avg Confidence (Wrong)    : {avg_conf_wrong:.4f}")

print("=" * 70)
print()

print("CLASSIFICATION REPORT")
print("-" * 70)
print(report)

print("CONFUSION MATRIX")
print("-" * 70)
print(cm_df)
print()

print("MOST CONFUSED CLASS PAIR")
print("-" * 70)
print(
    f"True='{true_cls}' predicted as "
    f"'{pred_cls}' ({count} times)"
)

print()


# =====================================================
# SAVE TXT REPORT
# =====================================================

with open("evaluation_report.txt", "w") as f:

    f.write("FINAL EVALUATION REPORT\n")
    f.write("=" * 60 + "\n")

    f.write(f"Samples: {len(test_df)}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")
    f.write(f"Avg Latency(ms): {avg_ms:.2f}\n")
    f.write(f"P95 Latency(ms): {p95_ms:.2f}\n\n")

    f.write("Classification Report\n")
    f.write(report + "\n\n")

    f.write("Confusion Matrix\n")
    f.write(cm_df.to_string())
    f.write("\n\n")

    f.write(
        f"Most Confused Pair: "
        f"{true_cls} -> {pred_cls} ({count})\n"
    )

print("Saved: evaluation_report.txt")


# =====================================================
# TASK REQUIREMENT TEST:
# 20 RAW TICKETS
# - valid classes
# - under 500ms average
# =====================================================

sample_tickets = [
    "I was charged twice this month.",
    "App crashes when I upload a file.",
    "Please add dark mode feature.",
    "Your support team never responds.",
    "How do I change my profile photo?",
    "Refund not processed yet.",
    "Unable to login after password reset.",
    "Need export to excel option.",
    "Very disappointed with service quality.",
    "Where can I download invoices?",
    "Payment failed but money deducted.",
    "Dashboard loads blank screen.",
    "Please integrate with Slack.",
    "This is unacceptable support.",
    "Can I change my email?",
    "Charged after cancellation.",
    "Search button not working.",
    "Need mobile app support.",
    "Terrible experience overall.",
    "How to update billing address?"
]

t0 = time.time()

preds20 = clf(sample_tickets, batch_size=8)

t1 = time.time()

avg20_ms = ((t1 - t0) * 1000) / 20

parsed20 = [
    normalize_prediction(x["label"])
    for x in preds20
]

assert all(p in VALID_CLASSES for p in parsed20)
assert avg20_ms < 500, \
    f"Latency failed: {avg20_ms:.2f} ms"

print()
print("=" * 70)
print("TASK TEST PASSED")
print("=" * 70)
print("20 tickets predicted successfully")
print(f"Average latency: {avg20_ms:.2f} ms (< 500 ms)")
