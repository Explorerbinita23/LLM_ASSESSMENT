# LLM Assessment Submission

This repository contains solutions for four sections of an Applied AI / LLM engineering assessment:

1. **LLM Pipeline Diagnosis**  
2. **Production-Grade RAG Pipeline**  
3. **Fine-Tuned Ticket Classifier**  
4. **Written Systems Design Review**

The project is organized so evaluators can run the key deliverables **locally in under 5 minutes** using the included pretrained artifacts.

---

# Quick Start (Recommended Evaluation Path)

## Prerequisites

- Python **3.10+**
- Windows (tested environment)
- Internet connection only for first-time Hugging Face model download

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt

2. Run Section 3 – Ticket Classifier (Fastest Demo)

Uses the included pretrained saved_model/.

cd section3
python evaluate_classifier.py
python latency_test.py

This will:

Load the pretrained classifier
Evaluate accuracy / F1 / confusion matrix
Run latency SLA validation (<500ms per ticket)
3. Run Section 2 – RAG Pipeline
cd section2
python evaluate.py

This will:

Build vector index from sample PDFs
Run retrieval evaluation
Output metrics and results
Expected Runtime
Task	Approx Time
Install dependencies	2–4 min
Section 3 evaluation	1–2 min
Section 3 latency test	<30 sec
Section 2 evaluation	1–2 min
Repository Structure
.
├── section1/
│   └── diagnose_pipeline.md
│
├── section2/
│   ├── rag_pipeline.py
│   ├── evaluate.py
│   ├── qa_pairs.json
│   ├── evaluation_results.json
│   └── sample_docs/
│
├── section3/
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── latency_test.py
│   ├── tickets_dataset.csv
│   ├── evaluation_report.txt
│   └── saved_model/
│
├── section4/
│   └── ANSWER.md
│
├── ANSWERS.md
├── DESIGN.md
├── README.md
└── requirements.txt
Section Details
Section 1 – Pipeline Diagnosis

File:

section1/diagnose_pipeline.md

Contains debugging analysis of a broken LLM pipeline.

Section 2 – Production-Grade RAG Pipeline

Implements:

PDF ingestion
Chunking
Embeddings
Vector retrieval
Source citations
Evaluation harness

Run:

cd section2
python evaluate.py
Section 3 – Fine-Tuned Classifier

Five-class support ticket classifier:

billing
technical_issue
feature_request
complaint
other

Uses a pretrained DistilBERT model.

Evaluate
cd section3
python evaluate_classifier.py
Latency Test
python latency_test.py

Checks:

Valid labels
20 raw ticket predictions
Average inference under 500ms
Section 4 – Systems Design Answers

File:

section4/ANSWER.md

Contains written systems design responses.

Optional: Re-Train Classifier From Scratch

If you want to reproduce training:

cd section3
python train_classifier.py

Note: Training takes longer than the recommended evaluation path.

Recommended Reviewer Flow
pip install -r requirements.txt

cd section3
python evaluate_classifier.py
python latency_test.py

cd ../section2
python evaluate.py

This demonstrates the primary deliverables quickly.

No API Keys Required

Everything runs locally.

No OpenAI key required
No Pinecone key required
No external paid APIs required
Troubleshooting
First Run Downloads Model Files

The first run may download tokenizer/model files from Hugging Face.

Subsequent runs will be faster.

Windows CPU Torch Install

If PyTorch issues occur:

pip install torch --index-url https://download.pytorch.org/whl/cpu
If saved_model/ Is Missing

Run:

cd section3
python train_classifier.py
Submission Highlights
Production-style RAG pipeline
Retrieval evaluation metrics
Fine-tuned CPU classifier
Latency-tested inference
Practical systems design reasoning
Author

Submitted as part of an LLM Engineering Assessment.
