# Section 2: Production-Grade RAG Pipeline Design

## Objective

Build a high-precision question-answering system over a repository of legal contracts and policy PDFs.

Example queries:

- What is the notice period in the NDA signed with Vendor X?
- Which contracts contain a limitation of liability clause above ₹1 crore?

System requirements:

- Exact source citation required
- Page-level traceability required
- Hallucinated answers unacceptable
- Must scale beyond initial corpus size

---

# 1. System Design Principles

Legal document retrieval differs from generic RAG because:

- Small wording changes change legal meaning
- Definitions may apply globally across sections
- Clauses reference other clauses
- Tables may contain financial thresholds
- Users require auditability, not approximate summaries
- False answers are higher risk than refusals

Therefore the pipeline is optimized for:

- High precision retrieval
- Deterministic citations
- Structured metadata filtering
- Refusal when evidence is weak
- Scale-ready architecture

---

# 2. Document Ingestion Strategy

## Input Characteristics

Corpus contains PDF contracts averaging ~40 pages.

Expected real-world PDF issues:

- Native text PDFs
- Scanned contracts
- OCR noise
- Two-column layouts
- Tables
- Repeated headers / footers
- Signature pages
- Annexures / appendices

## Ingestion Pipeline

```text
PDF Upload Queue
→ OCR / Text Extraction
→ Layout Parsing
→ Section Detection
→ Clause Parsing
→ Chunk Builder
→ Embedding Workers
→ Indexing