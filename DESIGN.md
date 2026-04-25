# DESIGN.md

## Purpose

This root-level document records the architectural and implementation decisions for **Section 2: Build a Production-Grade RAG Pipeline**.

The system is a legal document question-answering solution over contracts, NDAs, and policy PDFs. It is designed to prioritize:

- factual accuracy
- exact citations
- low hallucination risk
- scalable retrieval architecture
- reproducible local execution

---

# 1. Problem Context

The repository contains 500+ legal PDF documents (contracts, NDAs, internal policies), averaging ~40 pages each.

Users ask precise questions such as:

- What is the notice period in the NDA signed with Vendor X?
- Which contracts contain a limitation of liability clause above ₹1 crore?
- How long must records be retained under the finance policy?

The solution must:

- retrieve relevant evidence accurately
- cite exact source document + page
- refuse unsupported answers
- run locally for this assessment

---

# 2. System Architecture

```text
PDF Documents
→ Text Extraction
→ Cleaning / Parsing
→ Clause-Aware Chunking
→ Embeddings
→ Vector Index (FAISS)
→ BM25 Lexical Index
→ Hybrid Retrieval
→ Cross-Encoder Re-ranking
→ Confidence Scoring
→ Grounded Response with Citations
3. Chunking Strategy
Decision: Clause-Aware Chunking

Legal documents are structured around numbered clauses such as:

7.2 Notice Period
8.4 Limitation of Liability
9.1 Governing Law

Using fixed token windows would split important clauses across chunks and weaken traceability.

Instead, chunks are aligned to legal structure.

Hierarchy
Document
→ Page
→ Section / Clause
→ Chunk
Metadata Stored Per Chunk
{
  "document": "nda_vendor_x.pdf",
  "page": 2,
  "section": "Notice Period",
  "clause": "7.2"
}
Why This Choice
better legal retrieval precision
easier page citation
preserves semantic meaning
supports audits and review
4. Embedding Model Choice
Selected Model
BAAI/bge-base-en-v1.5
Why

Chosen for strong semantic retrieval quality while remaining practical for local execution.

It offers a better balance than ultra-small models while avoiding the cost/latency of hosted APIs.

Alternatives Considered
bge-large
OpenAI embeddings
Cohere embeddings
MiniLM sentence transformers

For this assignment, bge-base gave the best simplicity/performance trade-off.

5. Vector Store Choice
Selected: FAISS
Why FAISS
local execution
no external infrastructure
fast nearest-neighbor search
lightweight setup
suitable for assessment environment
Index Strategy

Small datasets:

IndexFlatIP

Larger datasets:

IVF / HNSW / PQ

This enables easy future scaling.

6. Retrieval Strategy
Final Design: Hybrid Retrieval
Dense Retrieval
+
BM25 Keyword Retrieval
+
Cross-Encoder Re-ranking
Why Not Only Dense Search?

Dense retrieval may miss:

clause numbers
vendor names
currency values
exact legal terms
Why Not Only BM25?

BM25 may fail on paraphrased user queries.

Why Hybrid Wins

It combines:

semantic understanding
exact lexical precision

This is especially valuable in legal search.

7. Re-ranking Layer
Selected Model
BAAI/bge-reranker-base

Used to reorder top candidates after hybrid retrieval.

This improves final precision for ambiguous questions such as:

Which contract contains liability above ₹1 crore?
8. Hallucination Mitigation

Legal workflows require high trust.

Three safeguards were implemented.

1. Retrieval Grounding

Answers are generated only from retrieved evidence chunks.

2. Confidence Threshold

If evidence is weak, the system returns:

Insufficient evidence found in retrieved documents.
3. Mandatory Citations

Every response includes source references:

{
  "document": "...",
  "page": 2,
  "clause": "7.2"
}
9. Output Contract

The pipeline supports:

result = pipeline.query(
    question="What is the notice period in the NDA with Vendor X?"
)

Returns:

{
  "answer": "...",
  "sources": [...],
  "confidence": 0.84
}
10. Evaluation Methodology

I created 10 manually written QA pairs across sample legal documents.

Coverage includes:

notice periods
liability clauses
payment terms
renewals
retention periods
Primary Metric
Precision@3

Measures whether the correct chunk appears in top 3 retrieval results.

Additional Metrics
Hit Rate
MRR
Query latency
Confidence behavior
11. Scaling to 50,000 Documents

At larger scale, bottlenecks become:

OCR / parsing throughput
embedding generation time
vector index memory
keyword search latency
reranker latency
Scaled Architecture
Upload Queue
→ OCR Workers
→ Parsing Workers
→ GPU Embedding Service
→ Metadata Store (Postgres)
→ Elasticsearch BM25
→ Pinecone / Qdrant Vector DB
→ Query Router
→ Re-ranker Service
→ LLM Answer Layer
→ Redis Cache
→ Audit Logs
Specific Remedies
Parsing Bottleneck

Parallel worker queues.

Embedding Bottleneck

Batch inference on GPU.

Search Bottleneck

Sharded ANN indexes.

Re-ranking Bottleneck

Re-rank only top 20 results.

Repeated Queries

Semantic cache via Redis.

12. Trade-offs Considered
Why not Pinecone for this assignment?

External dependency unnecessary for local execution.

Why not pure generation?

Legal answers require grounded evidence.

Why not giant embeddings?

Higher cost with limited incremental benefit at this scale.

13. Future Improvements
OCR support for scanned contracts
table extraction
multilingual contracts
clause linking to definitions
learning-to-rank from feedback
span-level citations (char_start, char_end)
14. Final Summary

This system was designed for precision-first legal retrieval.

Core priorities:

correctness over fluency
citations over guessing
measurable retrieval quality
scalable architecture
production realism

For legal AI systems, trust and traceability are more important than creative generation.