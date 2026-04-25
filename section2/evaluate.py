# section2/evaluate.py

import os
import json
import time
import statistics
from typing import Dict, List, Any

from rag_pipeline import ProductionLegalRAG


"""
FAANG-Level Evaluation Harness
------------------------------

Measures:

1. HitRate@3
   Whether any correct chunk appears in top 3.

2. MRR
   Mean Reciprocal Rank of first relevant result.

3. Avg Query Latency

4. Confidence Calibration
   Avg confidence on hits vs misses

5. Detailed JSON report

This evaluates retrieval quality separately
from generation quality.
"""


class RAGEvaluator:

    def __init__(
        self,
        docs_path="sample_docs",
        qa_path="qa_pairs.json",
        top_k=3
    ):

        self.docs_path = docs_path
        self.qa_path = qa_path
        self.top_k = top_k

        self.pipeline = ProductionLegalRAG(top_k=top_k)

        self.qa_pairs = self._load_qa()

    # ==================================================
    # LOAD QA
    # ==================================================

    def _load_qa(self):

        with open(self.qa_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================================================
    # BUILD INDEX
    # ==================================================

    def build(self):

        print("=" * 60)
        print("INDEXING LEGAL DOCUMENTS...")
        print("=" * 60)

        start = time.time()

        self.pipeline.ingest_documents(self.docs_path)

        end = time.time()

        print(f"Indexed Chunks : {len(self.pipeline.docs)}")
        print(f"Build Time     : {end-start:.2f}s")
        print("=" * 60)
        print()

    # ==================================================
    # NORMALIZATION
    # ==================================================

    def _norm(self, x):

        return str(x).strip().lower()

    def _clause_match(self, a, b):

        a = self._norm(a)
        b = self._norm(b)

        return a == b or a.startswith(b) or b.startswith(a)

    # ==================================================
    # MATCH LOGIC
    # ==================================================

    def _is_relevant(
        self,
        retrieved_item,
        expected
    ):

        # document required
        if self._norm(retrieved_item["document"]) != \
           self._norm(expected["document"]):
            return False

        # optional page
        if "page" in expected:
            if retrieved_item["page"] != expected["page"]:
                return False

        # optional clause
        if "clause" in expected:
            if not self._clause_match(
                retrieved_item["clause"],
                expected["clause"]
            ):
                return False

        return True

    # ==================================================
    # METRICS
    # ==================================================

    def _reciprocal_rank(
        self,
        retrieved,
        expected
    ):

        for rank, item in enumerate(retrieved, start=1):

            if self._is_relevant(item, expected):
                return 1.0 / rank

        return 0.0

    def _hit_at_k(
        self,
        retrieved,
        expected
    ):

        for item in retrieved:
            if self._is_relevant(item, expected):
                return 1

        return 0

    # ==================================================
    # EVALUATION
    # ==================================================

    def evaluate(self):

        total = len(self.qa_pairs)

        hits = 0
        rr_scores = []

        latencies = []

        conf_hits = []
        conf_misses = []

        rows = []

        for i, qa in enumerate(self.qa_pairs, start=1):

            question = qa["question"]
            expected = qa["expected_source"]

            # ------------------------------
            # RETRIEVAL ONLY
            # ------------------------------
            t0 = time.time()

            retrieved = self.pipeline.retrieve(question)

            t1 = time.time()

            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)

            confidence = self.pipeline._confidence(retrieved)

            hit = self._hit_at_k(retrieved, expected)
            rr = self._reciprocal_rank(retrieved, expected)

            hits += hit
            rr_scores.append(rr)

            if hit:
                conf_hits.append(confidence)
            else:
                conf_misses.append(confidence)

            top_doc = retrieved[0]["document"] if retrieved else "None"

            rows.append({
                "id": i,
                "question": question,
                "hit@3": bool(hit),
                "reciprocal_rank": round(rr, 4),
                "confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 2),
                "top_document": top_doc
            })

        # ==================================================
        # FINAL METRICS
        # ==================================================

        hitrate = hits / total if total else 0.0
        mrr = sum(rr_scores) / total if total else 0.0

        avg_latency = statistics.mean(latencies) \
            if latencies else 0.0

        p95_latency = sorted(latencies)[
            int(0.95 * len(latencies)) - 1
        ] if latencies else 0.0

        avg_conf_hit = statistics.mean(conf_hits) \
            if conf_hits else 0.0

        avg_conf_miss = statistics.mean(conf_misses) \
            if conf_misses else 0.0

        # ==================================================
        # PRINT REPORT
        # ==================================================

        print("=" * 70)
        print("LEGAL RAG RETRIEVAL EVALUATION REPORT")
        print("=" * 70)

        print(f"Questions                : {total}")
        print(f"HitRate@3               : {hitrate:.2%}")
        print(f"MRR                     : {mrr:.4f}")
        print(f"Avg Latency             : {avg_latency:.2f} ms")
        print(f"P95 Latency             : {p95_latency:.2f} ms")
        print(f"Avg Confidence (Hits)   : {avg_conf_hit:.3f}")
        print(f"Avg Confidence (Misses) : {avg_conf_miss:.3f}")

        print("=" * 70)
        print()

        print("Detailed Results")
        print("-" * 70)

        for row in rows:

            status = "PASS" if row["hit@3"] else "FAIL"

            print(
                f"[{row['id']:02d}] {status} | "
                f"RR={row['reciprocal_rank']:.2f} | "
                f"Conf={row['confidence']:.2f} | "
                f"{row['latency_ms']:.1f}ms | "
                f"{row['question']}"
            )

        # ==================================================
        # SAVE REPORT
        # ==================================================

        metrics = {
            "questions": total,
            "hitrate_at_3": round(hitrate, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "avg_confidence_hit": round(avg_conf_hit, 4),
            "avg_confidence_miss": round(avg_conf_miss, 4),
            "details": rows
        }

        out_file = "evaluation_results.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print()
        print(f"Saved: {out_file}")

        return metrics


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    evaluator = RAGEvaluator(
        docs_path="sample_docs",
        qa_path="qa_pairs.json",
        top_k=3
    )

    evaluator.build()

    evaluator.evaluate()
