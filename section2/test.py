from rag_pipeline import ProductionLegalRAG

pipeline = ProductionLegalRAG()
pipeline.ingest_documents("sample_docs")

result = pipeline.query(
    question="What is the notice period in the NDA with Vendor X?"
)

print(result)