"""Run Top-1/Top-3/Top-5 retrieval evaluation over the two sample domains.

Run from backend/: python scripts/evaluate_retrieval.py
Requires GEMINI_API_KEY because indexing and query retrieval use Gemini embeddings.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.evaluation.retrieval import evaluate_retrieval, find_relevant_chunk_ids
from app.ingestion import get_parser
from app.ingestion.chunker import chunk_document
from app.ingestion.normalizer import normalize
from app.agents.retrieval import RetrievalAgent
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.storage.documents import DocumentStore


def main() -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is required for embedding-based retrieval evaluation.", file=sys.stderr)
        return 2

    dataset_dir = ROOT / "evaluation" / "datasets"
    corpus_files = [dataset_dir / "hr_policy.txt", dataset_dir / "it_support.txt"]

    # Keep sample evaluation data separate from documents uploaded through the UI.
    evaluation_runtime = ROOT / "data" / "evaluation"
    shutil.rmtree(evaluation_runtime, ignore_errors=True)
    settings.storage_dir = str(evaluation_runtime / "storage")
    settings.index_dir = str(evaluation_runtime / "index")

    document_store = DocumentStore()
    vector_store = VectorStore(document_store=document_store)

    for path in corpus_files:
        document_id = f"milestone-sample-{path.stem}"
        if document_store.document_exists(document_id):
            continue
        parsed = get_parser(path.suffix.lstrip(".")).parse(path.read_bytes(), path.name, document_id)
        parsed.source_text = normalize(parsed.source_text)
        for section in parsed.sections:
            section["text"] = normalize(section["text"])
        chunks = chunk_document(parsed)
        document_store.save_document(parsed, chunks, path.stat().st_size, "milestone-evaluation")
        vector_store.add_chunks(chunks)

    cases_data = json.loads((dataset_dir / "queries.json").read_text(encoding="utf-8"))
    documents_by_filename = {doc["filename"]: doc["document_id"] for doc in document_store.list_documents()}
    cases = []
    for row in cases_data["queries"]:
        candidate_chunks = []
        evidence_phrases = row.get("relevant_evidence", [])
        for filename in row["relevant_files"]:
            document_id = documents_by_filename.get(filename)
            if document_id:
                candidate_chunks.extend(document_store.get_chunks_for_document(document_id))
        relevant_chunk_ids = find_relevant_chunk_ids(candidate_chunks, evidence_phrases)
        if row["relevant_files"] and not relevant_chunk_ids:
            raise RuntimeError(
                f"No labelled evidence chunk found for evaluation case {row.get('id')}. "
                "Check its relevant_evidence phrases against the sample document text."
            )
        cases.append({**row, "relevant_ids": relevant_chunk_ids})

    retrieval_agent = RetrievalAgent(retriever=Retriever(vector_store=vector_store))

    def retrieve_ranked(question: str, k: int) -> list[dict]:
        candidates = retrieval_agent.retrieve(question, k=max(settings.initial_retrieval_k, k))
        reranked = retrieval_agent.rerank(question, candidates)
        return [
            {
                "document_id": item.chunk.document_id,
                "chunk_id": item.chunk.chunk_id,
                "initial_score": round(item.score, 5),
                "rerank_score": round(item.rerank_score, 5),
            }
            for item in reranked[:k]
        ]

    result = evaluate_retrieval(cases, retrieve_ranked, low_score_threshold=settings.min_retrieval_score)
    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "retrieval_evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"query_count": result["query_count"], "metrics": result["metrics"], "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
