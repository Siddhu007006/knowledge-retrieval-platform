
"""Evaluate FAISS retrieval without the reranking stage."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.evaluation.retrieval import (
    evaluate_retrieval,
    find_relevant_chunk_ids,
)
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.storage.documents import DocumentStore


def main() -> int:
    evaluation_root = ROOT / "data" / "evaluation"
    storage_dir = evaluation_root / "storage"
    index_dir = evaluation_root / "index"

    required = [
        index_dir / "index.faiss",
        index_dir / "meta.json",
        storage_dir / "documents.json",
        storage_dir / "chunks.json",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing evaluation files:", missing, file=sys.stderr)
        return 1

    # Load the existing evaluation corpus and FAISS index.
    # Do not delete or rebuild either one.
    settings.storage_dir = str(storage_dir)
    settings.index_dir = str(index_dir)

    document_store = DocumentStore()
    vector_store = VectorStore(document_store=document_store)
    retriever = Retriever(vector_store=vector_store)

    dataset_dir = ROOT / "evaluation" / "datasets"
    cases_data = json.loads(
        (dataset_dir / "queries.json").read_text(encoding="utf-8")
    )

    documents_by_filename = {
        doc["filename"]: doc["document_id"]
        for doc in document_store.list_documents()
    }

    cases = []

    for row in cases_data["queries"]:
        candidate_chunks = []

        for filename in row["relevant_files"]:
            document_id = documents_by_filename.get(filename)
            if document_id:
                candidate_chunks.extend(
                    document_store.get_chunks_for_document(document_id)
                )

        relevant_ids = find_relevant_chunk_ids(
            candidate_chunks,
            row.get("relevant_evidence", []),
        )

        if row["relevant_files"] and not relevant_ids:
            raise RuntimeError(
                f"Could not label evidence for query {row.get('id')}"
            )

        cases.append({**row, "relevant_ids": relevant_ids})

    def retrieve_faiss_only(question: str, k: int) -> list[dict]:
        candidates = retriever.retrieve(
            question,
            k=max(settings.initial_retrieval_k, k),
        )

        return [
            {
                "document_id": candidate.chunk.document_id,
                "chunk_id": candidate.chunk.chunk_id,
                "initial_score": round(candidate.score, 5),
                "score": round(candidate.score, 5),
            }
            for candidate in candidates[:k]
        ]

    result = evaluate_retrieval(
        cases,
        retrieve_faiss_only,
        low_score_threshold=settings.min_retrieval_score,
    )

    result["retrieval_mode"] = "faiss_only"

    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "faiss_baseline.json"
    output_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "retrieval_mode": result["retrieval_mode"],
        "query_count": result["query_count"],
        "metrics": result["metrics"],
        "metrics_by_domain": result["metrics_by_domain"],
        "report": str(output_file),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
