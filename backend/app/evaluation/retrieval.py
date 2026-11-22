"""Offline retrieval evaluation metrics (Hit@K / Top-K retrieval accuracy)."""
from collections.abc import Callable, Iterable
from typing import Any


def hit_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float:
    """Return 1 when any relevant item occurs in the first k results, else 0."""
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Each evaluation query must define at least one relevant id")
    return float(any(item in relevant for item in list(ranked_ids)[:k]))


def find_relevant_chunk_ids(chunks: Iterable[Any], evidence_phrases: Iterable[str]) -> list[str]:
    """Map every labelled evidence phrase to all chunks that contain it.

    Phrases may be supported by different chunks. Return no labels when even
    one phrase cannot be found, so an incomplete evaluation label cannot
    silently inflate retrieval accuracy.
    """
    phrases = list(dict.fromkeys(phrase.casefold().strip() for phrase in evidence_phrases if phrase.strip()))
    if not phrases:
        return []
    matched_by_phrase = {phrase: set() for phrase in phrases}
    for chunk in chunks:
        chunk_text = str(getattr(chunk, "text", "")).casefold()
        chunk_id = getattr(chunk, "chunk_id", None)
        if not chunk_id:
            continue
        for phrase in phrases:
            if phrase in chunk_text:
                matched_by_phrase[phrase].add(str(chunk_id))
    if any(not chunk_ids for chunk_ids in matched_by_phrase.values()):
        return []
    return sorted(set().union(*matched_by_phrase.values()))


def evaluate_retrieval(
    cases: list[dict[str, Any]],
    retrieve: Callable[[str, int], Iterable[Any]],
    ks: tuple[int, ...] = (1, 3, 5),
    low_score_threshold: float = 0.25,
) -> dict[str, Any]:
    """Evaluate a retrieval callback against labelled relevant chunk IDs."""
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    per_case: list[dict[str, Any]] = []
    totals = {k: 0.0 for k in ks}
    domain_totals: dict[str, dict[str, Any]] = {}
    for case in cases:
        question = case["question"]
        relevant = case["relevant_ids"]
        ranked_items = list(retrieve(question, max(ks)))
        ranked_ids = [
            str(item.get("chunk_id") or item.get("id") or item.get("document_id"))
            if isinstance(item, dict) else str(item)
            for item in ranked_items
        ]
        is_answerable = bool(relevant)
        domain = str(case.get("domain") or "Unspecified")
        domain_bucket = domain_totals.setdefault(
            domain,
            {"query_count": 0, "answerable_query_count": 0, "unanswerable_query_count": 0,
             "hits": {k: 0.0 for k in ks}},
        )
        domain_bucket["query_count"] += 1
        scores = ({f"top_{k}_accuracy": hit_at_k(ranked_ids, relevant, k) for k in ks}
                  if is_answerable else {})
        if is_answerable:
            domain_bucket["answerable_query_count"] += 1
            for k in ks:
                totals[k] += scores[f"top_{k}_accuracy"]
                domain_bucket["hits"][k] += scores[f"top_{k}_accuracy"]
        else:
            domain_bucket["unanswerable_query_count"] += 1
        per_case.append({
            "id": case.get("id"),
            "domain": case.get("domain"),
            "category": case.get("category"),
            "question": question,
            "relevant_ids": relevant,
            "ranked_ids": ranked_ids,
            "ranked_results": ranked_items,
            "low_relevance_results": [
                item for item in ranked_items
                if isinstance(item, dict) and float(item.get("rerank_score", item.get("score", 1.0))) < low_score_threshold
            ],
            "answerable": is_answerable,
            "expected_behavior": "retrieve_evidence" if is_answerable else "abstain_due_to_missing_evidence",
            **scores,
        })
    answerable_count = sum(bool(case["relevant_ids"]) for case in cases)
    metrics_by_domain = {}
    for domain, bucket in domain_totals.items():
        denominator = bucket["answerable_query_count"]
        metrics_by_domain[domain] = {
            "query_count": bucket["query_count"],
            "answerable_query_count": denominator,
            "unanswerable_query_count": bucket["unanswerable_query_count"],
            "metrics": {
                f"top_{k}_accuracy": round(bucket["hits"][k] / denominator, 4) if denominator else 0.0
                for k in ks
            },
        }
    return {
        "query_count": len(cases),
        "answerable_query_count": answerable_count,
        "unanswerable_query_count": len(cases) - answerable_count,
        "metrics": {f"top_{k}_accuracy": round(value / answerable_count, 4) if answerable_count else 0.0 for k, value in totals.items()},
        "metrics_by_domain": metrics_by_domain,
        "results": per_case,
    }
