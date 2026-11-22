"""Unit tests for the grounding checker."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.models.workflow import ContextPackage, GroundingResult, RerankedChunk, ChunkModel
from app.generation.grounding import check_grounding, _parse_grounding_response


def _make_context():
    return ContextPackage(
        question="What is X?",
        rewritten_query="What is X?",
        selected_chunks=[
            RerankedChunk(
                chunk=ChunkModel(
                    chunk_id="c1",
                    document_id="d1",
                    filename="test.txt",
                    file_type="txt",
                    text="X is a test concept.",
                ),
                score=0.8,
                rerank_score=0.8,
            )
        ],
    )


class TestGroundingParser:
    def test_parse_supported(self):
        text = json.dumps({
            "status": "supported",
            "supported_claims": ["X is a test concept"],
            "unsupported_claims": [],
            "reasons": ["All claims found in evidence."],
        })
        result = _parse_grounding_response(text)
        assert result.status == "supported"
        assert len(result.supported_claims) == 1

    def test_parse_unsupported(self):
        text = json.dumps({
            "status": "unsupported",
            "supported_claims": [],
            "unsupported_claims": ["Y is not in evidence"],
            "reasons": ["No evidence for Y."],
        })
        result = _parse_grounding_response(text)
        assert result.status == "unsupported"

    def test_parse_with_code_fence(self):
        text = '```json\n{"status": "supported", "supported_claims": [], "unsupported_claims": [], "reasons": []}\n```'
        result = _parse_grounding_response(text)
        assert result.status == "supported"

    def test_parse_invalid_json(self):
        from app.core.exceptions import MalformedModelResponseError

        with pytest.raises(MalformedModelResponseError):
            _parse_grounding_response("not json at all")

    def test_parse_invalid_status_defaults(self):
        text = json.dumps({"status": "bogus", "supported_claims": [], "unsupported_claims": [], "reasons": []})
        result = _parse_grounding_response(text)
        assert result.status == "unsupported"


class TestCheckGrounding:
    def test_no_context_returns_unsupported(self):
        context = ContextPackage(
            question="q", rewritten_query="q", selected_chunks=[]
        )
        result = check_grounding(context, "answer")
        assert result.status == "unsupported"
