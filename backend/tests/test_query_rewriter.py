"""Unit tests for the query rewriter."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.generation.query_rewriter import rewrite, _parse_rewrite_response, RewriteResult


class TestRewriteParser:
    def test_parse_basic(self):
        text = json.dumps({"rewritten_query": "better query", "reason": "improved"})
        result = _parse_rewrite_response(text)
        assert result.rewritten_query == "better query"
        assert result.reason == "improved"

    def test_parse_with_code_fence(self):
        text = '```json\n{"rewritten_query": "q", "reason": "r"}\n```'
        result = _parse_rewrite_response(text)
        assert result.rewritten_query == "q"

    def test_parse_invalid(self):
        from app.core.exceptions import MalformedModelResponseError

        with pytest.raises(MalformedModelResponseError):
            _parse_rewrite_response("not json")


class TestRewrite:
    def test_rewrite_returns_result(self):
        result = rewrite("What is that?", document_ids=None)
        assert isinstance(result, RewriteResult)
        assert result.rewritten_query
        assert result.reason
