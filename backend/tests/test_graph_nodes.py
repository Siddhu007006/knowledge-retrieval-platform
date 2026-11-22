"""Unit tests for graph nodes (validate_query, rewrite_query, etc.)."""

import pytest

from app.core.exceptions import EmptyQueryError, QueryTooLongError
from app.graph.nodes import validate_query


class TestValidateQuery:
    def test_valid_query(self):
        state = {"original_question": "What is AI?", "request_id": "r1"}
        result = validate_query(state)
        assert "events" in result
        assert len(result["events"]) == 1

    def test_empty_query(self):
        state = {"original_question": "", "request_id": "r1"}
        with pytest.raises(EmptyQueryError):
            validate_query(state)

    def test_whitespace_query(self):
        state = {"original_question": "   ", "request_id": "r1"}
        with pytest.raises(EmptyQueryError):
            validate_query(state)

    def test_too_long_query(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.max_query_length", 10)
        state = {"original_question": "This is a very long query that exceeds the limit", "request_id": "r1"}
        with pytest.raises(QueryTooLongError):
            validate_query(state)
