"""Unit tests for the text normalizer."""

from app.ingestion.normalizer import normalize


class TestNormalizer:
    def test_collapse_whitespace(self):
        assert normalize("hello    world") == "hello world"

    def test_collapse_newlines(self):
        result = normalize("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_remove_control_chars(self):
        result = normalize("hello\x00\x01world")
        assert result == "helloworld"

    def test_normalize_line_endings(self):
        result = normalize("a\r\nb\r\nc")
        assert "\r" not in result

    def test_empty_input(self):
        assert normalize("") == ""

    def test_strips_outer_whitespace(self):
        assert normalize("  hello  ") == "hello"

    def test_preserves_paragraphs(self):
        result = normalize("para one\n\npara two")
        assert "para one" in result
        assert "para two" in result
        assert "\n\n" in result
