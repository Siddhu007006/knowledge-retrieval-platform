"""Clarification Agent for underspecified queries without a usable antecedent."""
import re


class ClarificationAgent:
    """Detect a narrow set of unresolved references before semantic retrieval."""

    _GENERIC = re.compile(r"^(why|how|what about (it|that|this)|explain (it|that|this)|tell me more)[?.! ]*$", re.I)
    _UNRESOLVED_REFERENCE = re.compile(
        r"\b(it|they|them)\b|\b(that|this|those|these)\s*(?:[?.!,]|$)|\b(that|this)\s+(one|thing)\b",
        re.I,
    )

    def analyze(self, question: str, history: list[dict[str, str]]) -> tuple[bool, str | None]:
        normalized = " ".join(question.split())
        if history:
            return False, None
        words = re.findall(r"[\w']+", normalized)
        is_generic = bool(self._GENERIC.fullmatch(normalized))
        has_unresolved_reference = bool(self._UNRESOLVED_REFERENCE.search(normalized)) and len(words) <= 8
        if is_generic or has_unresolved_reference:
            return True, "What does the reference in your question refer to? Please name the document, topic, or item you want me to retrieve information about."
        return False, None
