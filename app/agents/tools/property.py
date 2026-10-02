from __future__ import annotations

import re
from threading import Lock
from typing import Any


class PropertySearchService:

    def __init__(self) -> None:
        self._properties: list[dict[str, Any]] = []
        self._lock = Lock()

    def replace_all(self, properties: list[dict[str, Any]]) -> None:
        """Replace the development catalogue with defensive copies."""
        with self._lock:
            self._properties = [dict(item) for item in properties]

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._properties]

    def search(
        self,
        query: str,
        filters: dict,
    ) -> list[dict]:
        """Search the local development catalogue with text and hard filters."""
        terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) > 1}
        results: list[tuple[int, dict[str, Any]]] = []

        for item in self.list_all():
            nested_metadata = item.get("metadata")
            metadata = {
                **(nested_metadata if isinstance(nested_metadata, dict) else {}),
                **{key: value for key, value in item.items() if key != "metadata"},
            }
            if not self._matches(metadata, filters):
                continue
            haystack = " ".join(str(value) for value in item.values()).lower()
            score = sum(term in haystack for term in terms)
            if terms and score == 0:
                continue
            result = dict(item)
            result["score"] = float(score)
            results.append((score, result))

        results.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in results]

    @staticmethod
    def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if expected in (None, ""):
                continue
            if key.startswith("min_"):
                actual = metadata.get(key[4:])
                if actual is None or actual < expected:
                    return False
            elif key.startswith("max_"):
                actual = metadata.get(key[4:])
                if actual is None or actual > expected:
                    return False
            elif str(metadata.get(key, "")).lower() != str(expected).lower():
                return False
        return True

    @staticmethod
    def extract_requirements(query: str) -> dict[str, Any]:
        requirements: dict[str, Any] = {}
        bedrooms = re.search(r"(\d+)\s*(?:bed(?:room)?s?|غرف)", query, re.IGNORECASE)
        if bedrooms:
            requirements["bedrooms"] = int(bedrooms.group(1))
        return requirements

    @staticmethod
    def rewrite_query(query: str, requirements: dict[str, Any]) -> str:
        suffix = " ".join(f"{key}:{value}" for key, value in requirements.items())
        return f"{query.strip()} {suffix}".strip()

    @staticmethod
    def select(candidates: list[dict[str, Any]], requirements: dict[str, Any]) -> dict[str, Any] | None:
        del requirements
        return candidates[0] if candidates else None

    @staticmethod
    def format_response(selected_property: dict[str, Any]) -> str:
        property_id = selected_property.get("property_id") or selected_property.get("id") or "selected property"
        return f"The best matching option is {property_id}."


property_search_service = PropertySearchService()
