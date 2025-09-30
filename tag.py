# version: 0.1.0
# path: tag.py
"""Helpers for managing tag metadata associated with demonstration artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence, Union


def normalize_tag(tag: str) -> str:
    """Return a canonical representation for ``tag``.

    Whitespace is stripped from both ends and the function ensures the result is
    not empty. ``ValueError`` is raised for invalid input so that callers can
    catch the problem explicitly instead of silently storing unusable tags.
    """

    if not isinstance(tag, str):
        raise TypeError("Tags must be strings")
    trimmed = tag.strip()
    if not trimmed:
        raise ValueError("Tags must contain at least one non-whitespace character")
    return trimmed


class TagStore:
    """Maintain a mapping of identifiers to a stable, de-duplicated tag list."""

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive
        self._data: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    def _normalize_item_id(self, item_id: object) -> str:
        if isinstance(item_id, str):
            normalized = item_id.strip()
        else:
            normalized = str(item_id)
        if not normalized:
            raise ValueError("Item identifiers cannot be empty")
        return normalized

    def _normalize_tag(self, tag: str) -> tuple[str, str]:
        canonical = normalize_tag(tag)
        if not self.case_sensitive:
            return canonical.casefold(), canonical
        return canonical, canonical

    # ------------------------------------------------------------------
    # CRUD operations
    def set(self, item_id: object, tags: Iterable[str]) -> None:
        normalized_id = self._normalize_item_id(item_id)
        container: Dict[str, str] = {}
        for tag in tags:
            key, canonical = self._normalize_tag(tag)
            container[key] = canonical
        if container:
            self._data[normalized_id] = dict(sorted(container.items(), key=lambda kv: kv[1].casefold()))
        else:
            self._data.pop(normalized_id, None)

    def add(self, item_id: object, *tags: str) -> bool:
        normalized_id = self._normalize_item_id(item_id)
        if normalized_id not in self._data:
            self._data[normalized_id] = {}
        container = self._data[normalized_id]
        changed = False
        for tag in tags:
            key, canonical = self._normalize_tag(tag)
            if key not in container:
                container[key] = canonical
                changed = True
        if changed:
            self._data[normalized_id] = dict(sorted(container.items(), key=lambda kv: kv[1].casefold()))
        return changed

    def remove(self, item_id: object, *tags: str) -> bool:
        normalized_id = self._normalize_item_id(item_id)
        if normalized_id not in self._data:
            return False
        if not tags:
            del self._data[normalized_id]
            return True

        container = self._data[normalized_id]
        changed = False
        for tag in tags:
            key, _ = self._normalize_tag(tag)
            if key in container:
                del container[key]
                changed = True

        if not container:
            del self._data[normalized_id]
            return True

        if changed:
            self._data[normalized_id] = dict(sorted(container.items(), key=lambda kv: kv[1].casefold()))
        return changed

    # ------------------------------------------------------------------
    # Queries
    def get(self, item_id: object) -> list[str]:
        normalized_id = self._normalize_item_id(item_id)
        tags = self._data.get(normalized_id, {})
        return list(tags.values())

    def items(self) -> Iterator[tuple[str, list[str]]]:
        for item_id, tags in sorted(self._data.items(), key=lambda kv: kv[0]):
            yield item_id, list(tags.values())

    def find(self, *tags: str, match_all: bool = True) -> list[str]:
        if not tags:
            return [item_id for item_id, _ in self.items()]

        normalized_tags = [self._normalize_tag(tag)[0] for tag in tags]
        matches: list[str] = []
        for item_id, stored in self._data.items():
            keys = set(stored.keys())
            if match_all:
                if all(tag in keys for tag in normalized_tags):
                    matches.append(item_id)
            else:
                if any(tag in keys for tag in normalized_tags):
                    matches.append(item_id)
        return sorted(matches)

    # ------------------------------------------------------------------
    # Persistence helpers
    def to_dict(self) -> Dict[str, list[str]]:
        return {item_id: list(tags.values()) for item_id, tags in self._data.items()}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str]], *, case_sensitive: bool = False) -> "TagStore":
        store = cls(case_sensitive=case_sensitive)
        for item_id, tags in mapping.items():
            store.set(item_id, tags)
        return store

    def save(self, path: Union[str, Path], *, encoding: str = "utf-8") -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding=encoding) as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def load_tag_store(path: Union[str, Path], *, case_sensitive: bool = False) -> TagStore:
    file_path = Path(path)
    if not file_path.exists():
        return TagStore(case_sensitive=case_sensitive)

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise TypeError("Tag file must contain an object mapping identifiers to tags")
    parsed: Dict[str, Sequence[str]] = {}
    for key, value in data.items():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Tags for '{key}' must be stored as a sequence of strings")
        parsed[key] = list(value)
    return TagStore.from_mapping(parsed, case_sensitive=case_sensitive)


__all__ = ["TagStore", "normalize_tag", "load_tag_store"]

