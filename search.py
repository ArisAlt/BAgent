# version: 0.1.0
# path: search.py
"""Utility helpers for searching structured and plain-text log files.

The functions in this module are intentionally lightweight so they can be used
inside unit tests and ad-hoc scripts alike. They focus on JSON Lines (``.jsonl``)
inputs because that is the format used for bot demonstrations, yet they fall
back to simple string processing when a line is not valid JSON. Callers can
provide plain strings, compiled regular expressions or custom callables as the
query predicate. Optional ``fields`` allow targeting specific keys inside JSON
objects (including ``dot.notation`` for nested dictionaries).

Example
-------

```python
from search import search_in_file

results = search_in_file("logs/demonstrations/log.jsonl", "warp")
for match in results:
    print(match.path, match.line_number, match.entry["action"])
```

The search helpers return :class:`SearchResult` dataclasses containing the
parsed entry, the original raw text and metadata such as the source path and
line number.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

JsonLike = Mapping[str, Any]
QueryType = Union[str, re.Pattern[str], Sequence[Union[str, re.Pattern[str]]], Callable[[Any], bool]]


@dataclass(frozen=True)
class SearchResult:
    """Container describing a single match from :func:`search_in_file`.

    Attributes
    ----------
    path:
        Filesystem path to the scanned file.
    line_number:
        One-based line number where the match occurred.
    entry:
        Parsed representation of the line. JSON objects are returned as dicts;
        otherwise the raw string is provided.
    raw_text:
        The original line text prior to parsing. Useful for displaying context
        or for working with non-JSON inputs where ``entry`` is identical to the
        raw string.
    """

    path: Path
    line_number: int
    entry: Any
    raw_text: str


def iter_records(path: Union[str, Path], *, encoding: str = "utf-8", errors: str = "replace") -> Iterator[Tuple[int, str, Any]]:
    """Yield parsed records from ``path`` one line at a time.

    Parameters
    ----------
    path:
        File system path to scan.
    encoding / errors:
        Passed to :func:`Path.open` for decoding the file contents.

    Returns
    -------
    Iterator[Tuple[int, str, Any]]
        Yields ``(line_number, raw_text, entry)`` tuples where ``entry`` is the
        JSON-decoded object if possible, otherwise the original string.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot open '{file_path}': file does not exist")

    with file_path.open("r", encoding=encoding, errors=errors) as handle:
        for line_number, raw_text in enumerate(handle, start=1):
            stripped = raw_text.rstrip("\n")
            if not stripped:
                # Preserve blank lines as empty strings so downstream callers
                # can decide whether they are interesting.
                yield line_number, stripped, ""
                continue

            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                entry = stripped
            yield line_number, stripped, entry


def search_in_file(
    path: Union[str, Path],
    query: QueryType,
    *,
    fields: Optional[Sequence[str]] = None,
    case_sensitive: bool = False,
    limit: Optional[int] = None,
) -> list[SearchResult]:
    """Return matches for ``query`` inside a JSONL or plain-text file.

    ``query`` accepts several forms:

    - ``str``: substring search.
    - ``re.Pattern``: regular expression ``search``.
    - ``Sequence`` of the above: every element must match the same line.
    - ``Callable``: predicate receiving the parsed entry; returning ``True``
      marks the line as a match.

    Parameters
    ----------
    path:
        File to scan.
    query:
        Predicate describing what constitutes a match.
    fields:
        Optional iterable describing which JSON keys to search. Nested keys can
        be addressed with ``dot.notation``. When omitted the entire entry (or
        line) is considered.
    case_sensitive:
        When ``False`` (default) searches are performed using :meth:`str.casefold`.
    limit:
        Maximum number of matches to return. ``None`` means no limit.
    """

    matcher = _build_entry_matcher(query, fields=fields, case_sensitive=case_sensitive)
    results: list[SearchResult] = []

    for line_number, raw_text, entry in iter_records(path):
        if matcher(entry, raw_text):
            results.append(SearchResult(path=Path(path), line_number=line_number, entry=entry, raw_text=raw_text))
            if limit is not None and len(results) >= limit:
                break

    return results


def search_paths(
    paths: Iterable[Union[str, Path]],
    query: QueryType,
    *,
    fields: Optional[Sequence[str]] = None,
    case_sensitive: bool = False,
    limit_per_file: Optional[int] = None,
) -> list[SearchResult]:
    """Search across multiple files and return all matching records."""

    collected: list[SearchResult] = []
    for path in paths:
        matches = search_in_file(
            path,
            query,
            fields=fields,
            case_sensitive=case_sensitive,
            limit=limit_per_file,
        )
        collected.extend(matches)
    return collected


def _build_entry_matcher(
    query: QueryType,
    *,
    fields: Optional[Sequence[str]],
    case_sensitive: bool,
) -> Callable[[Any, str], bool]:
    if callable(query) and not isinstance(query, re.Pattern):
        return lambda entry, raw: bool(query(entry))

    text_matcher = _build_text_matcher(query, case_sensitive=case_sensitive)

    def matcher(entry: Any, raw_text: str) -> bool:
        searchable = _entry_to_searchable(entry, raw_text, fields)
        return text_matcher(searchable)

    return matcher


def _build_text_matcher(query: QueryType, *, case_sensitive: bool) -> Callable[[str], bool]:
    if isinstance(query, re.Pattern):
        pattern = query if case_sensitive or (query.flags & re.IGNORECASE) else re.compile(query.pattern, query.flags | re.IGNORECASE)
        return lambda value: bool(pattern.search(value))

    if isinstance(query, Sequence) and not isinstance(query, (str, bytes)):
        sub_matchers = [_build_text_matcher(subquery, case_sensitive=case_sensitive) for subquery in query]
        return lambda value: all(match(value) for match in sub_matchers)

    if isinstance(query, str):
        processed = query if case_sensitive else query.casefold()
        if case_sensitive:
            return lambda value: processed in value
        return lambda value: processed in value.casefold()

    raise TypeError("Unsupported query type: expected str, Pattern, Sequence or callable")


def _entry_to_searchable(entry: Any, raw_text: str, fields: Optional[Sequence[str]]) -> str:
    if fields and isinstance(entry, Mapping):
        values = []
        for field in fields:
            value = _extract_field(entry, field)
            if value is None:
                continue
            values.append(_stringify(value))
        if values:
            return " ".join(values)
        # Fall back to raw text when none of the requested fields are present.

    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        return json.dumps(entry, sort_keys=True)
    return raw_text


def _extract_field(entry: JsonLike, dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    current: Any = entry
    for part in parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return json.dumps(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    return str(value)


__all__ = [
    "SearchResult",
    "iter_records",
    "search_in_file",
    "search_paths",
]

