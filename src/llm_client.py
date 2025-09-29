# version: 0.1.0
# path: src/llm_client.py

"""Client helpers for delegating planning to a local LM Studio server."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests

from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are the planning module for EveBot. "
    "Respond with JSON containing an 'actions' list. Each action is a mapping "
    "with a lowercase 'type' key and optional parameters such as coordinates, "
    "keys, text, or wait durations."
)


class LMStudioClient:
    """Thin wrapper around the LM Studio HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        plan_path: str = "/v1/chat/completions",
        model: str = "lmstudio",
        temperature: float = 0.2,
        timeout: float = 10.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.plan_path = plan_path or "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._session = requests.Session()

    # ---------------------------------------------------------------------
    def plan_actions(self, perception: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Post ``perception`` to LM Studio and return the decoded plan."""

        if not self.base_url:
            logger.debug("LM Studio base URL not configured; skipping request")
            return None

        url = urljoin(f"{self.base_url}/", self.plan_path.lstrip("/"))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(perception, ensure_ascii=False, indent=2),
                },
            ],
            "temperature": self.temperature,
        }

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - runtime guard
            logger.warning("LM Studio request failed: %s", exc)
            return None

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive logging
            logger.warning("Invalid LM Studio response: %s", exc)
            return None

        content = self._extract_message_content(data)
        if not content:
            logger.warning("LM Studio response missing content: %s", data)
            return None

        actions = self._parse_actions(content)
        if actions is None:
            logger.warning("Unable to parse LLM plan: %s", content)
        return actions

    # ------------------------------------------------------------------
    def _extract_message_content(self, payload: Dict[str, Any]) -> Optional[str]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        return content if isinstance(content, str) else None

    # ------------------------------------------------------------------
    def _parse_actions(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Attempt to load an action list from the returned text."""

        candidates: Iterable[str]
        text = content.strip()
        if "```" in text:
            blocks = []
            for segment in text.split("```"):
                segment = segment.strip()
                if not segment:
                    continue
                if segment.startswith("json"):
                    segment = segment[4:].strip()
                blocks.append(segment)
            candidates = blocks
        else:
            candidates = [text]

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            actions = self._normalise_payload(parsed)
            if actions is not None:
                return actions
        return None

    # ------------------------------------------------------------------
    def _normalise_payload(self, payload: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(payload, list):
            return [self._ensure_action_dict(item) for item in payload if item]
        if isinstance(payload, dict):
            if "actions" in payload and isinstance(payload["actions"], list):
                return [
                    self._ensure_action_dict(item)
                    for item in payload["actions"]
                    if item
                ]
        return None

    # ------------------------------------------------------------------
    def _ensure_action_dict(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return {k.lower(): v for k, v in item.items()}
        if isinstance(item, str):
            return {"type": item.lower()}
        return {"type": "noop"}


__all__ = ["LMStudioClient"]
