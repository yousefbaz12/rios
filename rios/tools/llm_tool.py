"""MCP Tool Integration Layer — LLM adapter (Groq free tier by default)."""
from __future__ import annotations

import json
import os
import re
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def split_think(text: str) -> tuple[str, str]:
    """Extract a <think>...</think> reasoning trace. Returns (think, rest)."""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    return "", text


def extract_json(text: str) -> dict:
    """Robustly pull a JSON object out of an LLM reply."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(text[start:end + 1])


class LLMTool:
    """Generic LLM adapter with structured output support."""

    def __init__(self, provider: Literal["groq", "openai", "anthropic", "ollama"] | None = None,
                 model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self.model = model or self._default_model()
        self._client = self._build_client()

    def _default_model(self) -> str:
        return {
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama": "llama3.1:8b",
        }[self.provider]

    def _build_client(self):
        if self.provider == "groq":
            from openai import OpenAI
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                    "then either:\n  1) Set in PowerShell: $env:GROQ_API_KEY='gsk_...'\n"
                    "  2) Or create .env file with: GROQ_API_KEY=gsk_...")
            return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        elif self.provider == "openai":
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set.")
            return OpenAI(api_key=key)
        elif self.provider == "anthropic":
            import anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            return anthropic.Anthropic(api_key=key)
        elif self.provider == "ollama":
            from openai import OpenAI
            return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Send messages, return text response."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model, max_tokens=2048,
                temperature=temperature, messages=messages)
            return response.content[0].text
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature)
        return response.choices[0].message.content

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Send messages, parse JSON response. Retries once if invalid."""
        for attempt in range(2):
            text = self.chat(messages, temperature=temperature)
            _, rest = split_think(text)
            try:
                return extract_json(rest)
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "Your response was not valid JSON. "
                                                    "Respond with ONLY valid JSON."},
                    ]
                else:
                    raise ValueError(f"LLM did not return valid JSON after retry: {exc}")