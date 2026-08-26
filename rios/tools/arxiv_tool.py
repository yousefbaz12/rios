"""MCP Tool Integration Layer — arXiv search (stdlib-only, bounded, cached)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
CACHE_DIR = Path(os.environ.get("RIOS_CACHE", Path.home() / ".rios" / "cache"))


class ArxivError(RuntimeError):
    """arXiv unreachable after all attempts."""


class ArxivTool:
    def __init__(self, timeout: float = 15.0, attempts: int = 3,
                 cache_ttl_hours: float = 24.0):
        self.timeout = timeout
        self.attempts = attempts
        self._ttl = cache_ttl_hours * 3600

    def search(self, query: str, max_results: int = 5,
               use_cache: bool = True) -> list[dict]:
        path = self._cache_path(query, max_results)
        if use_cache and path.exists():
            blob = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - blob["ts"] < self._ttl:
                print(f"→ arXiv cache hit ({len(blob['papers'])} papers, 0.0s)", flush=True)
                return blob["papers"]

        qs = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        url = f"{API_URL}?{qs}"

        last_err: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            t0 = time.time()
            print(f"→ arXiv attempt {attempt}/{self.attempts} "
                  f"(timeout {self.timeout:.0f}s)…", flush=True)
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "RIOS/0.1 (research-os)"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    papers = self._parse(resp.read())
                print(f"← {len(papers)} papers in {time.time() - t0:.1f}s", flush=True)
                self._write_cache(path, papers)
                return papers
            except urllib.error.HTTPError as exc:
                last_err = exc
                backoff = 30.0 if exc.code == 429 else 3.0 if exc.code == 503 else 1.0
                print(f"   HTTP {exc.code} after {time.time() - t0:.1f}s — "
                      f"backing off {backoff:.0f}s", flush=True)
                time.sleep(backoff)
            except Exception as exc:
                last_err = exc
                print(f"   {type(exc).__name__} after {time.time() - t0:.1f}s: {exc}",
                      flush=True)
                time.sleep(2.0)

        raise ArxivError(f"arXiv unreachable after {self.attempts} attempts: {last_err}")

    @staticmethod
    def _parse(xml_bytes: bytes) -> list[dict]:
        root = ET.fromstring(xml_bytes)
        papers = []
        for e in root.findall(f"{ATOM}entry"):
            def text(tag: str) -> str:
                el = e.find(f"{ATOM}{tag}")
                return (el.text or "").replace("\n", " ").strip() if el is not None else ""
            links = {l.get("title"): l.get("href") for l in e.findall(f"{ATOM}link")}
            papers.append({
                "id": text("id").rsplit("/", 1)[-1],
                "title": text("title"),
                "authors": [(a.find(f"{ATOM}name").text or "").strip()
                            for a in e.findall(f"{ATOM}author")][:5],
                "abstract": text("summary"),
                "published": text("published")[:10],
                "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
                "url": text("id"),
                "pdf_url": links.get("pdf", ""),
            })
        return papers

    @staticmethod
    def _cache_path(query: str, max_results: int) -> Path:
        key = hashlib.sha256(
            f"{query.lower().strip()}|{max_results}".encode()).hexdigest()[:16]
        return CACHE_DIR / f"arxiv_{key}.json"

    @staticmethod
    def _write_cache(path: Path, papers: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ts": time.time(), "papers": papers},
                                   ensure_ascii=False), encoding="utf-8")