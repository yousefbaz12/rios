from __future__ import annotations
from rios.engines import make_knowledge_engine
from rios.cli import initial_state
from rios.core.supervisor import build_graph

CFG = {"configurable": {"thread_id": "test"}}

class FakeArxiv:
    def search(self, query, max_results=5):
        return [{"id": "0001", "title": "Real-Shaped Paper", "authors": ["A. Researcher"],
                 "abstract": "stub", "published": "2024-01-01", "categories": ["cs.AI"],
                 "url": "https://arxiv.org/abs/0001", "pdf_url": "https://arxiv.org/pdf/0001"}] * 3

def test_knowledge_engine_real_contract_offline():
    out = make_knowledge_engine(FakeArxiv())(initial_state("anything"))
    assert len(out["knowledge"]) == 3
    assert out["knowledge"][0]["url"].startswith("https://arxiv.org")

def test_pipeline_carries_real_knowledge_shape():
    final = build_graph(arxiv_tool=FakeArxiv()).invoke(initial_state("x"), CFG)
    assert final["phase"] == "done" and len(final["knowledge"]) == 3