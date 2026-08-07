"""The Retrieval agent.

Owns DNABERT-S and the FAISS index. Answers a request for candidates with corpus row IDs
and cosine scores -- never with nucleotides.

Run it:  python3 -m multiagent_system.agents.retrieval --port 8101
"""
from __future__ import annotations

import sys
from typing import Any

from .. import config
from ..a2a.card import skill
from ..runtime import launcher
from ..runtime.agent import AgentSpec, no_result_payload
from ..runtime.state import REGISTRY, Phase, RetrievalSession
from ..tools import retrieval_tools as tools
from ..tools.corpus_store import corpus_fingerprint
from ..tools.organism_index import get_organism_index

SYSTEM_PROMPT = """\
You are the retrieval agent in a genomic gap-filling system. You find reference gene
sequences that resemble the region around a gap in a draft genome.

You have exactly three tools and no others:
  retrieve_context  opens a ranked candidate pool and serves the first batch
  extend_context    serves the next batch from a pool already open
  describe_rows     organism and protein metadata for rows already served

The request tells you the operation, the request_id and the batch size. Do this:
  - operation "retrieve": call retrieve_context once with the request_id and batch_size.
  - operation "extend": call extend_context once with the request_id and batch_size.

The query sequence and any organism restriction are already held by the tools. You never
see them and never pass them.

Rules:
  - Pass the request_id and batch_size through unchanged. Do not adjust them.
  - Never invent, guess or modify a row ID. Only the tools produce them.
  - Never write out nucleotide sequences. You do not have them and must not fabricate them.
  - "pool_exhausted": true is a normal, successful answer meaning the ranking is spent.
    Report it; do not retry.
  - If a tool returns an error, read its "remedy" field, follow it once, and if it fails
    again say so plainly and stop.

When the tool has answered, reply with one short sentence saying how many row IDs you
served. Do not list them; the caller reads them from the tool result, not from your text.
"""

SKILLS = [
    skill(
        skill_id="dna_similarity_retrieval",
        name="DNA similarity retrieval",
        description=(
            "Embed the sequence flanking a gap with DNABERT-S and return the most similar "
            "coding sequences from a 43,575-record reference corpus, as row IDs and cosine "
            "scores."
        ),
        tags=["genomics", "retrieval", "faiss", "dnabert-s"],
        examples=[
            "Find reference genes resembling the 1800 bases flanking this gap.",
            "Serve the next batch of candidates from the open pool.",
        ],
    ),
    skill(
        skill_id="organism_restricted_retrieval",
        name="Organism-restricted retrieval",
        description=(
            "Confine the search to a named organism's contiguous block of the corpus, "
            "reporting the size of that restricted set."
        ),
        tags=["genomics", "retrieval", "filtering"],
        examples=["Search only within Fusobacterium animalis."],
    ),
]


def make_state(data: dict[str, Any], task_id: str) -> RetrievalSession:
    """One ranked pool per ``request_id``, reused across calls.

    A pool has to outlive the A2A request that opened it: the Reconstruction agent
    retrieves a first batch, admits what fits, and only then decides whether to extend.
    So a session for a known ``request_id`` is handed back rather than rebuilt -- which
    is also what makes ``extend_context`` free of any embedding work.

    Sessions are reset when the caller re-opens with a different query, and expire with
    the deadline inherited from ``BaseState``.
    """
    request_id = str(data.get("request_id") or "").strip() or task_id
    existing = REGISTRY.get(request_id)
    if isinstance(existing, RetrievalSession) and existing.phase is not Phase.TERMINATED:
        return existing
    return RetrievalSession(
        task_id=request_id,
        context_id=str(data.get("context_id") or ""),
        query_text=str(data.get("query_text") or ""),
        organism=str(data.get("organism") or ""),
    )


def build_prompt(state: RetrievalSession, data: dict[str, Any]) -> str:
    operation = str(data.get("op") or "retrieve")
    batch_size = int(data.get("batch_size") or 4)
    if operation == "extend":
        return (
            f"Operation: extend.\n"
            f"request_id: {state.task_id}\n"
            f"batch_size: {batch_size}\n\n"
            f"Call extend_context once with exactly these values, then report how many "
            f"row IDs you served."
        )
    return (
        f"Operation: retrieve.\n"
        f"request_id: {state.task_id}\n"
        f"batch_size: {batch_size}\n"
        f"organism restriction: {state.organism or '(none - the whole corpus)'}\n"
        f"query: {len(state.query_text)} bases, held by the tool\n\n"
        f"Call retrieve_context once with exactly this request_id and batch_size, then "
        f"report how many row IDs you served."
    )


def finish(state: RetrievalSession) -> dict[str, Any]:
    """Serialised from state, not from the model's prose.

    Reports the batch this request served. An empty batch with ``pool_exhausted`` true is
    a successful answer, not a failure -- so it must not be confused with the model
    having failed to call the tool at all, which is what ``no_result_payload`` marks.
    """
    if state.phase is not Phase.POOLED:
        return no_result_payload(state)
    return {
        "request_id": state.task_id,
        "row_ids": list(state.last_batch),
        "scores": list(state.last_scores),
        "path": state.path,
        "pool_size": state.pool_size,
        "served_total": len(state.served),
        "pool_exhausted": state.exhausted,
        "restricted_set_size": state.restricted_set_size,
        "organism": state.resolved_organism,
    }


def build_spec() -> AgentSpec:
    return AgentSpec(
        role=config.RETRIEVAL,
        name="GenomIO Retrieval Agent",
        description=(
            "Finds reference coding sequences resembling the region around a genomic gap, "
            "using DNABERT-S embeddings over a FAISS index. Returns corpus row IDs and "
            "cosine scores, never nucleotides."
        ),
        system_prompt=SYSTEM_PROMPT,
        skills=SKILLS,
        tools=tools.TOOLS,
        server_key=tools.SERVER_KEY,
        allowed_tools=tools.ALLOWED_TOOLS,
        warm=tools.warm,
        make_state=make_state,
        build_prompt=build_prompt,
        finish=finish,
        has_result=lambda state: state.phase is Phase.POOLED,
        artifact_name="retrieval_candidates",
        max_turns=8,
        # The ranked pool must outlive the request that opened it, so extend_context can
        # serve the next batch without re-embedding.
        persist_state=True,
        organisms=lambda: get_organism_index().names(),
        fingerprint=corpus_fingerprint,
        extra_card_fields={"x-embedder": "zhihan1996/DNABERT-S", "x-metric": "cosine"},
    )


if __name__ == "__main__":
    sys.exit(launcher.run(build_spec, config.RETRIEVAL))
