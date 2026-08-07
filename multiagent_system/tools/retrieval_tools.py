"""MCP tools for the Retrieval agent.

This agent owns the DNABERT-S embedder and the FAISS index. Keeping them in one process
makes the correspondence between embedding dimension and corpus snapshot an internal
invariant rather than a contract that can break at runtime.

Two rules shape the interface:

**Row IDs, never nucleotides.** ``retrieve_context`` in the original returned
``"\\n".join(record["sequence"] for _, record in hits)`` -- a pre-assembled block that
discarded the organism and protein metadata the records carry, and that the caller could
only accept or lose whole. Returning identifiers instead lets the Reconstruction agent
tokenize candidate by candidate and admit what fits, and keeps corpus DNA from crossing
an agent hop as an opaque blob.

**Incremental serving, not a fixed k.** The original defaulted to ``k=2``, a number with
no relationship to how much room the tokenizer had. Two records might overflow the window
or leave most of it unused, and neither case was reported. Here a ranked pool is opened
once and served in batches whose size comes from the agent that measured the budget.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
from claude_agent_sdk import tool

from .. import config
from ..runtime import errors
from ..runtime.blocking import run_blocking
from ..runtime.state import Phase, RetrievalSession, guarded
from .corpus_store import get_store
from .organism_index import get_organism_index

SERVER_KEY = config.SERVER_KEYS[config.RETRIEVAL]
TOOL_NAMES = ("retrieve_context", "extend_context", "describe_rows")
ALLOWED_TOOLS = [f"mcp__{SERVER_KEY}__{name}" for name in TOOL_NAMES]

PATH_UNRESTRICTED = "unrestricted"
PATH_RESTRICTED_OVERFETCH = "restricted_overfetch"
PATH_RESTRICTED_RANGE = "restricted_range"


class _Index:
    index: Any = None
    records: Any = None
    loaded: bool = False


INDEX = _Index()


def warm() -> dict[str, Any]:
    """Load the FAISS index and the embedder before the agent reports itself ready.

    Pre-warming is not an optimisation here. The first ``embed()`` call loads DNABERT-S
    and takes about 32 seconds; inside a tool call that delays the SDK's stdio pump and
    reads as a hang, and it can consume the caller's whole timeout.
    """
    from rag import dnabert_s
    from rag.index import load_or_build_index

    started = time.monotonic()
    INDEX.index, INDEX.records = load_or_build_index()
    dnabert_s.embed_one("ACGT" * 32)  # force the weight load now, not mid-request
    get_store()
    get_organism_index()
    INDEX.loaded = True
    return {
        "vectors": int(INDEX.index.ntotal),
        "organisms": len(get_organism_index().blocks),
        "warm_s": round(time.monotonic() - started, 2),
    }


# --------------------------------------------------------------------------- search
def _embed_query(query_text: str) -> np.ndarray:
    """Embed and L2-normalise.

    The index is an IndexHNSWFlat built with the default L2 metric over vectors that were
    normalised before insertion, so ranking is cosine only if the query is normalised the
    same way. Skipping this does not error -- it silently returns the wrong neighbours.
    """
    import faiss
    from rag import dnabert_s

    vector = dnabert_s.embed_one(query_text).reshape(1, -1).astype("float32")
    vector = np.ascontiguousarray(vector)
    faiss.normalize_L2(vector)
    return vector


def _cosine(distance: float) -> float:
    """Squared L2 between unit vectors back to cosine, as rag/index.py does."""
    return 1.0 - float(distance) / 2.0


def _search_unrestricted(query_text: str, k: int) -> tuple[list[int], list[float]]:
    vector = _embed_query(query_text)
    k = max(1, min(k, int(INDEX.index.ntotal)))
    distances, indices = INDEX.index.search(vector, k)
    rows, scores = [], []
    for distance, row in zip(distances[0], indices[0]):
        if int(row) < 0:
            continue
        rows.append(int(row))
        scores.append(round(_cosine(distance), 4))
    return rows, scores


def _search_restricted(
    query_text: str, block: tuple[int, int], k: int
) -> tuple[list[int], list[float], str]:
    """Confine the search to one organism's contiguous row block.

    Over-fetch and post-filter by default. HNSW with a sparse ``IDSelectorBatch`` returns
    all ``-1`` -- greedy graph traversal cannot reach enough accepted nodes -- and raising
    efSearch does not fix it. ``IDSelectorRange`` copes better with a contiguous block, so
    it is the fallback when post-filtering came up short, and its output is checked for
    the ``-1`` under-fill signature before being trusted.
    """
    import faiss

    lo, hi = block
    vector = _embed_query(query_text)
    wide = max(1, min(k, int(INDEX.index.ntotal)))
    distances, indices = INDEX.index.search(vector, wide)

    rows, scores = [], []
    for distance, row in zip(distances[0], indices[0]):
        row = int(row)
        if lo <= row < hi:
            rows.append(row)
            scores.append(round(_cosine(distance), 4))
    if rows:
        return rows, scores, PATH_RESTRICTED_OVERFETCH

    selector = faiss.IDSelectorRange(lo, hi)
    params = faiss.SearchParametersHNSW()
    params.sel = selector
    params.efSearch = max(config.RESTRICTED_MIN_K, 128)
    distances, indices = INDEX.index.search(vector, min(k, hi - lo), params=params)
    for distance, row in zip(distances[0], indices[0]):
        row = int(row)
        if row < 0:  # the under-fill signature; drop rather than report a bogus hit
            continue
        if lo <= row < hi:
            rows.append(row)
            scores.append(round(_cosine(distance), 4))
    return rows, scores, PATH_RESTRICTED_RANGE


def _fingerprint(query_text: str, organism: str) -> str:
    digest = hashlib.sha256(f"{query_text}|{organism}".encode("utf-8"))
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- tools
@tool(
    "retrieve_context",
    "Open a ranked pool of corpus candidates for this gap and serve the first batch. "
    "The query sequence is already held by this agent; you only choose the batch size. "
    "Returns row IDs and cosine scores only, never nucleotide sequences.",
    {"request_id": str, "batch_size": int},
)
@guarded("retrieve_context", id_field="request_id", requires=None)
async def retrieve_context(args: dict[str, Any], state: RetrievalSession) -> dict[str, Any]:
    batch_size = _as_int(args.get("batch_size"), 0)
    if not (config.MIN_BATCH_SIZE <= batch_size <= config.MAX_BATCH_SIZE):
        return errors.tool_error(
            errors.BAD_BATCH_SIZE,
            remedy=f"batch_size must be between {config.MIN_BATCH_SIZE} and "
            f"{config.MAX_BATCH_SIZE}.",
            request_id=state.task_id,
            given=args.get("batch_size"),
        )

    # From state, not from the model. The query never enters the prompt, so it cannot be
    # paraphrased, truncated or refused -- and QUERY_CHANGED below can only fire when the
    # *caller* genuinely re-opened the pool with something different.
    query_text = state.query_text.strip()
    if not query_text:
        return errors.tool_error(
            errors.EMPTY_QUERY,
            remedy="This request carried no query sequence. Report the problem and stop.",
            request_id=state.task_id,
        )

    organism = state.organism.strip()
    fingerprint = _fingerprint(query_text, organism)

    if state.phase is Phase.POOLED:
        if state.query_fingerprint != fingerprint:
            # A paraphrased or re-sliced query mid-run would silently change what the
            # scores mean while the caller kept treating the pool as one ranking.
            return errors.tool_error(
                errors.QUERY_CHANGED,
                remedy="A pool is already open for this request_id with a different "
                "query. Call extend_context to continue it.",
                request_id=state.task_id,
            )
        rows, scores = state.next_batch(batch_size)
        return errors.tool_ok(_payload(state, rows, scores, reused=True))

    if organism:
        organisms = get_organism_index()
        block = organisms.block(organism)
        if block is None:
            return errors.tool_error(
                errors.UNKNOWN_ORGANISM,
                remedy="Retry without an organism to search the whole corpus, or use one "
                "of the names listed.",
                request_id=state.task_id,
                given=organism,
                known=organisms.names(),
            )
        wide = max(
            config.RESTRICTED_MIN_K,
            min(config.RESTRICTED_MAX_K, (block[1] - block[0]) * 2),
        )
        rows, scores, path = await run_blocking(_search_restricted, query_text, block, wide)
        state.resolved_organism = organisms.resolve(organism) or organism
        state.restricted_set_size = block[1] - block[0]
        if not rows:
            return errors.tool_error(
                errors.RESTRICTED_POOL_EMPTY,
                remedy="The organism-restricted search returned nothing. Retry without "
                "an organism to search the whole corpus.",
                request_id=state.task_id,
                organism=state.resolved_organism,
                restricted_set_size=state.restricted_set_size,
            )
    else:
        wide = max(
            config.POOL_MIN_K, min(config.POOL_MAX_K, batch_size * config.POOL_OVERFETCH)
        )
        rows, scores = await run_blocking(_search_unrestricted, query_text, wide)
        path = PATH_UNRESTRICTED

    state.query_fingerprint = fingerprint
    state.path = path
    state.ranked_row_ids = rows
    state.ranked_scores = scores
    state.cursor = 0
    state.advance(Phase.POOLED)

    served, served_scores = state.next_batch(batch_size)
    return errors.tool_ok(_payload(state, served, served_scores, reused=False))


@tool(
    "extend_context",
    "Serve the next batch of row IDs from the pool already open for this request_id. "
    "Costs no embedding work. Returns an empty list with pool_exhausted true when the "
    "ranking is spent, which is a normal terminal signal and not an error.",
    {"request_id": str, "batch_size": int},
)
@guarded("extend_context", id_field="request_id", requires=Phase.POOLED)
async def extend_context(args: dict[str, Any], state: RetrievalSession) -> dict[str, Any]:
    batch_size = _as_int(args.get("batch_size"), 0)
    if not (config.MIN_BATCH_SIZE <= batch_size <= config.MAX_BATCH_SIZE):
        return errors.tool_error(
            errors.BAD_BATCH_SIZE,
            remedy=f"batch_size must be between {config.MIN_BATCH_SIZE} and "
            f"{config.MAX_BATCH_SIZE}.",
            request_id=state.task_id,
            given=args.get("batch_size"),
        )
    rows, scores = state.next_batch(batch_size)
    return errors.tool_ok(_payload(state, rows, scores, reused=True))


@tool(
    "describe_rows",
    "Return organism, accession and protein metadata for row IDs already served in this "
    "request. Diagnostic only; never returns nucleotides.",
    {"request_id": str, "row_ids": list},
)
@guarded("describe_rows", id_field="request_id", requires=Phase.POOLED, max_calls=4)
async def describe_rows(args: dict[str, Any], state: RetrievalSession) -> dict[str, Any]:
    raw = args.get("row_ids")
    if not isinstance(raw, list) or not raw:
        return errors.tool_error(
            errors.BAD_ROW_IDS,
            remedy="row_ids must be a non-empty list of row numbers already served.",
            request_id=state.task_id,
        )
    store = get_store()
    served = set(state.served)
    described = []
    for item in raw[: config.MAX_BATCH_SIZE]:
        try:
            row = int(item)
        except (TypeError, ValueError):
            continue
        if row in served and 0 <= row < len(store):
            described.append(await run_blocking(store.summary, row))
    return errors.tool_ok({"request_id": state.task_id, "records": described})


def _payload(
    state: RetrievalSession, rows: list[int], scores: list[float], *, reused: bool
) -> dict[str, Any]:
    exhausted = state.exhausted
    return {
        "request_id": state.task_id,
        "row_ids": rows,
        "scores": scores,
        "path": state.path,
        "pool_size": state.pool_size,
        "served_total": len(state.served),
        "pool_exhausted": exhausted,
        "restricted_set_size": state.restricted_set_size,
        "organism": state.resolved_organism,
        "reused_pool": reused,
        "next_step": (
            "The pool is exhausted; report these row IDs and finish."
            if exhausted
            else "More candidates remain; the caller may ask for another batch."
        ),
    }


TOOLS = [retrieve_context, extend_context, describe_rows]


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
