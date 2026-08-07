"""Configuration for the GenomIO multi-agent system.

Every value is overridable from the environment so a SLURM launch, a local run and a
test can share one code path. Importing this module also puts ``<repo>/src`` on
``sys.path``, which is what lets the agents reuse ``rag.index`` / ``rag.dnabert_s``
without touching anything under ``src/``.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    return value if value.strip() else default


# --------------------------------------------------------------------------- roles
COORDINATOR = "coordinator"
RETRIEVAL = "retrieval"
RECONSTRUCTION = "reconstruction"
ROLES = (COORDINATOR, RETRIEVAL, RECONSTRUCTION)

# MCP server keys. The tool id the model sees is mcp__<server_key>__<tool_name>, so
# these strings are load-bearing: drift between here and create_sdk_mcp_server(name=...)
# produces a silent "no tools available" rather than an error.
SERVER_KEYS = {
    COORDINATOR: "coordinator_agent",
    RETRIEVAL: "retrieval_agent",
    RECONSTRUCTION: "reconstruction_agent",
}

# --------------------------------------------------------------------------- network
BASE_PORT = _env_int("GENOMIO_BASE_PORT", 8100)
PORTS = {
    COORDINATOR: _env_int("GENOMIO_PORT_COORDINATOR", BASE_PORT),
    RETRIEVAL: _env_int("GENOMIO_PORT_RETRIEVAL", BASE_PORT + 1),
    RECONSTRUCTION: _env_int("GENOMIO_PORT_RECONSTRUCTION", BASE_PORT + 2),
}
BIND_HOST = _env_str("GENOMIO_BIND_HOST", "0.0.0.0")

# Peer URLs. deploy.sh overrides these per node; locally they default to localhost.
PEER_URLS = {
    COORDINATOR: _env_str("GENOMIO_URL_COORDINATOR", f"http://127.0.0.1:{PORTS[COORDINATOR]}"),
    RETRIEVAL: _env_str("GENOMIO_URL_RETRIEVAL", f"http://127.0.0.1:{PORTS[RETRIEVAL]}"),
    RECONSTRUCTION: _env_str(
        "GENOMIO_URL_RECONSTRUCTION", f"http://127.0.0.1:{PORTS[RECONSTRUCTION]}"
    ),
}

# --------------------------------------------------------------------------- science
# Bases either side of the gap used to build the *retrieval* query. 900 sits above the
# measured precision table in docs/dnabert_s_retrieval.md (600 bp -> P@1 0.780) and
# inside DNABERT-S's ~2000 bp token ceiling once both flanks are concatenated.
QUERY_FLANK = _env_int("GENOMIO_QUERY_FLANK", 900)

# Bases either side of the gap kept in the *model* input. 0 disables trimming.
# Measured on gap1 (contigs 5951 + 13996 bp): untrimmed base input is 3305 of 4096
# tokens, leaving ~583 free after the mask block; at 2000 it is 682 tokens, ~3200 free.
MODEL_FLANK = _env_int("GENOMIO_MODEL_FLANK", 2000)

# GENA-LM's context window. Matches MAX_LENGTH in src/core/gap_filler_rag.py.
MAX_LENGTH = _env_int("GENOMIO_MAX_LENGTH", 4096)

# Tokens held back so the mask block and separators always fit.
SAFETY_TOKENS = _env_int("GENOMIO_SAFETY_TOKENS", 32)

# Probability floor for candidate tokens during sampling. Hardcoded at 0.01 inside the
# original predict_until_length; here it is an argument with this default.
DEFAULT_THRESHOLD = _env_float("GENOMIO_THRESHOLD", 0.01)

MAX_ATTEMPTS = _env_int("GENOMIO_MAX_ATTEMPTS", 10)

# Acceptance gate: |produced - target| <= max(3, ceil(TOLERANCE * target)).
TOLERANCE = _env_float("GENOMIO_TOLERANCE", 0.05)

GENA_LM_MODEL = _env_str("GENOMIO_GENA_LM", "AIRI-Institute/gena-lm-bigbird-base-t2t")

# --------------------------------------------------------------------------- retrieval
# Candidates fetched from FAISS per pool. Over-fetching is how the restricted path stays
# correct: HNSW graph traversal with a sparse IDSelectorBatch returns all -1, so we
# search wide and post-filter instead.
POOL_OVERFETCH = _env_int("GENOMIO_POOL_OVERFETCH", 4)
POOL_MIN_K = _env_int("GENOMIO_POOL_MIN_K", 32)
POOL_MAX_K = _env_int("GENOMIO_POOL_MAX_K", 512)
RESTRICTED_MIN_K = _env_int("GENOMIO_RESTRICTED_MIN_K", 256)
RESTRICTED_MAX_K = _env_int("GENOMIO_RESTRICTED_MAX_K", 4000)

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = _env_int("GENOMIO_MAX_BATCH_SIZE", 64)

# --------------------------------------------------------------------------- budgets
# Hard stops that do not depend on the model cooperating.
MAX_TURNS = _env_int("GENOMIO_MAX_TURNS", 24)
TASK_DEADLINE_S = _env_float("GENOMIO_TASK_DEADLINE_S", 900.0)
MAX_CALLS_PER_TOOL = _env_int("GENOMIO_MAX_CALLS_PER_TOOL", 12)

PEER_TIMEOUT_S = _env_float("GENOMIO_PEER_TIMEOUT_S", 600.0)
CLIENT_TIMEOUT_S = _env_float("GENOMIO_CLIENT_TIMEOUT_S", 3600.0)
HEALTH_TIMEOUT_S = _env_float("GENOMIO_HEALTH_TIMEOUT_S", 10.0)

AGENT_MODEL = _env_str("GENOMIO_AGENT_MODEL", "claude-sonnet-4-5")

SEED = _env_int("GENOMIO_SEED", 20260807)

# --------------------------------------------------------------------------- paths
CACHE_DIR = os.path.join(REPO_ROOT, ".cache", "rag_index")
RECORDS_PATH = os.path.join(CACHE_DIR, "records.jsonl")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
CORPUS_DIR = os.path.join(REPO_ROOT, "rag_corpus_uniform")

DATA_DIR = os.path.join(REPO_ROOT, "data", "simulated_draft_genomes")
CONTIGS_DIR = os.path.join(DATA_DIR, "contigs")
GAPS_DIR = os.path.join(DATA_DIR, "gaps")

RUN_DIR = os.path.join(os.path.dirname(__file__), ".run")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

VENV_PYTHON = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python3")

PROTOCOL_VERSION = "0.3.0"
AGENT_VERSION = "1.0.0"
