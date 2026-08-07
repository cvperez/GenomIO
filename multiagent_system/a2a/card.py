"""Agent Card construction -- the A2A discovery document.

Served at both ``/.well-known/agent-card.json`` (the path in the A2A specification) and
``/.well-known/agent-configuration`` (the path the agentic-deployment-engine transport
fetches). Identical bodies; the alias exists so either client works.

Two non-standard fields are carried in ``metadata``:

  x-corpus-fingerprint  hash of the retrieval index manifest. The Coordinator compares
                        Retrieval's against Reconstruction's before delegating, because
                        the FAISS row -> record mapping is positional: if the two agents
                        ever read different corpus snapshots, every result is
                        confidently wrong rather than obviously broken.
  x-organisms           the organism names Retrieval can restrict a search to.
"""
from __future__ import annotations

from typing import Any

from .. import config


def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    skills: list[dict[str, Any]],
    corpus_fingerprint: str = "",
    organisms: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if corpus_fingerprint:
        metadata["x-corpus-fingerprint"] = corpus_fingerprint
    if organisms:
        metadata["x-organisms"] = list(organisms)

    return {
        "protocolVersion": config.PROTOCOL_VERSION,
        "name": name,
        "description": description,
        "url": url,
        "preferredTransport": "JSONRPC",
        "version": config.AGENT_VERSION,
        "provider": {
            "organization": "GenomIO",
            "url": "https://github.com/GenomIO",
        },
        "capabilities": {
            # Both false on purpose. There is one long call per run and the caller
            # blocks on the final artifact, so neither buys anything here. The server
            # answers message/stream and the pushNotificationConfig methods with
            # -32601 rather than pretending.
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["text", "text/plain", "application/json"],
        "defaultOutputModes": ["text", "text/plain", "application/json"],
        "securitySchemes": {},
        "security": [],
        "skills": skills,
        "metadata": metadata,
    }


def skill(
    *,
    skill_id: str,
    name: str,
    description: str,
    tags: list[str],
    examples: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": skill_id,
        "name": name,
        "description": description,
        "tags": tags,
    }
    if examples:
        entry["examples"] = examples
    return entry
