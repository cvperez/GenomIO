"""The Coordinator.

Loads no models and wraps no RAG code. It slices the retrieval query, checks its peers
agree on the corpus, delegates, and applies the acceptance gate -- the check that turns
"RAG-enabled" from a label into a measured precondition.

Run it:  python3 -m multiagent_system.agents.coordinator --port 8100
"""
from __future__ import annotations

import sys
from typing import Any

from .. import config
from ..a2a.card import skill
from ..runtime import launcher
from ..runtime.agent import AgentSpec, no_result_payload
from ..runtime.state import CoordTask, Phase
from ..tools import coordinator_tools as tools

SYSTEM_PROMPT = """\
You are the coordinator of a genomic gap-filling system. You do not fill gaps yourself
and you do not handle sequences; you delegate and then you check the answer.

Call your four tools in this order, once each:

1. prepare_query(task_id)
   Slices the retrieval query, verifies both peer agents are ready and reading the same
   reference corpus, and picks restricted or unrestricted mode.

2. delegate_reconstruction(task_id)
   Sends the task to the reconstruction agent and waits. This can take several minutes;
   that is expected. It returns a summary of what came back.

3. evaluate_result(task_id)
   Applies the acceptance gate: the produced length must be within tolerance of the
   target, every character must be A, C, G, T or N, and the number of retrieved context
   tokens that actually reached the model must be greater than zero.

4. emit_trace(task_id)
   Produces the run record.

Rules:
  - Never fill the gap yourself and never write out a nucleotide sequence. You are not
    given one.
  - Never adjust gap_length.
  - If a tool returns an error, read its "remedy" field and follow it. If a peer is
    unreachable or fails, still call emit_trace so the run leaves a record, then say
    plainly what failed. A rejected result is a valid outcome; do not retry to obtain a
    nicer one.

Finish with two or three sentences: the verdict, how many context tokens were admitted,
and the produced length against the target.
"""

SKILLS = [
    skill(
        skill_id="fill_genomic_gap",
        name="Fill a genomic gap",
        description=(
            "Given two flanking contigs and a target gap length, orchestrate retrieval and "
            "reconstruction and return the filled sequence with a verdict and a trace."
        ),
        tags=["genomics", "orchestration", "rag"],
        examples=[
            "Fill the 870 base gap between contig1 and contig2 of AP012051.1.",
            "Fill this gap using only Fusobacterium animalis references.",
        ],
    ),
    skill(
        skill_id="acceptance_gate",
        name="Acceptance gate",
        description=(
            "Verify that a reconstruction has a plausible length, a valid nucleotide "
            "alphabet, and -- the check the previous pipeline lacked -- that retrieved "
            "context actually reached the model."
        ),
        tags=["genomics", "validation", "quality"],
        examples=["Did retrieved context actually reach the model on this run?"],
    ),
]


def make_state(data: dict[str, Any], task_id: str) -> CoordTask:
    left = str(data.get("left") or "")
    right = str(data.get("right") or "")
    gap_length = int(data.get("gap_length") or 0)
    if not left or not right:
        raise ValueError("both 'left' and 'right' contig sequences are required")
    if gap_length <= 0:
        raise ValueError("'gap_length' must be a positive integer")

    return CoordTask(
        task_id=task_id,
        context_id=str(data.get("context_id") or ""),
        accession=str(data.get("accession") or ""),
        gap_id=str(data.get("gap_id") or "(unnamed)"),
        gap_length=gap_length,
        organism=str(data.get("organism") or ""),
        left=left,
        right=right,
        true_sequence=str(data.get("true_sequence") or ""),
    )


def build_prompt(state: CoordTask, data: dict[str, Any]) -> str:
    return (
        f"Fill one gap in a draft genome.\n\n"
        f"task_id: {state.task_id}\n"
        f"accession: {state.accession or '(unspecified)'}\n"
        f"gap_id: {state.gap_id}\n"
        f"gap_length: {state.gap_length}\n"
        f"left contig: {len(state.left)} bases\n"
        f"right contig: {len(state.right)} bases\n"
        f"organism restriction: {state.organism or '(none)'}\n\n"
        f"The sequences are held by the tools. Start with prepare_query using the "
        f"task_id above."
    )


def finish(state: CoordTask) -> dict[str, Any]:
    """The Coordinator's answer, assembled from state.

    ``build_trace`` runs whether or not the model got that far, so a run that stopped
    early still leaves a record saying where it stopped rather than leaving nothing.
    """
    trace = tools.build_trace(state)
    result = state.peer_result or {}
    sequence = str(result.get("sequence", ""))

    payload: dict[str, Any] = {
        "task_id": state.task_id,
        "gap_id": state.gap_id,
        "accession": state.accession,
        "gap_length": state.gap_length,
        "sequence": sequence,
        "trace": trace,
    }
    if not sequence or state.phase is not Phase.EVALUATED:
        payload.update(no_result_payload(state))
        payload["trace"] = trace
    return payload


def has_result(state: CoordTask) -> bool:
    return bool((state.peer_result or {}).get("sequence")) and state.phase is Phase.EVALUATED


def build_spec() -> AgentSpec:
    return AgentSpec(
        role=config.COORDINATOR,
        name="GenomIO Coordinator Agent",
        description=(
            "Orchestrates retrieval-augmented genomic gap filling: slices the query flanks, "
            "delegates to the reconstruction and retrieval agents, and gates the result on "
            "length, alphabet and whether retrieved context actually reached the model."
        ),
        system_prompt=SYSTEM_PROMPT,
        skills=SKILLS,
        tools=tools.TOOLS,
        server_key=tools.SERVER_KEY,
        allowed_tools=tools.ALLOWED_TOOLS,
        warm=lambda: {"models": "none"},
        make_state=make_state,
        build_prompt=build_prompt,
        finish=finish,
        has_result=has_result,
        artifact_name="gap_fill",
        max_turns=config.MAX_TURNS,
        extra_card_fields={
            "x-query-flank": config.QUERY_FLANK,
            "x-tolerance": config.TOLERANCE,
        },
    )


if __name__ == "__main__":
    sys.exit(launcher.run(build_spec, config.COORDINATOR))
