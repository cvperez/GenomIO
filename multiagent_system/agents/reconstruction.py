"""The Reconstruction agent.

Owns GENA-LM, so it is the only component that knows the token budget -- and therefore
the only one that can decide how much retrieved context fits. It negotiates with the
Retrieval agent over A2A until the window is full or the candidate pool is spent, then
generates.

Run it:  python3 -m multiagent_system.agents.reconstruction --port 8102
"""
from __future__ import annotations

import sys
from typing import Any

from .. import config
from ..a2a.card import skill
from ..runtime import launcher
from ..runtime.agent import AgentSpec, no_result_payload
from ..runtime.state import Phase, ReconTask
from ..tools import reconstruction_tools as tools
from ..tools.corpus_store import corpus_fingerprint

SYSTEM_PROMPT = """\
You are the reconstruction agent in a genomic gap-filling system. You fill a gap between
two contigs using a masked language model, conditioned on reference sequences retrieved
by a peer agent.

Follow this protocol, in this order:

1. measure_budget(task_id)
   Reports how many of the 4096 token positions are free for retrieved context.

2. request_candidates(task_id, batch_size)
   Asks the retrieval agent for candidate row IDs. Choose batch_size from the free token
   budget: a reference gene costs roughly 50 to 500 tokens, so about one candidate per
   300 free tokens is sensible. Stay between 1 and 64.

3. admit_candidates(task_id, row_ids, free_tokens)
   Pass the row IDs you were just given and the exact free_tokens value this agent
   currently reports. It tokenizes each candidate and admits those that fit.

4. If free tokens remain and the pool is not exhausted, call
   extend_candidates(task_id, batch_size) and admit again. Repeat until the window is
   full or "pool_exhausted" is true. Do not loop more than a few times.

5. generate(task_id, gap_length, threshold)
   Use the gap_length given in the request, unchanged, and threshold 0.01.

Rules that the tools enforce, so do not fight them:
  - You cannot admit before measuring, or generate before admitting.
  - gap_length must be exactly the number in the request. Never adjust or recompute it.
  - free_tokens must be the value the agent currently reports; any other value is
    rejected and the correct one is returned to you.
  - Never invent a row ID or a nucleotide sequence. You are not given sequences and must
    not fabricate them.
  - If a tool returns an error, read its "remedy" field and follow it. If the same call
    fails twice, stop and say what happened.

When generate succeeds, reply with one short sentence. The caller reads the result from
the tool output, not from your text, so do not repeat any sequence.
"""

SKILLS = [
    skill(
        skill_id="gap_reconstruction",
        name="Retrieval-augmented gap reconstruction",
        description=(
            "Fill a gap between two contigs with GENA-LM, admitting only as much retrieved "
            "context as the 4096-token window can actually hold, with the context placed "
            "ahead of the sequence so truncation cannot silently remove it."
        ),
        tags=["genomics", "masked-language-model", "gena-lm", "rag"],
        examples=[
            "Fill the 870 base gap between contig1 and contig2 of AP012051.1.",
        ],
    ),
    skill(
        skill_id="token_budget_measurement",
        name="Token budget measurement",
        description=(
            "Report how many of the model's 4096 token positions remain free for retrieved "
            "context once the flanks and mask block are accounted for."
        ),
        tags=["genomics", "tokenization", "budget"],
        examples=["How much room is there for context on this gap?"],
    ),
]


def make_state(data: dict[str, Any], task_id: str) -> ReconTask:
    left = str(data.get("left") or "")
    right = str(data.get("right") or "")
    gap_length = int(data.get("gap_length") or 0)
    if not left or not right:
        raise ValueError("both 'left' and 'right' contig sequences are required")
    if gap_length <= 0:
        raise ValueError("'gap_length' must be a positive integer")

    query_text = str(data.get("query_text") or "")
    if not query_text:
        query_text = left[-config.QUERY_FLANK:] + right[: config.QUERY_FLANK]

    return ReconTask(
        task_id=task_id,
        context_id=str(data.get("context_id") or ""),
        gap_length=gap_length,
        left=left,
        right=right,
        organism=str(data.get("organism") or ""),
        query_text=query_text,
        # A stable id so the retrieval agent's ranked pool is reused across the
        # request/extend round trips rather than rebuilt each time.
        retrieval_request_id=f"pool-{task_id}",
    )


def build_prompt(state: ReconTask, data: dict[str, Any]) -> str:
    return (
        f"Fill one gap in a draft genome.\n\n"
        f"task_id: {state.task_id}\n"
        f"gap_id: {data.get('gap_id', '(unnamed)')}\n"
        f"gap_length: {state.gap_length}   <- pass this to generate unchanged\n"
        f"left contig: {len(state.left)} bases\n"
        f"right contig: {len(state.right)} bases\n"
        f"organism restriction: {state.organism or '(none)'}\n\n"
        f"The sequences themselves are held by the tools; you never handle them. Start "
        f"with measure_budget using the task_id above, then follow the protocol."
    )


def finish(state: ReconTask) -> dict[str, Any]:
    if state.result is None:
        return no_result_payload(
            state,
            {
                "context_tokens_admitted": state.context_tokens_admitted,
                "candidates_offered": len(state.offered_row_ids),
                "candidates_consumed": len(state.admitted_row_ids),
                "budget": state.budget,
            },
        )
    return dict(state.result)


def build_spec() -> AgentSpec:
    return AgentSpec(
        role=config.RECONSTRUCTION,
        name="GenomIO Reconstruction Agent",
        description=(
            "Fills genomic gaps with GENA-LM. Measures the token window, negotiates "
            "candidate batches with the retrieval agent, admits only what fits, and places "
            "the retrieved context ahead of the sequence so truncation cannot delete it."
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
        artifact_name="reconstruction_result",
        max_turns=config.MAX_TURNS,
        fingerprint=corpus_fingerprint,
        extra_card_fields={
            "x-generator": config.GENA_LM_MODEL,
            "x-max-length": config.MAX_LENGTH,
            "x-model-flank": config.MODEL_FLANK,
        },
    )


if __name__ == "__main__":
    sys.exit(launcher.run(build_spec, config.RECONSTRUCTION))
