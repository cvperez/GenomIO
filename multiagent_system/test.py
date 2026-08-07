#!/usr/bin/env python3
"""End-to-end acceptance test for the GenomIO multi-agent system.

The successor to ``gap-filler-agents/test.py``, which printed a great deal and asserted
nothing. This one asserts, exits non-zero on failure, and works unchanged against a local
three-process fleet or three SLURM compute nodes.

    python3 multiagent_system/test.py --mode local
    python3 multiagent_system/test.py --mode slurm
    python3 multiagent_system/test.py --mode slurm --endpoints-file .run/endpoints.json

The assertion that matters is #3, ``context_tokens_admitted > 0``. The old pipeline could
not have passed it: it appended retrieved context after the sequence and then truncated
from the end, so on any long contig pair the model never saw the retrieved evidence --
and the run was recorded as RAG-enabled regardless.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

if __package__ in (None, ""):  # allow `python3 multiagent_system/test.py`
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multiagent_system import config, dataset, fleet, gate  # noqa: E402
from multiagent_system.a2a import types as a2a  # noqa: E402
from multiagent_system.a2a.client import SyncA2AClient  # noqa: E402

ACCESSION = "AP012051.1"
GAP_ID = "AP012051.1_gap1"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Checks:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed.append(name)
            print(f"  {GREEN}PASS{RESET}  {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
        else:
            self.failed.append((name, detail))
            print(f"  {RED}FAIL{RESET}  {name}" + (f"\n        {detail}" if detail else ""))
        return ok

    @property
    def ok(self) -> bool:
        return not self.failed


# --------------------------------------------------------------------------- preflight
def preflight(accession: str, gap_id: str) -> dict[str, Any]:
    """Validate the data before spending a minute launching models.

    The old harness took ``contigs[0]`` and ``contigs[1]``. The FASTA is sorted
    longest-first, so those were contig31 (190 kb) and contig22 (167 kb) -- two contigs
    with no gap between them -- while the 870 bp target came from gap1. Here the flanks
    are matched by genome coordinate.
    """
    case = dataset.load_gap_case(accession, gap_id)
    if gap_id == GAP_ID:
        assert case["gap_length"] == 870, case["gap_length"]
        assert case["left_id"] == "AP012051.1_contig1", case["left_id"]
        assert case["right_id"] == "AP012051.1_contig2", case["right_id"]
        assert case["left_length"] == 5951 and case["right_length"] == 13996
    print(
        f"{DIM}  data: {case['gap_id']} length={case['gap_length']} "
        f"left={case['left_id']}({case['left_length']}bp) "
        f"right={case['right_id']}({case['right_length']}bp){RESET}"
    )
    return case


# --------------------------------------------------------------------------- assertions
def assert_run(case: dict[str, Any], task: dict[str, Any], organism: str) -> Checks:
    checks = Checks()
    state = a2a.task_state(task)
    data = a2a.artifact_data(task)
    trace = data.get("trace") or {}
    sequence = str(data.get("sequence") or "")

    # 1 -- the run completed at all
    checks.check(
        "1. coordinator task completed",
        state is a2a.TaskState.COMPLETED,
        f"state={state.value}"
        + (f" error={data.get('error')}" if data.get("error") else "")
        + (f" phase={data.get('phase')}" if data.get("phase") else ""),
    )
    if state is not a2a.TaskState.COMPLETED:
        print(f"\n{DIM}  coordinator trace: {json.dumps(trace, indent=2)[:2000]}{RESET}")
        return checks

    # 2 -- the acceptance gate passed
    verdict = str(trace.get("verdict", ""))
    checks.check(
        "2. acceptance gate verdict is ACCEPTED",
        verdict == gate.VERDICT_ACCEPTED,
        f"verdict={verdict} -- {gate.explain(list(trace.get('reasons') or []))}",
    )

    # 3 -- THE assertion: retrieved context actually reached the model
    admitted = int(trace.get("context_tokens_admitted", 0) or 0)
    checks.check(
        "3. context tokens admitted > 0",
        admitted > 0,
        f"{admitted} tokens of retrieved context reached the model "
        f"(free budget was {trace.get('free_tokens')})",
    )

    # 4 -- candidates were negotiated, not fixed
    offered = int(trace.get("candidates_offered", 0) or 0)
    consumed = int(trace.get("candidates_consumed", 0) or 0)
    checks.check(
        "4. 1 <= candidates consumed <= offered",
        1 <= consumed <= offered,
        f"offered={offered} consumed={consumed}",
    )

    # 5 -- the produced length is plausible
    produced = len(sequence)
    tolerance = gate.tolerance_bases(case["gap_length"], config.TOLERANCE)
    checks.check(
        "5. produced length within tolerance",
        abs(produced - case["gap_length"]) <= tolerance,
        f"{produced} vs target {case['gap_length']} (tolerance +/-{tolerance})",
    )

    # 6 -- it is DNA
    bad = sorted(set(sequence) - gate.VALID_BASES)
    checks.check(
        "6. alphabet is a subset of ACGTN",
        not bad and int(trace.get("alphabet_violations", 0) or 0) == 0,
        f"unexpected characters: {bad}" if bad else "",
    )

    # 7 -- the window was never overrun, so nothing was silently deleted
    checks.check(
        "7. model input was not truncated",
        trace.get("truncated") is False,
        f"input_tokens={trace.get('input_tokens')} of {config.MAX_LENGTH}",
    )

    # 8 -- gap_length survived every hop untouched
    budget_gap = ((data.get("budget") or {}) or {}).get("gap_length")
    lengths = {
        "request": case["gap_length"],
        "coordinator payload": data.get("gap_length"),
        "trace": trace.get("gap_length"),
    }
    if budget_gap is not None:
        lengths["budget report"] = budget_gap
    checks.check(
        "8. gap_length unchanged at every hop",
        len(set(v for v in lengths.values() if v is not None)) == 1,
        ", ".join(f"{k}={v}" for k, v in lengths.items()),
    )

    # 9 -- retrieval never handed nucleotides across an agent boundary
    leaks = find_sequence_leaks(trace)
    checks.check(
        "9. retrieval returned row IDs, never nucleotides",
        not leaks,
        f"leaked keys: {leaks}" if leaks else
        f"admitted rows {list(trace.get('admitted_row_ids') or [])[:8]} are all integers",
    )

    # 10 -- the right retrieval mode was taken
    expected_mode = "restricted" if organism else "unrestricted"
    mode_ok = trace.get("mode") == expected_mode
    detail = f"mode={trace.get('mode')} expected={expected_mode}"
    if organism:
        mode_ok = mode_ok and int(trace.get("restricted_set_size", 0) or 0) >= 0
    checks.check("10. retrieval mode is correct", mode_ok, detail)

    # 11 -- no model reached a tool outside its own agent
    foreign = find_foreign_tool_calls(trace, data.get("llm_tool_calls") or [])
    total_calls = (
        len(trace.get("reconstruction_tool_calls") or [])
        + len(trace.get("retrieval_tool_calls") or [])
        + len(data.get("llm_tool_calls") or [])
    )
    checks.check(
        "11. every tool call was one of the agent's own",
        not foreign,
        f"foreign calls: {sorted(set(foreign))}" if foreign
        else f"{total_calls} calls across three agents, all mcp__",
    )

    # 12 -- one ranked pool served the whole negotiation
    reused, why = check_pool_reuse(trace)
    checks.check("12. retrieval pool was reused, not rebuilt", reused, why)

    return checks


def find_foreign_tool_calls(
    trace: dict[str, Any], coordinator_calls: list[dict[str, Any]]
) -> list[str]:
    """Any tool call made by any of the three models that is not one of their own.

    A stray built-in never fails a run -- it just means the model had a route it should
    not have had. The first smoke run reached ``ToolSearch`` this way.
    """
    foreign: list[str] = []
    for calls in (
        trace.get("reconstruction_tool_calls") or [],
        trace.get("retrieval_tool_calls") or [],
        coordinator_calls or [],
    ):
        for call in calls:
            name = str(call.get("name", ""))
            if not name.startswith("mcp__"):
                foreign.append(name)
    return foreign


def check_pool_reuse(trace: dict[str, Any]) -> tuple[bool, str]:
    """Did one ranked pool serve the whole negotiation, or was it rebuilt each round?

    A rebuilt pool still returns plausible row IDs, so this is invisible without
    inspecting the rounds: a real pool keeps one ``pool_size`` while ``served_total``
    climbs.
    """
    rounds = trace.get("retrieval_rounds") or []
    if len(rounds) < 2:
        return True, f"only {len(rounds)} retrieval round(s); nothing to reuse"
    sizes = {r.get("pool_size") for r in rounds}
    totals = [r.get("served_total", 0) for r in rounds]
    if len(sizes) != 1:
        return False, f"pool_size changed across rounds: {sorted(sizes)} -- pool was rebuilt"
    if totals != sorted(totals) or len(set(totals)) != len(totals):
        return False, f"served_total did not advance monotonically: {totals}"
    return True, (
        f"{len(rounds)} rounds served {totals[-1]} rows from one pool of {sizes.pop()}"
    )


def find_sequence_leaks(trace: dict[str, Any]) -> list[str]:
    """Walk the trace for any nucleotide payload where an identifier belongs."""
    leaks: list[str] = []
    forbidden = {"sequences", "candidate_sequences", "rag_matches", "context_blocks"}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in forbidden:
                    leaks.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node[:20]:
                walk(item, path)

    walk(trace, "trace")
    rows = trace.get("admitted_row_ids") or []
    if not all(isinstance(r, int) for r in rows):
        leaks.append("trace.admitted_row_ids contains non-integers")
    return leaks


# --------------------------------------------------------------------------- the run
def submit(endpoint: str, case: dict[str, Any], organism: str) -> dict[str, Any]:
    client = SyncA2AClient(endpoint, timeout=config.CLIENT_TIMEOUT_S)
    message = a2a.build_message(
        role="user",
        parts=[
            a2a.data_part(
                {
                    "accession": case["accession"],
                    "gap_id": case["gap_id"],
                    "gap_length": case["gap_length"],
                    "left": case["left"],
                    "right": case["right"],
                    "organism": organism,
                }
            ),
            a2a.text_part(
                f"Fill the {case['gap_length']} base gap {case['gap_id']} between "
                f"{case['left_id']} and {case['right_id']}."
            ),
        ],
    )
    return client.send(message)


def report(case, task, trace, elapsed, checks) -> None:
    data = a2a.artifact_data(task)
    sequence = str(data.get("sequence") or "")
    print(f"\n{'-' * 74}\n  run summary\n{'-' * 74}")
    rows = [
        ("gap", f"{case['gap_id']}  target {case['gap_length']} bases"),
        ("verdict", trace.get("verdict", "-")),
        ("mode", trace.get("mode", "-")),
        ("query flank", f"{trace.get('query_flank')} bases either side"),
        ("model flank", f"{trace.get('model_flank')} bases either side"),
        (
            "token budget",
            f"base {trace.get('base_tokens')} + context {trace.get('context_tokens_admitted')}"
            f" -> input {trace.get('input_tokens')} of {config.MAX_LENGTH}"
            f"  (free was {trace.get('free_tokens')})",
        ),
        (
            "candidates",
            f"{trace.get('candidates_consumed')} admitted of "
            f"{trace.get('candidates_offered')} offered",
        ),
        ("truncated", str(trace.get("truncated"))),
        (
            "generation",
            f"{trace.get('generation_attempts')} attempts, seed {trace.get('seed_mask_count')}"
            f" -> final {trace.get('final_mask_count')} masks, "
            f"converged={trace.get('converged')}",
        ),
        ("produced", f"{trace.get('produced_length')} bases, "
                     f"{trace.get('alphabet_violations')} alphabet violations"),
        ("elapsed", f"{elapsed:.1f}s"),
    ]
    for label, value in rows:
        print(f"  {label:<14} {value}")
    if sequence:
        print(f"  {'sequence':<14} {sequence[:60]}...")
        print(f"  {'true gap':<14} {case['true_sequence'][:60]}...")
    print(
        f"\n  {GREEN if checks.ok else RED}{len(checks.passed)} passed, "
        f"{len(checks.failed)} failed{RESET}"
    )


def save(case, task, trace, elapsed, checks, endpoints) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = os.path.join(config.RESULTS_DIR, f"{stamp}_{case['gap_id']}.json")
    data = a2a.artifact_data(task)
    with open(path, "w") as handle:
        json.dump(
            {
                "timestamp": a2a.utcnow(),
                "accession": case["accession"],
                "gap_id": case["gap_id"],
                "gap_length": case["gap_length"],
                "left_id": case["left_id"],
                "right_id": case["right_id"],
                "endpoints": endpoints,
                "config": {
                    "query_flank": config.QUERY_FLANK,
                    "model_flank": config.MODEL_FLANK,
                    "max_length": config.MAX_LENGTH,
                    "tolerance": config.TOLERANCE,
                    "agent_model": config.AGENT_MODEL,
                },
                "predicted_sequence": data.get("sequence", ""),
                "true_sequence": case["true_sequence"],
                "trace": trace,
                "elapsed_s": round(elapsed, 2),
                "checks_passed": checks.passed,
                "checks_failed": [{"name": n, "detail": d} for n, d in checks.failed],
                "accepted": checks.ok,
            },
            handle,
            indent=2,
        )
    return path


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "slurm"), default="local")
    parser.add_argument("--endpoints-file", default="", help="slurm mode: agent URLs")
    parser.add_argument("--endpoints", default="", help="role=url,role=url")
    parser.add_argument("--accession", default=ACCESSION)
    parser.add_argument("--gap", default=GAP_ID)
    parser.add_argument("--organism", default="", help="restrict retrieval to one organism")
    parser.add_argument("--base-port", type=int, default=config.BASE_PORT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ready-timeout", type=float, default=420.0)
    parser.add_argument("--keep-alive", action="store_true", help="local: leave agents up")
    args = parser.parse_args(argv)

    print(f"\n{'=' * 74}\n  GenomIO multi-agent system -- end to end ({args.mode})\n{'=' * 74}")

    print("\npreflight")
    try:
        case = preflight(args.accession, args.gap)
    except Exception as exc:
        print(f"  {RED}FAIL{RESET}  data preflight: {type(exc).__name__}: {exc}")
        return 1
    print(f"  {GREEN}PASS{RESET}  data and flank selection")

    local: fleet.LocalFleet | None = None
    if args.endpoints:
        endpoints = fleet.parse_endpoints(args.endpoints)
    elif args.mode == "slurm":
        path = args.endpoints_file or fleet.endpoints_file()
        if not os.path.exists(path):
            print(f"  {RED}FAIL{RESET}  no endpoints at {path}; run deploy.sh first")
            return 1
        endpoints = fleet.load_endpoints(path)
    else:
        print(f"\nlaunching a local fleet on ports {args.base_port}-{args.base_port + 2}")
        local = fleet.LocalFleet(base_port=args.base_port, threads=args.threads)
        endpoints = local.start()

    missing = [r for r in fleet.MODULES if r not in endpoints]
    if missing:
        print(f"  {RED}FAIL{RESET}  no endpoint for {missing}")
        return 1
    for role, url in endpoints.items():
        print(f"  {role:<15} {url}")

    exit_code = 1
    try:
        print(f"\nwaiting for models (up to {args.ready_timeout:.0f}s)")
        started = time.monotonic()
        ready, status = fleet.wait_until_ready(
            endpoints,
            timeout=args.ready_timeout,
            fleet=local,
            on_tick=lambda s: print(
                f"  {DIM}{time.monotonic() - started:5.0f}s  "
                + "  ".join(f"{r}={'up' if v else '...'}" for r, v in s.items())
                + RESET
            ),
        )
        if not ready:
            print(f"  {RED}FAIL{RESET}  agents not ready: {status}")
            if local:
                for role in fleet.MODULES:
                    print(f"\n{DIM}--- {role} log ---\n{local.tail(role)}{RESET}")
            return 1
        print(f"  {GREEN}PASS{RESET}  all three ready in {time.monotonic() - started:.0f}s")
        for role, health in status.items():
            print(f"  {DIM}  {role:<15} {health.get('warm', {})}{RESET}")

        print(f"\nsubmitting {case['gap_id']} to the coordinator")
        run_started = time.monotonic()
        try:
            task = submit(endpoints[config.COORDINATOR], case, args.organism)
        except Exception as exc:
            print(f"  {RED}FAIL{RESET}  coordinator call: {type(exc).__name__}: {exc}")
            if local:
                for role in fleet.MODULES:
                    print(f"\n{DIM}--- {role} log ---\n{local.tail(role)}{RESET}")
            return 1
        elapsed = time.monotonic() - run_started
        print(f"  returned in {elapsed:.1f}s")

        print("\nassertions")
        checks = assert_run(case, task, args.organism)
        trace = (a2a.artifact_data(task) or {}).get("trace") or {}
        report(case, task, trace, elapsed, checks)
        path = save(case, task, trace, elapsed, checks, endpoints)
        print(f"  {DIM}written to {path}{RESET}")
        exit_code = 0 if checks.ok else 1
    finally:
        if local and not args.keep_alive:
            print("\nstopping the local fleet")
            local.stop()
        elif local:
            print(f"\n{DIM}fleet left running (--keep-alive){RESET}")

    print(f"\n{GREEN + 'RESULT: PASS' if exit_code == 0 else RED + 'RESULT: FAIL'}{RESET}\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
