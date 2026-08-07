"""The preconditions, exercised by calling tool handlers directly.

This is the file that decides whether the project's central claim holds. The old
pipeline asked for an ordering in a system prompt and nothing checked it. Here the
ordering is a precondition inside each handler, so these tests are the proof: they call
the tools out of order, with mutated arguments and past their budgets, and assert the
tools refuse before doing any work.

No LLM and no models are involved -- a fake tokenizer and a fake corpus store stand in,
so the arithmetic is exactly predictable.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from multiagent_system import config
from multiagent_system.runtime import errors
from multiagent_system.runtime.state import (
    REGISTRY,
    CoordTask,
    Phase,
    ReconTask,
    RetrievalSession,
)
from multiagent_system.tools import mlm
from multiagent_system.tools import reconstruction_tools as rt
from multiagent_system.tools import retrieval_tools as vt

pytestmark = pytest.mark.usefixtures("fake_store", "fake_organisms")

LEFT = "ACGT" * 300   # 1200 bases
RIGHT = "TGCA" * 300  # 1200 bases
GAP = 200


def run(coro):
    return asyncio.run(coro)


def payload(result: dict[str, Any]) -> dict[str, Any]:
    return errors.read_result(result)


def code(result: dict[str, Any]) -> str:
    return errors.error_code(result)


@pytest.fixture(autouse=True)
def clean_registry(fake_tokenizer):
    REGISTRY.clear()
    rt.MODELS.tokenizer = fake_tokenizer
    rt.MODELS.model = object()
    rt.MODELS.loaded = True
    yield
    REGISTRY.clear()
    rt.MODELS.tokenizer = None
    rt.MODELS.model = None
    rt.MODELS.loaded = False


@pytest.fixture
def recon_task():
    task = ReconTask(
        task_id="t-recon",
        context_id="ctx-1",
        gap_length=GAP,
        left=LEFT,
        right=RIGHT,
        query_text=LEFT[-100:] + RIGHT[:100],
    )
    REGISTRY.put(task)
    return task


@pytest.fixture
def retrieval_session():
    """The query lives on the session, put there from the A2A request.

    It is never a tool argument, so the model chooses when to search but not what to
    search for.
    """
    session = RetrievalSession(
        task_id="r-1", context_id="ctx-1", query_text="ACGT" * 20, organism=""
    )
    REGISTRY.put(session)
    return session


@pytest.fixture
def stub_peer(monkeypatch):
    """Stand in for the A2A hop to the retrieval agent."""
    calls: list[dict[str, Any]] = []
    pool = list(range(12))
    cursor = {"at": 0}

    async def fake_ask(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if request.get("fail"):
            return {"ok": False, "peer_task_state": "failed"}
        size = int(request.get("batch_size", 0))
        start = cursor["at"]
        rows = pool[start: start + size]
        cursor["at"] = start + len(rows)
        return {
            "ok": True,
            "row_ids": rows,
            "scores": [0.8] * len(rows),
            "path": "unrestricted",
            "pool_size": len(pool),
            "served_total": cursor["at"],
            "pool_exhausted": cursor["at"] >= len(pool),
            "restricted_set_size": 0,
        }

    monkeypatch.setattr(rt, "_ask_retrieval", fake_ask)
    return calls


# ====================================================================== reconstruction
class TestOrderingIsAConstraint:
    def test_admit_before_measure_is_refused_by_name(self, recon_task):
        result = run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [0, 1], "free_tokens": 100
        }))
        assert errors.is_error(result)
        assert code(result) == errors.BUDGET_NOT_MEASURED
        # The remedy names the exact next tool, emitted at the moment of violation.
        assert "measure_budget" in payload(result)["remedy"]
        assert recon_task.context_tokens_admitted == 0

    def test_generate_before_anything_is_refused(self, recon_task):
        result = run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP, "threshold": 0.01
        }))
        assert code(result) == errors.NOTHING_ADMITTED
        assert "admit_candidates" in payload(result)["remedy"]

    def test_request_candidates_before_measure_is_refused(self, recon_task, stub_peer):
        result = run(rt.request_candidates.handler({"task_id": "t-recon", "batch_size": 4}))
        assert code(result) == errors.BUDGET_NOT_MEASURED
        assert stub_peer == []  # the peer was never even contacted

    def test_unknown_task_is_refused(self):
        result = run(rt.measure_budget.handler({"task_id": "nope"}))
        assert code(result) == errors.UNKNOWN_TASK


class TestBudgetIsMeasuredNotGuessed:
    def test_measure_budget_reports_real_numbers(self, recon_task):
        data = payload(run(rt.measure_budget.handler({"task_id": "t-recon"})))
        assert data["base_tokens"] > 0
        assert data["free_tokens"] == recon_task.free_tokens_remaining
        assert data["max_length"] == config.MAX_LENGTH
        assert recon_task.phase is Phase.MEASURED
        # The seed comes from the measured ratio, not from gap_length/4.
        assert data["bases_per_token"] == pytest.approx(6.0, abs=0.5)
        assert data["seed_mask_count"] == mlm.seed_mask_count(GAP, data["bases_per_token"])
        assert data["seed_mask_count"] != int(GAP / 4)

    def test_model_flank_trims_the_contigs(self, recon_task, monkeypatch):
        monkeypatch.setattr(config, "MODEL_FLANK", 500)
        data = payload(run(rt.measure_budget.handler({"task_id": "t-recon"})))
        assert data["left_len"] == 1200 and data["left_used"] == 500
        assert data["right_len"] == 1200 and data["right_used"] == 500

    def test_zero_flank_disables_trimming(self, recon_task, monkeypatch):
        monkeypatch.setattr(config, "MODEL_FLANK", 0)
        data = payload(run(rt.measure_budget.handler({"task_id": "t-recon"})))
        assert data["left_used"] == data["left_len"] == 1200

    def test_a_wrong_free_tokens_value_is_rejected_and_corrected(self, recon_task):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        result = run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [0], "free_tokens": 999999
        }))
        assert code(result) == errors.BUDGET_MISMATCH
        # The true value comes back, so the model can only proceed by using it.
        assert payload(result)["free_tokens"] == recon_task.free_tokens_remaining
        assert recon_task.context_tokens_admitted == 0

    def test_admission_is_bounded_by_the_budget(self, recon_task):
        """Records cost 22, 32, 42, 52 and 62 tokens under the fake tokenizer.

        With 60 free, the first two fit and the rest cannot -- so admission is bounded
        by a measured number rather than by a fixed k that never knew the window size.
        """
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        recon_task.free_tokens_remaining = 60
        data = payload(run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": list(range(5)), "free_tokens": 60
        })))
        assert data["admitted_count"] == 2
        assert data["rejected_count"] == 3
        assert data["context_tokens_admitted"] == 54  # 22 + 32
        assert data["free_tokens_remaining"] == 6
        assert data["reject_reason_by_row"]

    def test_a_record_that_does_not_fit_does_not_stop_the_scan(self, recon_task):
        """Greedy first-fit: a later, shorter record is still admitted."""
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        recon_task.free_tokens_remaining = 45
        data = payload(run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [4, 0], "free_tokens": 45
        })))
        assert data["rejected"] == [4]   # 62 tokens, too big
        assert data["admitted"] == [0]   # 22 tokens, admitted after the rejection

    def test_duplicate_rows_are_counted_not_double_charged(self, recon_task):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        first = payload(run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [0, 1],
            "free_tokens": recon_task.free_tokens_remaining,
        })))
        second = payload(run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [0, 1],
            "free_tokens": recon_task.free_tokens_remaining,
        })))
        assert second["duplicates"] == 2
        assert second["context_tokens_admitted"] == first["context_tokens_admitted"]

    @pytest.mark.parametrize("bad", [[], "0,1", [None], ["x"], [999999]])
    def test_malformed_row_ids_are_rejected(self, recon_task, bad):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        result = run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": bad,
            "free_tokens": recon_task.free_tokens_remaining,
        }))
        assert code(result) == errors.BAD_ROW_IDS


class TestGenerateGuards:
    @pytest.fixture
    def admitted(self, recon_task, monkeypatch):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [0, 1],
            "free_tokens": recon_task.free_tokens_remaining,
        }))

        def fake_fill(context, left, right, gap_length, tokenizer, model, **kwargs):
            result = mlm.FillResult(
                sequence="ACGT" * (gap_length // 4),
                produced_length=gap_length,
                target_length=gap_length,
                attempts_used=2,
                seed_mask_count=kwargs.get("seed_masks", 0),
                final_mask_count=33,
                threshold=kwargs.get("threshold", 0.01),
                input_tokens=400,
                untruncated_tokens=400,
                converged=True,
            )
            # The context must be in front; assert it here so the ordering cannot
            # regress without this test noticing.
            assert mlm.build_model_input(context, left, right).startswith(context[:20])
            return result

        monkeypatch.setattr(mlm, "fill_to_length", fake_fill)
        return recon_task

    def test_a_mutated_gap_length_is_refused(self, admitted):
        """The planner_tools.py bug: an adjusted target silently replacing the real one."""
        result = run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP - 1, "threshold": 0.01
        }))
        assert code(result) == errors.GAP_LENGTH_MUTATED
        assert payload(result)["gap_length"] == GAP
        assert admitted.result is None

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, "x"])
    def test_an_out_of_range_threshold_is_refused(self, admitted, bad):
        result = run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP, "threshold": bad
        }))
        # "x" falls back to the default rather than erroring; the numeric ones are caught.
        if isinstance(bad, str):
            assert not errors.is_error(result)
        else:
            assert code(result) == errors.BAD_THRESHOLD

    def test_generate_succeeds_and_reports_counters(self, admitted):
        data = payload(run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP, "threshold": 0.01
        })))
        assert data["produced_length"] == GAP
        assert data["attempts_used"] == 2
        assert data["context_tokens_admitted"] > 0
        assert data["candidates_consumed"] == 2
        assert data["truncated"] is False
        assert "sequence" not in data  # kept out of the model's context
        assert admitted.phase is Phase.GENERATED
        assert admitted.result["sequence"]

    def test_generating_twice_returns_the_cached_result(self, admitted):
        first = payload(run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP, "threshold": 0.01
        })))
        second = payload(run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP, "threshold": 0.01
        })))
        assert second.get("cached") is True
        assert second["produced_length"] == first["produced_length"]

    def test_generate_is_refused_when_nothing_was_admitted(self, recon_task):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        recon_task.advance(Phase.ADMITTING)  # phase reached, but zero tokens admitted
        result = run(rt.generate.handler({
            "task_id": "t-recon", "gap_length": GAP, "threshold": 0.01
        }))
        assert code(result) == errors.EMPTY_CONTEXT


class TestBudgetsAndDeadlines:
    def test_the_call_budget_terminates_the_task(self, recon_task):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        limit = config.MAX_CALLS_PER_TOOL
        for _ in range(limit):
            run(rt.admit_candidates.handler({
                "task_id": "t-recon", "row_ids": [0],
                "free_tokens": recon_task.free_tokens_remaining,
            }))
        exhausted = run(rt.admit_candidates.handler({
            "task_id": "t-recon", "row_ids": [0],
            "free_tokens": recon_task.free_tokens_remaining,
        }))
        assert code(exhausted) == errors.CALL_BUDGET_EXHAUSTED
        assert recon_task.phase is Phase.TERMINATED
        # Everything afterwards short-circuits, including other tools.
        after = run(rt.report_state.handler({"task_id": "t-recon"}))
        assert code(after) == errors.TASK_TERMINATED

    def test_an_expired_deadline_terminates_the_task(self, recon_task):
        recon_task.deadline = recon_task.created_at - 1.0
        result = run(rt.measure_budget.handler({"task_id": "t-recon"}))
        assert code(result) == errors.DEADLINE_EXCEEDED
        assert recon_task.phase is Phase.TERMINATED


class TestNegotiationEdge:
    def test_batches_flow_and_the_pool_reports_exhaustion(self, recon_task, stub_peer):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        first = payload(run(rt.request_candidates.handler({
            "task_id": "t-recon", "batch_size": 5
        })))
        assert first["row_ids"] == [0, 1, 2, 3, 4]
        assert first["pool_exhausted"] is False

        run(rt.extend_candidates.handler({"task_id": "t-recon", "batch_size": 5}))
        last = payload(run(rt.extend_candidates.handler({
            "task_id": "t-recon", "batch_size": 5
        })))
        assert last["pool_exhausted"] is True
        assert recon_task.offered_row_ids == list(range(12))

    @pytest.mark.parametrize("bad", [0, -1, 1000, "many"])
    def test_a_nonsense_batch_size_never_reaches_the_peer(self, recon_task, stub_peer, bad):
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        result = run(rt.request_candidates.handler({"task_id": "t-recon", "batch_size": bad}))
        assert code(result) == errors.BAD_BATCH_SIZE
        assert stub_peer == []

    def test_a_peer_failure_is_reported_not_papered_over(self, recon_task, monkeypatch):
        async def dead_peer(_request):
            return {"ok": False, "peer_task_state": "failed", "detail": "boom"}

        monkeypatch.setattr(rt, "_ask_retrieval", dead_peer)
        run(rt.measure_budget.handler({"task_id": "t-recon"}))
        result = run(rt.request_candidates.handler({"task_id": "t-recon", "batch_size": 4}))
        assert code(result) == errors.PEER_UNREACHABLE
        assert "do not invent" in payload(result)["remedy"]


# ========================================================================== retrieval
class TestRetrievalPool:
    @pytest.fixture
    def stub_search(self, monkeypatch):
        def fake(query_text, k):
            rows = list(range(min(k, 12)))
            return rows, [round(0.9 - 0.01 * i, 4) for i in rows]

        monkeypatch.setattr(vt, "_search_unrestricted", fake)

    def test_extend_without_a_pool_is_refused(self, retrieval_session):
        result = run(vt.extend_context.handler({"request_id": "r-1", "batch_size": 3}))
        assert code(result) == errors.NO_POOL
        assert "retrieve_context" in payload(result)["remedy"]

    def test_a_pool_is_served_incrementally(self, retrieval_session, stub_search):
        first = payload(run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 4
        })))
        assert first["row_ids"] == [0, 1, 2, 3]
        assert first["path"] == "unrestricted"
        assert first["pool_exhausted"] is False
        # Only row IDs and scores cross the boundary.
        assert "sequence" not in first and "sequences" not in first
        assert all(isinstance(r, int) for r in first["row_ids"])

        second = payload(run(vt.extend_context.handler({"request_id": "r-1", "batch_size": 4})))
        assert second["row_ids"] == [4, 5, 6, 7]
        assert second["reused_pool"] is True

    def test_an_exhausted_pool_is_a_normal_signal_not_an_error(
        self, retrieval_session, stub_search
    ):
        run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 12
        }))
        result = run(vt.extend_context.handler({"request_id": "r-1", "batch_size": 4}))
        assert not errors.is_error(result)
        assert payload(result)["row_ids"] == []
        assert payload(result)["pool_exhausted"] is True

    def test_reopening_a_pool_with_a_different_query_is_refused(
        self, retrieval_session, stub_search
    ):
        """A re-opened pool whose query changed would silently mix two rankings."""
        run(vt.retrieve_context.handler({"request_id": "r-1", "batch_size": 2}))
        retrieval_session.query_text = "TTTT" * 20  # as if the caller re-opened it
        result = run(vt.retrieve_context.handler({"request_id": "r-1", "batch_size": 2}))
        assert code(result) == errors.QUERY_CHANGED

    def test_a_request_carrying_no_query_is_refused(self, retrieval_session):
        """And a query passed as a tool argument is ignored, because it is not one."""
        retrieval_session.query_text = "  "
        result = run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 2, "query_text": "ACGT" * 20
        }))
        assert code(result) == errors.EMPTY_QUERY

    def test_an_unknown_organism_lists_the_known_ones(self, retrieval_session, stub_search):
        retrieval_session.organism = "Tyrannosaurus"
        result = run(vt.retrieve_context.handler({"request_id": "r-1", "batch_size": 2}))
        assert code(result) == errors.UNKNOWN_ORGANISM
        assert "Alpha" in payload(result)["known"]

    def test_restricted_search_stays_inside_the_organism_block(
        self, retrieval_session, fake_organisms, monkeypatch
    ):
        lo, hi = fake_organisms.block("Beta")

        def fake_restricted(query_text, block, k):
            return list(range(block[0], block[1])), [0.7] * (block[1] - block[0]), (
                vt.PATH_RESTRICTED_OVERFETCH
            )

        monkeypatch.setattr(vt, "_search_restricted", fake_restricted)
        retrieval_session.organism = "Beta"
        data = payload(run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 4,
        })))
        assert data["path"] == vt.PATH_RESTRICTED_OVERFETCH
        assert data["organism"] == "Beta"
        assert data["restricted_set_size"] == hi - lo
        assert all(lo <= r < hi for r in data["row_ids"])

    def test_an_empty_restricted_pool_is_distinguishable_from_a_failure(
        self, retrieval_session, monkeypatch
    ):
        monkeypatch.setattr(
            vt, "_search_restricted", lambda q, b, k: ([], [], vt.PATH_RESTRICTED_RANGE)
        )
        retrieval_session.organism = "Beta"
        result = run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 4,
        }))
        assert code(result) == errors.RESTRICTED_POOL_EMPTY
        assert payload(result)["restricted_set_size"] > 0

    def test_describe_rows_returns_metadata_but_never_nucleotides(
        self, retrieval_session, stub_search
    ):
        run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 3
        }))
        data = payload(run(vt.describe_rows.handler({"request_id": "r-1", "row_ids": [0, 1]})))
        assert len(data["records"]) == 2
        for record in data["records"]:
            assert "sequence" not in record
            assert record["organism"] and record["length"] > 0

    def test_describe_rows_ignores_rows_that_were_never_served(
        self, retrieval_session, stub_search
    ):
        run(vt.retrieve_context.handler({
            "request_id": "r-1", "batch_size": 2
        }))
        data = payload(run(vt.describe_rows.handler({"request_id": "r-1", "row_ids": [9, 10]})))
        assert data["records"] == []


# ======================================================================== coordinator
class TestCoordinatorGate:
    @pytest.fixture
    def coord_task(self):
        task = CoordTask(
            task_id="t-coord",
            context_id="ctx-1",
            gap_id="AP012051.1_gap1",
            gap_length=GAP,
            left=LEFT,
            right=RIGHT,
        )
        REGISTRY.put(task)
        return task

    def test_evaluate_before_delegating_is_refused(self, coord_task):
        from multiagent_system.tools import coordinator_tools as ct

        result = run(ct.evaluate_result.handler({"task_id": "t-coord"}))
        assert code(result) == errors.NO_RESULT

    def test_the_gate_rejects_a_perfect_sequence_with_no_admitted_context(self, coord_task):
        from multiagent_system.tools import coordinator_tools as ct

        coord_task.advance(Phase.DELEGATED)
        coord_task.peer_result = {
            "sequence": "ACGT" * (GAP // 4),
            "context_tokens_admitted": 0,
        }
        data = payload(run(ct.evaluate_result.handler({"task_id": "t-coord"})))
        assert data["accepted"] is False
        assert data["verdict"] == "REJECTED:NO_CONTEXT"
        assert "truncation" in data["diagnosis"]

    def test_the_gate_accepts_a_real_result(self, coord_task):
        from multiagent_system.tools import coordinator_tools as ct

        coord_task.advance(Phase.DELEGATED)
        coord_task.peer_result = {
            "sequence": "ACGT" * (GAP // 4),
            "context_tokens_admitted": 512,
            "candidates_offered": 6,
            "candidates_consumed": 2,
            "attempts_used": 3,
        }
        data = payload(run(ct.evaluate_result.handler({"task_id": "t-coord"})))
        assert data["accepted"] is True
        assert data["verdict"] == "ACCEPTED"

        trace = payload(run(ct.emit_trace.handler({"task_id": "t-coord"})))
        assert trace["context_tokens_admitted"] == 512
        assert trace["candidates_offered"] == 6
        assert trace["verdict"] == "ACCEPTED"
        assert trace["gap_length"] == GAP

    def test_the_query_is_flanks_not_whole_contigs(self):
        from multiagent_system.tools import coordinator_tools as ct

        left, right = "A" * 100000, "C" * 100000
        query = ct.build_query(left, right)
        assert len(query) == 2 * config.QUERY_FLANK
        assert set(query) == {"A", "C"}  # both flanks present, unlike the original
