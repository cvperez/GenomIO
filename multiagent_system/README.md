# GenomIO Multi-Agent System

Three A2A agents that fill a gap in a draft genome using DNABERT-S retrieval and
GENA-LM generation. Each agent is driven by a language model through the Claude Agent
SDK; every capability it has is an MCP tool.

It replaces the LangChain pipeline in `gap-filler-agents/`, which could not run and would
not have been trustworthy if it had. What it adds is one number: **how many tokens of
retrieved context actually reached the generator**, gated so nothing is returned when
that number is zero.

```
                       Coordinator  (no models, ~2 s startup)
                            |  A2A: message/send
                            v
                     Reconstruction  (GENA-LM, ~3 s startup)
                            |  A2A: message/send   <-- the negotiation edge
                            v
                       Retrieval  (DNABERT-S + FAISS, ~30 s startup)
```

---

## Quick start

```bash
# once: an isolated virtualenv that inherits torch/faiss/transformers from the system
python3 -m venv --system-site-packages multiagent_system/.venv
multiagent_system/.venv/bin/pip install claude-agent-sdk

# unit and contract tests -- no LLM, no models, ~4 s
multiagent_system/.venv/bin/python3 -m pytest multiagent_system/tests \
    -c multiagent_system/pytest.ini

# end to end on one host: launches three agents, fills gap1, asserts, tears down
multiagent_system/.venv/bin/python3 multiagent_system/test.py --mode local

# end to end across three ARES nodes
salloc -N 3 -p compute -t 02:00:00 --no-shell -J genomio-mas    # note the JOBID
./multiagent_system/deploy.sh <JOBID>
multiagent_system/.venv/bin/python3 multiagent_system/test.py --mode slurm
scancel <JOBID>
```

The venv is required: `claude_agent_sdk` and `mcp` live only inside it, and installing
them there rather than into the user site-packages is what keeps the existing
torch/transformers/faiss stack untouched. Rollback is `rm -rf multiagent_system/.venv`.

Authentication is the `claude` CLI subscription. `ANTHROPIC_API_KEY` is unset by both
the launcher and `deploy.sh`, so a stray key in a shell cannot silently switch billing to
API credits.

---

## The three agents

### Coordinator — `agents/coordinator.py`, port 8100

Loads no models and wraps no RAG code. Slices the retrieval query flanks, checks its
peers are healthy and reading the same corpus, delegates, and applies the acceptance
gate.

| Tool | Purpose |
|---|---|
| `prepare_query` | Slice the query, verify peers, choose restricted vs unrestricted |
| `delegate_reconstruction` | Hand the task to the reconstruction agent over A2A |
| `evaluate_result` | Apply the acceptance gate |
| `emit_trace` | Produce the run record |

### Retrieval — `agents/retrieval.py`, port 8101

Owns DNABERT-S and the 43,575-vector FAISS index, loaded once at startup.

| Tool | Purpose |
|---|---|
| `retrieve_context` | Open a ranked candidate pool, serve the first batch |
| `extend_context` | Serve the next batch — no re-embedding |
| `describe_rows` | Organism and protein metadata for rows already served |

**It returns row IDs and cosine scores, never nucleotides.**

### Reconstruction — `agents/reconstruction.py`, port 8102

Owns GENA-LM, so it is the only component that knows the token budget.

| Tool | Purpose |
|---|---|
| `measure_budget` | Tokenize the base input; report free positions of 4,096 |
| `request_candidates` | Ask retrieval for a batch sized to that budget |
| `extend_candidates` | Ask for more when room remains |
| `admit_candidates` | Resolve row IDs, tokenize each, admit what fits |
| `generate` | Run the masked fill |
| `report_state` | Read-only diagnostic |

---

## The three invariants

These are the reason the system is worth building, and each is enforced by code rather
than requested in a prompt.

### 1. The answer is read from state, never from the model's text

Tool handlers write to a per-task state object; the A2A reply is serialised from that
state. The model's prose becomes an advisory `TextPart` that nothing reads.

This is not theoretical. During bring-up, one retrieval run ended with the model
emitting `API Error: Sonnet 4.5 can't help with this` **after** its tool had already run.
The task still completed correctly, with the right row IDs, because the answer never
depended on what the model said.

### 2. Ordering is a data dependency, not an instruction

The old pipeline's system prompt said "You MUST use these tools in the correct order" and
nothing checked it — `AgentExecutor` imposed no ordering, no iteration cap and no failure
handling. Here every handler runs a precondition prologue first:

```json
{"error":"PRECONDITION_FAILED","code":"BUDGET_NOT_MEASURED",
 "phase":"CREATED","required_phase":"MEASURED",
 "remedy":"Call mcp__reconstruction_agent__measure_budget with this task_id first; it
           reports how many tokens are free for retrieved context."}
```

The `remedy` is the only steering that matters, because it is emitted at the moment of
violation by code that knows the real state.

| Code | Fires when |
|---|---|
| `BUDGET_NOT_MEASURED` | admitting before measuring |
| `BUDGET_MISMATCH` | `free_tokens` is not the agent's current value; the true one is returned |
| `GAP_LENGTH_MUTATED` | a target length other than the one the task carries |
| `NOTHING_ADMITTED` / `EMPTY_CONTEXT` | generating with no context |
| `NO_POOL` / `QUERY_CHANGED` | extending a pool that does not exist, or one re-opened with a different query |
| `CALL_BUDGET_EXHAUSTED` / `DEADLINE_EXCEEDED` | a looping model; both terminate the task |

### 3. Sequences are never tool arguments and never enter a prompt

Every tool is keyed by `task_id` or `request_id`. The contigs, the query and the
retrieved sequences live in agent state. The old planner interpolated the nucleotide
string into the prompt and made the model re-emit it as a tool argument, with no
checksum and no length assertion.

Removing the query from the prompt also fixed a real failure and cut latency from 46 s to
13 s per retrieval call.

---

## Reading a trace

Every run writes `results/<timestamp>_<gap_id>.json`. The fields that matter:

| Field | Read it as |
|---|---|
| `context_tokens_admitted` | **The headline.** Tokens of retrieved context that reached the model. Zero means this run was not retrieval-augmented, whatever it is labelled. |
| `free_tokens` | How much room there was, measured before anything was requested |
| `truncated` | `true` means the input overran the 4,096-token window, so something was cut |
| `candidates_offered` / `candidates_consumed` | The negotiation: how many were served and how many fitted |
| `verdict` | `ACCEPTED`, or `REJECTED:` plus reasons |
| `generation_attempts`, `seed_mask_count`, `final_mask_count` | The length search, which the old code printed and discarded |

**The verdicts are deliberately distinguishable:**

- `REJECTED:NO_CONTEXT` — *this system* is broken. Retrieval never reached the model.
- `REJECTED:LENGTH` or `REJECTED:ALPHABET` — the underlying masked language model
  produced something poor. That is a known limitation this project does not fix.

---

## What A2A support is and is not

Implemented: agent cards at `/.well-known/agent-card.json` and the
`/.well-known/agent-configuration` alias; `GET /health`; JSON-RPC 2.0 `message/send`,
`tasks/get`, `tasks/cancel` at `POST /`; `GET /tasks/{id}`; all nine `TaskState` values;
`contextId` grouping; artifacts carrying a `DataPart` and a `TextPart`; terminal-task
immutability.

Not implemented, and answered with `-32601` rather than silently accepted:
`message/stream` and SSE, push notifications and `tasks/pushNotificationConfig/*`,
`tasks/resubscribe`, `FilePart`, authentication. `capabilities.streaming` and
`capabilities.pushNotifications` are both `false` in the card.

`message/send` blocks until the task reaches a terminal state. There is one long call per
run and both callers want the answer, so polling would add machinery for no benefit.

**There is no authentication.** These ports must not be exposed beyond the cluster's
private network. On ARES the compute nodes sit on `172.25.x.x` and are unreachable from
outside the allocation, which is the only thing protecting them.

The HTTP layer is the Python standard library — `fastapi` and `uvicorn` are not installed
here and nothing needs them. A useful side effect is that `a2a/` and `tools/` import under
the system interpreter, so the protocol tests run without the SDK.

---

## Configuration

Everything in `config.py` is environment-overridable.

| Variable | Default | What it does |
|---|---|---|
| `GENOMIO_QUERY_FLANK` | 900 | Bases either side used to build the *retrieval* query |
| `GENOMIO_MODEL_FLANK` | 2000 | Bases either side kept in the *model* input; `0` disables trimming |
| `GENOMIO_MAX_LENGTH` | 4096 | GENA-LM's context window |
| `GENOMIO_TOLERANCE` | 0.05 | Length tolerance for the gate |
| `GENOMIO_AGENT_MODEL` | `claude-sonnet-4-5` | The model driving all three agents |
| `GENOMIO_MAX_TURNS` | 24 | Hard cap per agent conversation |
| `GENOMIO_MAX_CALLS_PER_TOOL` | 12 | Per-tool call budget; exceeding it terminates the task |
| `GENOMIO_TASK_DEADLINE_S` | 900 | Wall-clock budget per task |
| `GENOMIO_SEED` | 20260807 | `torch.manual_seed`, for reproducibility |

`MODEL_FLANK` is the one worth understanding. Measured on gap1 (contigs of 5,951 and
13,996 bases):

| Setting | Base input | Free for context |
|---|---|---|
| no trimming | 3,305 tokens | ~583 |
| **2000 (default)** | **682 tokens** | **~3,236** |
| the old harness's contig pair, untrimmed | 58,776 tokens | none — 14x over the window |

---

## Layout

```
multiagent_system/
├── agents/       coordinator.py  retrieval.py  reconstruction.py
├── tools/        *_tools.py (MCP tools)  mlm.py (generation)
│                 corpus_store.py  organism_index.py
├── a2a/          types.py  store.py  card.py  server.py  client.py
├── runtime/      launcher.py  agent.py  state.py  blocking.py  errors.py
├── tests/        contract, protocol, gate, flank-selection and naming tests
├── config.py  gate.py  dataset.py  fleet.py
├── test.py       end-to-end acceptance test
├── deploy.sh     SLURM fleet launch
└── RESULTS.md    measured outcomes
```

Nothing under `src/` is modified. `tools/mlm.py` is a new module rather than an edit to
`src/core/gap_filler_rag.py`, so the original pipeline stays runnable.

---

## Operational notes

**Process structure.** asyncio owns the main thread; the HTTP server runs on a daemon
thread and hands work back with `run_coroutine_threadsafe`. The Claude Agent SDK binds an
anyio task group and a CLI subprocess to whichever loop created them, so building it on a
worker produces "cancel scope in a different task". All torch and faiss work goes through
`runtime/blocking.py`, which combines `asyncio.to_thread` with one process-global lock —
`src/rag/dnabert_s.py` keeps its tokenizer and model in unguarded module globals.

**Health gating.** `/health` reports `models_loaded` only after warm-up returns. Sending
work into a 30-second DNABERT-S load looks exactly like a hang.

**Tool-id coupling.** A tool is `mcp__<server_key>__<tool_name>`. If `allowed_tools` and
the registered names disagree, the SDK does not raise — the model is told it has no tools
and starts talking. `LLMAgent.__init__` checks this at startup and `tests/test_naming.py`
checks it statically.

**Restricting the built-in tools.** `allowed_tools` alone let `ToolSearch` through;
setting `tools` as well is what confines the model to this agent's own capabilities.
`tests/test_agent_runtime.py` pins this, and `test.py` assertion 11 re-checks it on a
live run.

**Organism restriction.** Each organism occupies one contiguous block of corpus rows, so
restriction over-fetches and post-filters. A sparse `faiss.IDSelectorBatch` on this HNSW
index returns all `-1` — greedy graph traversal cannot reach enough accepted nodes, and
raising `efSearch` does not help. `IDSelectorRange` is a tagged fallback whose output is
checked for that under-fill signature.

**On ARES.** `ssh` to a compute node is refused by `pam_slurm_adopt` without an
allocation; `salloc` first and it works, and `deploy.sh` falls back to `srun --overlap`
and says which path it took. `$HOME` and `/mnt/common` are shared, so the venv, the FAISS
cache and `~/.claude` need no staging. `TMPDIR` is pointed at NVMe because the compute
nodes' root filesystem is effectively full.

---

## Known limitations

**The generator is unchanged.** `predict_until_length` filled every mask in a single
forward pass, so hundreds of adjacent masks mostly saw other masks and collapsed onto a
few likely tokens. `tools/mlm.py` keeps that behaviour deliberately — the four changes it
does make are the context position, the returned counters, the `threshold` argument and
the measured mask seed, so any difference is attributable to those and not to a quietly
different sampler. The output is still low-entropy and not really DNA; see `RESULTS.md`.

**The gate is about plumbing, not correctness.** It verifies that retrieved context
reached the model, that the length is plausible and that the output is nucleotides. It
says nothing about whether the reconstruction is right.

Measured on gap1: it is not. Scored properly, the predictions land at 33-42% identity
against the true gap, and composition-matched random DNA scores the same. The system
provably delivers context to the generator; the generator provably cannot use it. See
`RESULTS.md` section 6, including why the restricted run's apparently better 42% is an
artefact of base composition rather than a signal.

**Single task at a time.** One process-global model lock, one task per agent in flight.
Concurrency would need per-task model instances or a queue.

**In-memory state.** Task state and retrieval pools do not survive a restart. Everything
worth keeping is written to `logs/tasks-<role>.jsonl` and `results/`.

**Terminal gaps are declined.** `gap33` of AP012051.1 runs past the end of the assembly
and has no right-hand contig. 32 of the 33 gaps are reconstructable; the odd one out
fails preflight rather than being filled against an invented flank.
