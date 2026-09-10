# GenomIO Multi-Agent System

## Overview

This implementation coordinates retrieval-augmented reconstruction of a gap between two genomic contigs. Three agents use the Claude Agent SDK to select and invoke MCP tools. DNABERT-S supplies DNA retrieval; GENA-LM supplies masked-language-model generation. The system records how much retrieved context is admitted and evaluates the output's length and alphabet.

It is separate from the older LangChain workflow in [gap-filler-agents/](../gap-filler-agents). Its purpose is to make retrieval, context admission, and reconstruction outcomes observable. An accepted result is not evidence that the missing biological sequence was recovered.

## Architecture

```text
test.py: load the gap and its adjacent contigs
    |
    v
Coordinator (8100): prepare, delegate, evaluate, trace
    | A2A message/send
    v
Reconstruction (8102): measure budget, admit candidates, generate
    | A2A message/send, repeated batch requests
    v
Retrieval (8101): DNABERT-S + FAISS, persistent ranked pool
    |
    +--> row IDs and scores returned to Reconstruction

Reconstruction resolves rows against the local corpus store,
returns generation data to Coordinator, then Coordinator returns
the sequence with its verdict and trace to the caller.
```

The three roles run as separate processes locally or on allocated SLURM nodes. [agents/](agents) defines their prompts and state-to-artifact conversion; [tools/](tools) implements capabilities; [a2a/](a2a) implements transport; [runtime/](runtime) provides the SDK loop, state registry, guards, and launcher. The directory layout is unchanged.

## Agent Responsibilities

### Coordinator Agent

[agents/coordinator.py](agents/coordinator.py) validates that both contigs and a positive target length are supplied. It builds the retrieval query from the gap-adjacent flanks, checks peer readiness and reported corpus fingerprints, delegates reconstruction, and evaluates the returned sequence. It loads no DNA embedding or generation model; its orchestration still uses the configured agent LLM.

### Reconstruction Agent

[agents/reconstruction.py](agents/reconstruction.py) owns the GENA-LM tokenizer/model. It measures the available context budget, requests candidate batches, resolves corpus rows locally, admits sequences that fit, and generates the gap sequence. A stable `pool-<task_id>` request ID allows several retrieval calls to use the same ranking.

### Retrieval Agent

[agents/retrieval.py](agents/retrieval.py) owns the DNABERT-S embedder and cached FAISS index. It opens a ranked pool and returns row IDs, cosine scores, and pool status in batches. It supports unrestricted search and organism-restricted search. `describe_rows` provides metadata for already served rows. It does not return retrieved nucleotide sequences to Reconstruction; Reconstruction reads those from its local record store.

## Communication

The [A2A client](a2a/client.py) sends JSON-RPC 2.0 `message/send` requests to `POST /`. Message `DataPart` objects carry structured input; `contextId` groups related tasks. Coordinator delegates `op="reconstruct"` with the contigs, target length, query, and optional organism. Reconstruction sends `op="retrieve"` or `op="extend"`, a stable `request_id`, and a requested batch size to Retrieval.

Each request creates an A2A task. Results are artifacts containing structured `DataPart` data serialized from tool-maintained state. Optional `TextPart` narrative is advisory; callers read structured artifacts rather than parsing model prose. Initial agent prompts contain identifiers and sequence lengths instead of the full DNA payload. Tool summaries can include a 60-character generated-sequence preview, so the implementation does not exclude all nucleotide text from LLM-visible results.

The transport also implements `tasks/get`, `tasks/cancel`, `GET /tasks/{id}`, `GET /health`, and agent cards at `/.well-known/agent-card.json` and `/.well-known/agent-configuration`. `message/send` waits for the task result. Streaming, push notifications, file parts, and authentication are not implemented. These endpoints are intended for a trusted execution network.

The [task registry](runtime/state.py) stores state in memory. Retrieval sessions persist across requests; they do not survive process restart. Task deadlines are checked at tool entry, and calls have budgets. These checks do not constitute an interrupt of an already running model operation. The task store prevents rewriting terminal status; protocol completion and the biological acceptance verdict are separate concepts.

## Tools

MCP names use `mcp__<server_key>__<tool_name>`. The server keys are `coordinator_agent`, `retrieval_agent`, and `reconstruction_agent`. The runtime checks registered names against allowed tools and supplies both `tools` and `allowed_tools` to the SDK.

| Agent | Tool | Role in the workflow |
|---|---|---|
| Coordinator | `prepare_query` | Slice query flanks, check peers and reported corpus identity, select retrieval mode |
| Coordinator | `delegate_reconstruction` | Send the prepared request and retain the peer result |
| Coordinator | `evaluate_result` | Apply the length, alphabet, and admitted-context gate |
| Coordinator | `emit_trace` | Return the run record; final serialization also builds a trace |
| Retrieval | `retrieve_context` | Open/reuse a ranked pool and serve a batch |
| Retrieval | `extend_context` | Serve the next pool slice without re-embedding |
| Retrieval | `describe_rows` | Describe organism/protein metadata for served rows |
| Reconstruction | `measure_budget` | Tokenize flanks and reserve mask/safety positions |
| Reconstruction | `request_candidates` | Open the peer pool with an agent-chosen batch size |
| Reconstruction | `extend_candidates` | Request more candidates from that pool |
| Reconstruction | `admit_candidates` | Resolve row IDs, count tokens, admit records that fit |
| Reconstruction | `generate` | Invoke the masked fill and retain sequence/counters |
| Reconstruction | `report_state` | Return phase, budget, counts, and trace diagnostics |

Implementations: [coordinator tools](tools/coordinator_tools.py), [retrieval tools](tools/retrieval_tools.py), and [reconstruction tools](tools/reconstruction_tools.py).

## Execution Flow

1. [dataset.py](dataset.py) reads contig FASTA headers and gap TSV coordinates. It selects the contig ending at the gap start and the one beginning at the gap end. Missing or ambiguous flanks fail preflight; the loader does not simply take the first two FASTA records.
2. Coordinator prepares the query and checks that Retrieval and Reconstruction report ready and have matching corpus fingerprints, then delegates.
3. Reconstruction trims the model flanks and measures free token positions. Its LLM chooses candidate batch sizes within the configured bounds.
4. Retrieval embeds the query and creates a pool. Reconstruction reads offered rows from its local corpus store and admits sequences that fit. Additional batches reuse the pool until generation is requested or the pool is exhausted.
5. [tools/mlm.py](tools/mlm.py) assembles context before the flanks and mask slot, performs the existing masked-fill length search, and returns the sequence plus attempts, token lengths, truncation, and alphabet diagnostics.
6. Coordinator evaluates the result and returns a sequence, verdict, and trace. A rejected sequence remains available for inspection. A completed A2A task is not necessarily an `ACCEPTED` reconstruction.

## Retrieval and Context Management

The default retrieval query uses 900 bases from each side of the gap. Model input separately retains up to 2,000 bases from each flank; `GENOMIO_MODEL_FLANK=0` disables that trimming. Reconstruction computes:

```text
free_tokens = max(0, MAX_LENGTH - base_tokens - seed_mask_count - SAFETY_TOKENS)
```

`base_tokens` is measured with the context slot empty. The initial mask count uses the tokenizer's measured bases-per-token ratio on the actual flanks. Defaults are a 4,096-token window and 32 safety tokens.

Candidates are tokenized individually. A whole record is admitted when its cost fits; an oversized record is skipped, allowing later shorter records to fit. Already admitted rows are not added twice. The `free_tokens` argument must equal the stored remaining budget. The counter `context_tokens_admitted` sums admission costs; it is not an attention measurement or proof of biological use. The assembled input is measured separately and `truncated` is reported because subsequent mask-count adjustments can change its size.

The local [corpus store](tools/corpus_store.py) indexes byte offsets in `.cache/rag_index/records.jsonl`. Peer fingerprints use the manifest contents and records-file size, not a full hash of every sequence. [Organism lookup](tools/organism_index.py) resolves names and unambiguous partial matches to contiguous row ranges. Restricted retrieval over-fetches and filters rows; a range-selector fallback is attempted if no rows remain. Neither a metadata embedding index nor rank fusion is used here.

## Validation

The [acceptance gate](gate.py) checks a nonempty output, characters within `ACGTN`, positive admitted-context count, and length error no larger than `max(3, ceil(TOLERANCE * gap_length))`. The default tolerance is 0.05. Rejection reasons include `EMPTY`, `LENGTH`, `ALPHABET`, and `NO_CONTEXT`.

Additional safeguards and tests cover different properties:

- Tool guards require known state and prerequisite phases, and check deadlines/call budgets before running handlers. Generation requires admitted context and the original target length.
- Admission validates row bounds, duplicates, and budget consistency. It does not independently enforce that every requested row was previously offered; the integration harness compares consumed/offered counts, not complete row-set provenance.
- [test.py](test.py) checks completion, acceptance, nonzero admission, candidate counts, length, alphabet, absence of truncation, unchanged target length, returned retrieval fields, retrieval mode, tool confinement, and pool reuse.
- [tests/](tests) covers protocol behavior, state/tool contracts, corpus access, flank selection, generation helpers, gate behavior, naming, and deployment parsing using lightweight fixtures where possible.

The gate itself does not check truncation, homology, sequence identity, or correct use of the retrieved biology. No property should be inferred solely from the number of passing assertions.

## Deployment and Execution

### Dependencies and preparation

The documented deployment targets Linux/Bash, locally or on the ARES SLURM cluster. It requires Python, the project ML/retrieval dependencies in [requirements.txt](../requirements.txt), the Claude Agent SDK and its MCP dependencies, and `httpx` for the asynchronous A2A client. The SDK is not listed in the root requirements. Historical runs in [RESULTS.md](RESULTS.md) record Python 3.10.12 and SDK 0.2.132; there is no fully pinned environment for reproducing them.

The existing launch procedure creates a virtual environment inheriting the installed ML stack. From the repository root:

```bash
python3 -m venv --system-site-packages multiagent_system/.venv
multiagent_system/.venv/bin/pip install claude-agent-sdk
```

The agents use the authenticated Claude CLI. `runtime/launcher.py`, the local fleet, and `deploy.sh` remove `ANTHROPIC_API_KEY` from their launch environment to use CLI authentication. Credentials and remote model access must already be available; they are not provided by this repository.

Build the shared DNA index before launching Reconstruction, whose corpus store requires the cached records:

```bash
python3 scripts/build_rag_index.py --check
python3 scripts/build_rag_index.py
```

The build can download model weights and take substantial CPU time. All agents need the same `.cache/rag_index/` and corpus snapshot. Model warm-up loads GENA-LM for Reconstruction and DNABERT-S/FAISS for Retrieval before they report ready.

### Local execution

```bash
multiagent_system/.venv/bin/python3 multiagent_system/test.py --mode local
```

The harness launches three processes, waits for health readiness, submits the default AP012051.1 gap1 case, checks the result, saves JSON, and tears down the processes. It accepts `--accession`, `--gap`, `--organism`, `--base-port`, `--threads`, `--ready-timeout`, and `--keep-alive`. The default case is an 870-base gap. This is a live model-backed run, not a lightweight test.

### SLURM execution

The checked-in [deploy.sh](deploy.sh) demonstrates one role per node on ARES:

```bash
salloc -N 3 -p compute -t 02:00:00 --no-shell -J genomio-mas
./multiagent_system/deploy.sh <JOBID>
multiagent_system/.venv/bin/python3 multiagent_system/test.py --mode slurm
scancel <JOBID>
```

Replace `<JOBID>` with the allocation ID. The script also supports `--alloc`. It resolves allocated hosts, sets peer URLs, starts roles using SSH with an `srun --overlap` fallback, and records endpoints under `.run/`. The harness can instead take `--endpoints-file` or `--endpoints`. Shared repository/cache/credentials and the cluster's temporary-storage layout are deployment assumptions in the script, not general multi-cluster portability guarantees.

### Configuration and outputs

[config.py](config.py) is the source of defaults; supported environment overrides include:

| Variable | Default | Purpose |
|---|---|---|
| `GENOMIO_BASE_PORT` | 8100 | Coordinator port; Retrieval/Reconstruction default to the next two |
| `GENOMIO_URL_COORDINATOR`, `GENOMIO_URL_RETRIEVAL`, `GENOMIO_URL_RECONSTRUCTION` | localhost role URLs | Peer endpoints |
| `GENOMIO_QUERY_FLANK` | 900 | Retrieval flank length in bases |
| `GENOMIO_MODEL_FLANK` | 2000 | Model flank length; zero disables trimming |
| `GENOMIO_MAX_LENGTH`, `GENOMIO_SAFETY_TOKENS` | 4096, 32 | Token window and reservation |
| `GENOMIO_TOLERANCE` | 0.05 | Gate length tolerance |
| `GENOMIO_THRESHOLD`, `GENOMIO_MAX_ATTEMPTS` | 0.01, 10 | Generation settings |
| `GENOMIO_GENA_LM` | `AIRI-Institute/gena-lm-bigbird-base-t2t` | Reconstruction model |
| `GENOMIO_AGENT_MODEL` | `claude-sonnet-4-5` | LLM driving the three agents |
| `GENOMIO_MAX_TURNS`, `GENOMIO_MAX_CALLS_PER_TOOL` | 24, 12 | Conversation/tool limits; some tools have smaller explicit caps |
| `GENOMIO_TASK_DEADLINE_S` | 900 | Deadline checked by tool guards |
| `GENOMIO_SEED` | 20260807 | Torch seed set at model startup |

The seed alone does not guarantee identical complete runs: the agent chooses batches, and sampling state advances. Other retrieval-pool and timeout settings are defined in `config.py`; paths there are repository-relative constants, not all environment-overridable.

Runtime outputs, ignored by Git, include `multiagent_system/results/<timestamp>_<gap_id>.json`, `multiagent_system/logs/tasks-<role>.jsonl`, process logs, and `.run/` endpoint/process metadata. JSON records include the task result, trace, and harness checks. Important trace fields are `context_tokens_admitted`, `free_tokens`, `input_tokens`, `truncated`, `candidates_offered`, `candidates_consumed`, `verdict`, `generation_attempts`, and `admitted_row_ids`.

## Results and Validation Evidence

[RESULTS.md](RESULTS.md) preserves historical local/SLURM measurements, trace excerpts, and test summaries. It reports 2,942 admitted tokens for an unrestricted gap1 run, with 3,850 input tokens within the 4,096-token window. Its reconstruction comparisons include composition-matched random baselines and do not demonstrate reliable recovery of the true gap. Findings concern that recorded case, not all genomes or possible generators.

The raw timestamped JSON files and logs cited there are absent from this checkout. Its older ten-assertion wording and later twelve-check listing describe different documentation stages; the current harness is the source for implemented checks. Historical pass counts are not a test run performed by opening this README.

Run lightweight tests with heavy tests disabled:

```bash
multiagent_system/.venv/bin/python3 -m pytest multiagent_system/tests -c multiagent_system/pytest.ini
```

The test configuration skips tests marked `heavy` unless `GENOMIO_HEAVY=1`. If the SDK is absent, SDK-dependent test modules are excluded. Deployment-parsing tests require a compatible Linux/Bash environment. The unmarked `test_agent_specs_build_and_validate_their_own_naming` also reads the real cached corpus through the Retrieval agent card, so a fresh checkout without `records.jsonl` can fail that test even with heavy tests disabled. Report skips and environment failures separately from passes.

## Operational Limits

Model work uses a process-global lock through [runtime/blocking.py](runtime/blocking.py). The current harness submits one gap at a time; concurrent multi-gap behavior is not established by the recorded evidence. State is in memory, and the HTTP protocol has no authentication. Metadata fusion, broader biological validation, and changes to masked generation remain outside this implementation. See [known generation/scoring issues](../docs/gap_filling_known_bugs.md) and the historical report's future-work discussion.
