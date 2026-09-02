# Multi-Agent System

---

# Why Multi-Agent?
## Limits of the Single-Agent Pipeline

- **Tool ordering requested, not enforced**
  - Correct execution depended on the prompt.

- **Genomic payload passed through the LLM**
  - Sequences reproduced by the model between tools.

- **Retrieval returned no context**
  - Substring matching found 0 matches in 37,645 records.

- **Retrieved context silently truncated**
  - Fixed `k=2`, blind to the available token budget.

**The problem was not only model quality. The pipeline did not prove that the retrieval step influenced the reconstruction step.**

---

# Agent Communication Protocols: A2A vs. ANP

| Criterion | A2A | ANP |
|---|---|---|
| **Discovery** | Agent Cards over HTTP | DIDs + distributed indexes |
| **Task model** | Stateful lifecycle + artifacts | No standard task lifecycle |
| **Long-running tasks** | Native support | Application-dependent |
| **Deployment** | Standard HTTP infrastructure | Decentralized identity layer |

**A2A better matches GenomIO because the agents are known in advance and the workflow requires stateful, long-running tasks in a controlled environment.**

**In this project, A2A is used as the service-to-service protocol between agents. MCP is used inside each agent to expose its local tools to its own LLM controller.**

---

# How A2A Works

**Implemented flow:** `Agent Card -> Message -> Task -> Artifact`

## 1. Agent Card: discovery and capability declaration

Each agent publishes an Agent Card at `/.well-known/agent-card.json`. Other agents use it to know:

- where the agent is reachable,
- which skills it exposes,
- whether it supports streaming or push notifications,
- which corpus snapshot it is using.

**Simplified from implementation:**

```python
return {
    "protocolVersion": config.PROTOCOL_VERSION,
    "name": name,
    "description": description,
    "url": url,
    "preferredTransport": "JSONRPC",
    "skills": skills,
    "metadata": metadata,
}
```

The actual runtime also adds `metadata["x-tools"]`, so the card exposes the MCP tool names available inside that agent.

## 2. Message: structured request, not free text

The Reconstruction Agent asks the Retrieval Agent for candidates by building an A2A message whose useful payload is a `DataPart`.

```python
message = a2a_types.build_message(
    role="user",
    parts=[a2a_types.data_part(payload)],
    context_id=payload.get("context_id") or "",
)
task = await _retrieval_client().send(message)
```

`send()` wraps the message as JSON-RPC:

```python
rpc_request("message/send", {"message": message})
```

## 3. Task lifecycle: stateful remote execution

When a message arrives, the receiving A2A server creates a task, runs one agent turn, and returns the terminal task.

```text
submitted -> working -> completed
                    \-> failed

malformed request or corpus mismatch -> rejected
```

## 4. Artifact: machine-readable result

The requesting agent reads the result from the task artifact, not from the LLM's natural-language answer.

```python
state = a2a_types.task_state(task)
data = a2a_types.artifact_data(task)
```

For Retrieval, that artifact contains `row_ids`, `scores`, `pool_size`, `served_total`, `pool_exhausted`, and retrieval path metadata.

```text
Reconstruction Agent
        | A2A message: request candidate rows
        v
 Retrieval Agent
        | A2A artifact: row_ids + scores + pool state
        v
Reconstruction Agent
```

**Key distinction:**  
**A2A -> communication between agents**  
**MCP -> tools executed inside each agent**

---

# Multi-Agent Architecture

- A2A communication between agents.
- MCP tools within each agent.
- Deployed across 3 ARES nodes.
- Three specialized agents: **Coordinator, Retrieval and Reconstruction**.
- Structured task state for genomic data and intermediate results.
- Direct **Retrieval ↔ Reconstruction** interaction for context extension.
- Coordinator validation before task completion.

```text
                         User
                           │
                           ▼
                  Coordinator Agent
                           │
                           ▼
                    Retrieval Agent
                           │
                           │
                           ├───────────────────┐
                           ▼                   ▼
                      RAG Corpus      Reconstruction Agent
                                              │
                                              ▼
                                            GENA-LM
```

## Responsibility split

| Agent | Owns | Does not do |
|---|---|---|
| **Coordinator** | workflow policy, peer checks, acceptance gate, trace | no retrieval, no generation, no sequence fabrication |
| **Retrieval** | DNABERT-S, FAISS index, ranked candidate pool | no gap filling, no nucleotide sequence transfer |
| **Reconstruction** | GENA-LM, token budget, candidate admission, generation | no corpus search, no final acceptance decision |

The architecture is deliberately asymmetric: Retrieval knows the ranking but not the GENA-LM context budget; Reconstruction knows the budget but not the FAISS ranking. The negotiation between them is therefore required, not decorative.

---

# Agent Configuration: Prompts & Tools

Each agent is an LLM-controlled service, but the LLM is only allowed to call that agent's own MCP tools. The prompt describes the protocol; the tool guards enforce it.

| Agent | Main Prompt Instructions | Tools |
|---|---|---|
| **Coordinator** | Do not fill gaps and do not handle nucleotide sequences. Call tools in order: prepare the query, delegate reconstruction, evaluate the returned result, emit the trace. If a peer fails, report it rather than fabricating a result. | `prepare_query()`, `delegate_reconstruction()`, `evaluate_result()`, `emit_trace()` |
| **Retrieval** | Own DNABERT-S and FAISS. For `retrieve`, open a ranked candidate pool; for `extend`, serve the next slice from that same pool. Return row IDs and scores only. Never write nucleotide sequences. | `retrieve_context()`, `extend_context()`, `describe_rows()` |
| **Reconstruction** | Own GENA-LM and the token budget. Measure free capacity, request candidates over A2A, admit only rows that fit, extend while space remains, then generate with the original `gap_length`. | `measure_budget()`, `request_candidates()`, `extend_candidates()`, `admit_candidates()`, `generate()`, `report_state()` |

## Tool confinement

The runtime passes both `tools` and `allowed_tools` to the Claude Agent SDK. This is important because an early smoke run showed that `allowed_tools` alone still allowed a built-in `ToolSearch` call.

```python
"tools": list(self.spec.allowed_tools),
"allowed_tools": list(self.spec.allowed_tools),
"permission_mode": "bypassPermissions",
"setting_sources": [],
```

At startup, `LLMAgent` also checks that the allowed MCP tool IDs exactly match the tools registered for that agent:

```python
expected = {f"mcp__{spec.server_key}__{t.name}" for t in spec.tools}
if expected != set(spec.allowed_tools):
    raise RuntimeError(...)
```

This turns a silent configuration drift into a startup failure.

---

# Negotiation

**Problem solved:** retrieval cannot know how many candidate sequences fit inside GENA-LM's context window; reconstruction cannot rank corpus records by similarity. The system makes them negotiate instead of choosing a fixed `k`.

```text
1. measure_budget()
   Reconstruction tokenizes the base input:
   left flank + mask block + right flank.

2. request_candidates()
   Reconstruction asks Retrieval for an initial ranked batch.

3. admit_candidates()
   Reconstruction resolves row IDs locally, tokenizes each candidate,
   and admits only the records that fit.

4. extend_candidates()
   If there is still room, Reconstruction asks Retrieval for more rows
   from the same ranked pool.

5. generate()
   GENA-LM runs with the admitted context placed before the sequence.
```

## What makes this more than prompt discipline

- `request_candidates()` is refused until `measure_budget()` has created a real budget.
- `admit_candidates()` rejects a stale or invented `free_tokens` value.
- `extend_candidates()` uses the same `request_id`, so Retrieval continues the existing pool.
- `generate()` is refused unless at least one candidate was admitted.
- `gap_length` must match the original task value exactly.

## Why this matters

The previous pipeline used `k=2` independently of the available context. In the tested run, the multi-agent negotiation admitted **2,942 retrieved-context tokens out of 3,236 free tokens**: about **91%** of the available capacity.

With the previous fixed `k=2` behavior, the same window would have used roughly **11%** of the available capacity and would not have reported whether the model actually saw the retrieved context.

---

# What Was Tested?

**The tests were designed to answer one question:** does the multi-agent architecture enforce the properties that the single-agent pipeline only requested in a prompt?

## 1. Agent coordination

The end-to-end acceptance harness submits one gap to the Coordinator and verifies the complete route:

```text
User -> Coordinator -> Reconstruction <-> Retrieval -> Reconstruction -> Coordinator
```

Validated behavior:

- Coordinator task reaches A2A `completed`.
- Coordinator delegates reconstruction and receives a machine-readable artifact.
- Peer readiness is checked through `/health`.
- Retrieval and Reconstruction corpus fingerprints must match before delegation.
- Coordinator keeps final acceptance behind `evaluate_result()`.

## 2. Retrieval-Reconstruction negotiation

Validated behavior:

- Reconstruction requests candidate batches from Retrieval over A2A.
- Retrieval returns `row_ids` and `scores`, not nucleotide sequences.
- `extend_candidates()` asks for the next batch from the same ranked pool.
- Pool reuse is observable in the trace: `pool_size` remains constant while `served_total` increases.
- If the pool has disappeared, `extend_context()` reports `NO_POOL` instead of silently rebuilding a new ranking.

## 3. Context and budget management

Validated behavior:

- `measure_budget()` reports `base_tokens`, `free_tokens`, `reserved_for_masks`, `model_flank`, and `seed_mask_count`.
- Candidate admission is bounded by the measured free-token budget.
- A candidate that is too large is rejected without stopping the scan; later smaller candidates can still be admitted.
- Duplicate rows are counted but not charged twice.
- Wrong `free_tokens` values are rejected with `BUDGET_MISMATCH`.
- Final model input is checked as `truncated=False`.
- `context_tokens_admitted > 0` is required; otherwise the run cannot be called retrieval-augmented.

## 4. State and data integrity

Validated behavior:

- Tools are keyed by `task_id` or `request_id`, not by large sequence arguments.
- No MCP tool schema accepts `sequence`, `left`, `right`, `context`, or `rag_matches`.
- `gap_length` remains unchanged across request, budget, artifact, and trace.
- A2A results are read from structured artifact `DataPart`s.
- The LLM's prose is kept only as audit text and is never parsed as the result.

## 5. Execution dependency enforcement

Invalid ordering was tested directly:

- `request_candidates()` before `measure_budget()` is refused.
- `admit_candidates()` before `measure_budget()` is refused.
- `generate()` before admission is refused.
- `evaluate_result()` before delegation is refused.
- call-budget and deadline violations terminate the task.

This is the central architectural improvement: execution order is enforced by state and tool preconditions, not trusted to model obedience.

## 6. Final validation gate

The Coordinator acceptance gate validates:

- output is non-empty,
- produced length is within tolerance of `gap_length`,
- alphabet is a subset of `ACGTN`,
- admitted retrieval context is greater than zero.

The live acceptance harness also checks:

- retrieval mode is correct,
- model input was not truncated,
- candidate consumption is valid,
- no model called a foreign or built-in tool,
- the retrieval pool was reused rather than rebuilt.

## Worked

- A2A delegation worked between independent services.
- The direct Retrieval ↔ Reconstruction negotiation worked.
- Context admission adapted to measured GENA-LM capacity.
- Retrieval pool state persisted across A2A requests.
- Structured task state prevented nucleotide payloads from being passed through LLM messages.
- Coordinator validation became a real completion gate.

## Did Not Work / Limitations

- The previous single-agent failures were not final multi-agent failures: fixed `k=2`, prompt-only ordering, silent truncation, and LLM-mediated genomic payloads were addressed by the new architecture.
- The multi-agent system fixed the **plumbing**, not the biological quality of the generator.
- GENA-LM still produced sequences with poor identity against the true gap. The acceptance gate verifies delivery and basic validity, not biological correctness.
- Unrestricted DNABERT-S retrieval worked mechanically, but on the real query it degraded after rank 1; organism-restricted retrieval was more reliable.
- Task state and retrieval pools are in memory, so they do not survive process restart.
- The implementation assumes known agents in a controlled deployment; decentralized discovery, authentication, and cross-domain trust were outside the implemented scope.

---

# Multi-Agent System Improvements

## 1. Independent responsibilities

The system separates orchestration, retrieval, and reconstruction into three services instead of asking one agent to do everything.

- **Coordinator** makes policy decisions: prepare the query, verify peers, delegate, validate, trace.
- **Retrieval** handles similarity search: DNABERT-S embedding, FAISS search, ranked pools, organism restriction.
- **Reconstruction** handles model-context decisions: GENA-LM token budget, candidate admission, masked generation.

This matters because each agent owns the information needed for its decision. Retrieval owns ranking; Reconstruction owns the context window; Coordinator owns acceptance.

## 2. Structured execution

Large genomic values are kept in structured task state rather than passed through natural-language reasoning.

- The A2A message carries structured `DataPart`s.
- MCP tools receive identifiers such as `task_id` and `request_id`.
- The final artifact is serialized from state.
- LLM text is not treated as the source of truth.

This directly addresses the previous failure mode where sequences were reproduced through the model with no checksum or guarantee of fidelity.

## 3. Enforced dependencies

The implementation turns workflow order into code-level preconditions.

| Required dependency | Enforced by |
|---|---|
| budget before candidates | `request_candidates()` requires phase `MEASURED` |
| current budget before admission | `admit_candidates()` checks exact `free_tokens` |
| admitted context before generation | `generate()` requires phase `ADMITTING` and `context_tokens_admitted > 0` |
| delegated result before evaluation | `evaluate_result()` requires phase `DELEGATED` |
| matching corpus snapshots | Coordinator compares Agent Card corpus fingerprints |

This is stronger than telling the LLM to call tools in the right order.

## 4. Adaptive context management

Retrieval adapts dynamically to the available token budget.

- The ranked candidate pool is opened once.
- Later rounds serve the next slice of the same pool.
- Candidate sequences are admitted only if they fit the remaining GENA-LM window.
- The final run used **2,942 / 3,236 free tokens**, about **91%** of available capacity.
- The previous fixed `k=2` configuration would have used about **11%** of the same capacity.

## 5. Validation and consistency

The Coordinator validates the reconstruction before task completion.

- length within tolerance,
- valid nucleotide alphabet,
- retrieved context actually admitted,
- no silent truncation in the acceptance harness,
- row IDs remain consistent through corpus fingerprints.

The main outcome is not that the generated DNA became biologically correct. The main outcome is that the system can now prove whether retrieval reached the generator, preserve the intermediate state, and distinguish pipeline failures from model-quality failures.
