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

---

# Agent Communication Protocols: A2A vs. ANP

| Criterion | A2A | ANP |
|---|---|---|
| **Discovery** | Agent Cards over HTTP | DIDs + distributed indexes |
| **Task model** | Stateful lifecycle + artifacts | No standard task lifecycle |
| **Long-running tasks** | Native support | Application-dependent |
| **Deployment** | Standard HTTP infrastructure | Decentralized identity layer |

**A2A better matches GenomIO because the agents are known in advance and the workflow requires stateful, long-running tasks in a controlled environment.**

---

# How A2A Works

**Flow:** `Agent Card -> Message -> Task -> Artifact`

- Each agent exposes an Agent Card at `/.well-known/agent-card.json`.
- The card advertises identity, endpoint, skills, capabilities, tools, and consistency metadata.

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

**Reconstruction -> Retrieval A2A request**

```python
message = a2a_types.build_message(
    role="user",
    parts=[a2a_types.data_part(payload)],
    context_id=payload.get("context_id") or "",
)
task = await _retrieval_client().send(message)
```

- `send()` wraps the message as JSON-RPC `message/send`.
- The server creates a task, runs the agent, and returns the terminal task.

```text
submitted -> working -> completed
                    \-> failed

malformed request -> rejected
```

- Results are returned as an A2A artifact `DataPart`, not parsed from LLM prose:

```python
state = a2a_types.task_state(task)
data = a2a_types.artifact_data(task)
```

```text
Reconstruction Agent
        | A2A request: row candidate batch
        v
 Retrieval Agent
        | artifact: row_ids, scores, pool state
        v
Reconstruction Agent
```

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

---

# Agent Configuration: Prompts & Tools

| Agent | Main Prompt Instructions | Tools |
|---|---|---|
| **Coordinator** | Orchestrate only: do not fill gaps or handle sequences. Call `prepare_query`, delegate reconstruction, apply the acceptance gate, then emit the trace. Validate length, alphabet, and that retrieved context reached the model. | `prepare_query()`, `delegate_reconstruction()`, `evaluate_result()`, `emit_trace()` |
| **Retrieval** | Own DNABERT-S and FAISS. For `retrieve`, open a ranked pool and serve a first batch; for `extend`, serve the next batch from the same pool. Return row IDs and scores only, never nucleotide sequences. | `retrieve_context()`, `extend_context()`, `describe_rows()` |
| **Reconstruction** | Own GENA-LM and the token budget. Measure free context capacity, request candidates over A2A, admit only candidates that fit, extend while space remains, then generate with the original `gap_length`. | `measure_budget()`, `request_candidates()`, `extend_candidates()`, `admit_candidates()`, `generate()`, `report_state()` |

**Runtime confinement**

- Each agent registers only its own MCP server and tool IDs.
- Built-in shell, filesystem, search, and web tools are excluded.
- A startup check rejects mismatches between registered tools and `allowed_tools`.

```python
"tools": list(self.spec.allowed_tools),
"allowed_tools": list(self.spec.allowed_tools),
"permission_mode": "bypassPermissions",
"setting_sources": [],
```

---

# Negotiation

1. **`measure_budget()`**  
   Reconstruction Agent computes free token capacity.

2. **`request_candidates()`**  
   Asks the Retrieval Agent for ranked candidates.

3. **`admit_candidates()`**  
   Adds candidates while they fit the budget.

4. **`extend_candidates()`**  
   More rounds from the same ranked pool while space remains.

5. **`generate()`**  
   Reconstruction runs with the admitted context.

**Retrieval can adapt to the context available for each reconstruction instead of relying on a fixed number of candidates.**

---

# What Was Tested?

**Agent coordination**

- End-to-end flow through `Coordinator -> Reconstruction <-> Retrieval -> Coordinator`.
- Coordinator task reaches `completed`, delegates reconstruction, receives the artifact, and keeps final completion behind `evaluate_result()`.
- Peer readiness and corpus fingerprints are checked before delegation.

**Retrieval-Reconstruction interaction**

- Reconstruction can request candidate batches from Retrieval over A2A.
- `extend_candidates()` continues the same ranked pool rather than rebuilding it.
- Pool reuse is checked through constant `pool_size` and monotonically increasing `served_total`.

**Context management**

- `measure_budget()` reports real `base_tokens`, `free_tokens`, model flank usage, and mask reservation.
- `admit_candidates()` accepts only records that fit the current free budget.
- Wrong `free_tokens` values are rejected and corrected.
- Final input is checked as not truncated, and admitted context tokens must be `> 0`.

**State and data integrity**

- Tools are keyed by `task_id` or `request_id`, not by raw sequence payloads.
- `gap_length` is checked unchanged across request, budget report, artifact, and trace.
- Retrieval returns row IDs and metadata, never nucleotide sequences.
- A2A artifacts are serialized from structured state; LLM text is audit-only.

**Execution dependencies**

- Out-of-order calls are refused before work starts:
  - `request_candidates()` before `measure_budget()`
  - `admit_candidates()` before `measure_budget()`
  - `generate()` before `admit_candidates()`
- Call budgets and deadlines terminate looping tasks.

**Final validation**

- Coordinator gate validates:
  - produced length within tolerance,
  - alphabet subset of `ACGTN`,
  - retrieved context actually reached the model.
- The acceptance harness also checks retrieval mode, no truncation, no foreign tools, pool reuse, and valid candidate consumption.

**Worked**

- Final integration checks passed for the intended plumbing properties.
- A2A delegation and Retrieval <-> Reconstruction negotiation worked across independent services.
- Context extension adapted to measured budget and reused the ranked pool.
- Structured state preserved genomic values and prevented natural-language sequence handoff.
- Coordinator validation acted as the completion gate.

**Did Not Work / Limitations**

- These are **not** final multi-agent failures: the previous single-agent system had fixed `k=2`, silent truncation, prompt-only ordering, and genomic payloads passed through LLM messages.
- In the final multi-agent system, the plumbing was validated, but the generator was not fixed: GENA-LM still produced sequences with poor biological identity against the true gap.
- Retrieval machinery worked, but unrestricted DNABERT-S retrieval degraded beyond rank 1 on the real query; organism-restricted retrieval was more reliable.
- The implementation covers known agents in a controlled environment; dynamic discovery/security beyond the local A2A cards was not implemented.

---

# Multi-Agent System Improvements

## Independent Responsibilities

- Coordination, retrieval, and reconstruction operate as independent services.
- Each agent manages its own tools and resources.

## Structured Execution

- Genomic data and intermediate results remain in structured task state.
- Tool preconditions preserve dependencies between operations.

## Adaptive Context Management

- Retrieval adapts dynamically to the available token budget.
- The ranked candidate pool is reused across retrieval rounds.
- **91%** of the available context capacity was used, vs. **~11%** with the previous fixed `k=2` configuration.

## Validation and Consistency

- The Coordinator validates the reconstruction before task completion.
- Corpus fingerprints and execution checks maintain consistency across agents.
