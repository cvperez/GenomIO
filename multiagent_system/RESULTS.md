# Results

> Historical validation report: the measurements and excerpts below are preserved. The cited timestamped logs and JSON artifacts are absent from this checkout. See [README.md](README.md) for current tool behavior, validation limits, and execution instructions. The older ten-assertion summary and later twelve-check listing reflect different documentation stages.

What was built, what was measured, and what the numbers do and do not support.
All figures below come from runs on this machine; nothing is estimated.

Environment: ARES login node `ares.ares.local` and compute nodes `ares-comp-10/11/12`.
Python 3.10.12, CPU only, no GPU. `claude-agent-sdk` 0.2.132 driving
`claude-sonnet-4-5` through the local CLI on subscription auth.

---

## 1. The acceptance test

Ten assertions on gap1 of AP012051.1 (870 bases, between `contig1` and `contig2`).
**All ten pass, in both deployment modes.**

| | local, 3 processes | SLURM, 3 nodes |
|---|---|---|
| fleet ready | 13 s | 51 s (cold DNABERT-S) |
| run time | 250 s | 210 s, 220 s, 167 s over three runs |
| assertions | 10 / 10 | 10 / 10, every run |

On SLURM the coordinator ran on `ares-comp-10`, retrieval on `ares-comp-11` and
reconstruction on `ares-comp-12` : one agent per node, talking over the cluster network.
Four runs in total (one local, three distributed, one of them organism-restricted) and
every one passed all ten assertions.

Run artefacts: `results/20260807-080705_AP012051.1_gap1.json` (local),
`results/20260807-081420_...` (SLURM), `results/20260807-081809_...` (restricted).

```
  1. coordinator task completed                     state=completed
  2. acceptance gate verdict is ACCEPTED            verdict=ACCEPTED
  3. context tokens admitted > 0                    2942 tokens reached the model
  4. 1 <= candidates consumed <= offered            offered=17 consumed=17
  5. produced length within tolerance               872 vs 870 (+/-44)
  6. alphabet is a subset of ACGTN                  0 violations
  7. model input was not truncated                  3850 of 4096 tokens
  8. gap_length unchanged at every hop              870 at all three checkpoints
  9. retrieval returned row IDs, never nucleotides  all integers
 10. retrieval mode is correct                      unrestricted
 11. every tool call was one of the agent's own     18 calls, all mcp__
 12. retrieval pool was reused, not rebuilt         4 rounds, 18 rows, one pool of 40
```

Assertion 3 is the one the system exists for. The old pipeline could not have passed it:
it appended retrieved context *after* the sequence and then tokenized with
`truncation=True`, which removes from the end.

---

## 2. The token budget, measured

Gap1, contigs of 5,951 and 13,996 bases:

| | base input | free for context | outcome |
|---|---|---|---|
| `MODEL_FLANK=2000` (default) | 682 tokens | 3,236 | 2,942 admitted, input 3,850 of 4,096 |
| no trimming | 3,305 tokens | ~583 | a few records fit |
| the old harness's contig pair, untrimmed | **58,776 tokens** | none | 14x over the window |

That last row is the failure being detected, measured rather than argued about.
`gap-filler-agents/test.py` took `contigs[0]` and `contigs[1]`; the FASTA is sorted
longest-first, so those are `contig31` (190,651 bp) and `contig22` (167,069 bp) : two
contigs with no gap between them, passed alongside gap1's 870 bp target.

---

## 3. The negotiation edge worked

The Reconstruction agent measured 3,236 free tokens and then ran several
request/admit/extend rounds against the Retrieval agent, ending at 17 candidates and
2,942 tokens : 91% of the available budget. Under the old fixed `k=2` the same window
would have taken roughly 350 tokens, about 11% of it, with no report either way.

The retrieval pool persisting across A2A calls is what makes this cheap: `retrieve_context`
embeds the query once, and `extend_context` serves the next slice of the same ranking with
no further embedding work. Measured directly:

```
[retrieve] 12.7s  rows=[26907, 25672, 26199, 13778, 14871, 5327]  served_total=6
[extend]   11.0s  rows=[13014, 7156, 6729, 5877]                  served_total=10
```

---

## 4. Reproducibility

The local and SLURM runs produced **byte-identical** output sequences: same 872 bases,
same 54.0% GC, same 4.85 3-mer entropy, same 17 admitted row IDs. Given the same admitted
records, generation is deterministic : `torch.manual_seed` is set at agent startup.

That determinism is *conditional*. Which rows get admitted depends on the batch sizes the
model chooses, so it is not guaranteed across runs. The test asserts only on invariants,
never on sequence content.

---

## 5. Restricted retrieval

Restricting to *Fusobacterium animalis* (corpus rows 23070–25496):

| | unrestricted | restricted |
|---|---|---|
| candidates admitted | 17 | 27 |
| context tokens | 2,942 | 2,861 |
| every admitted row inside the organism block | n/a | yes |
| produced | 872 bases | 871 bases |

Over-fetch-and-post-filter was used, not `IDSelectorBatch`. I confirmed on this index
that a sparse `IDSelectorBatch` returns all `-1`: HNSW traversal starts from the entry
point and only accepts selected nodes, so a restrictive filter starves the greedy walk.
Raising `efSearch` does not fix it.

---

## 6. What the output actually looks like

This is where honesty matters more than the passing tests.

| run | length | GC % | 3-mer entropy |
|---|---|---|---|
| unrestricted | 872 | 54.0 | 4.85 |
| restricted | 871 | 12.6 | 4.38 |
| **the true gap** | **870** | **31.5** | **5.58** |

The predictions are the right length and are valid nucleotides, and they are not DNA.
Entropy 4.85 against a real 5.58 means the output is more repetitive than genuine
sequence, and both runs miss the true GC content badly : in opposite directions.

The cause is documented in `docs/gap_filling_known_bugs.md` and is **not fixed here**:
`predict_until_length` places hundreds of `[MASK]` tokens in a row and fills them all in a
single forward pass. Each mask mostly sees other masks, none can see what the others
became, and they collapse onto a few likely tokens. `tools/mlm.py` keeps that behaviour on
purpose so that the four changes it does make are attributable.

**So: the plumbing is fixed and measurable; the generator is not.** The system now proves
that retrieved context reached the model. Whether the model can *use* context of this kind
is a property of a masked language model, and no amount of orchestration moves it.

### Did it reconstruct the gap? No.

Identity against the true 870-base gap, scored with `Bio.Align.PairwiseAligner`
(match +1, mismatch −1, gap open −10, extend −1), **not** the `pairwise2.align.globalxx`
used elsewhere in this repository, which charges nothing for mismatches or gaps and rates
random DNA at 64%:

| run | length | identity | context tokens |
|---|---|---|---|
| local, unrestricted | 872 | 33.03% | 2,942 |
| 3-node, unrestricted | 872 | 33.03% | 2,942 |
| 3-node, unrestricted (rerun) | 872 | 34.83% | 2,942 |
| 3-node, restricted | 871 | **42.17%** | 2,861 |
| *random DNA, same length* | 870 | *35.23%* | n/a |
| *the true gap against itself* | 870 | *100.00%* | n/a |

The unrestricted runs sit **below the random floor**. The restricted run sits seven
points above it, which looks like a signal, and is not one:

| sequence | GC % | identity |
|---|---|---|
| restricted prediction | 12.6 | **42.17%** |
| **random DNA at the same 12.6% GC** | 12.6 | **42.02%** |
| unrestricted prediction | 54.0 | 33.03% |
| random DNA at the same 54.0% GC | 54.0 | 34.99% |
| random DNA matched to the true gap's 31.5% GC | 31.5 | 40.24% |

The restricted prediction beats composition-matched random by 0.15 percentage points.
The unrestricted one loses to it by 2. **Neither run recovered any sequence information
beyond base composition.** The apparent advantage of restricted retrieval was entirely an
artefact of the output drifting AT-rich against an AT-rich target, which is also why
uniform random at 50% GC scores worse than random at the true composition. Any identity
number on this task has to be read against a composition-matched floor, not a uniform one.

This is the ceiling the design predicted: GENA-LM is a masked language model, and it
treats retrieved context as more tokens rather than as auxiliary evidence it can attend
to. No amount of orchestration moves that.

What the system did change is which explanation is available. Before, a disappointing
score had three candidate causes (retrieval was poor, the context never reached the
model, or the model cannot use context of this kind) and the outputs could not separate
them. The second is now ruled out by measurement: 2,942 tokens demonstrably reached the
generator, with `truncated=False`. That leaves the first and third, and makes them
testable.

---

## 6b. Did DNABERT-S retrieval work?

The machinery: **yes, verifiably.** The biology on this query: **rank 1 only.**

### The index is sane

Searching the index with a corpus record's own sequence returns that record at cosine
1.000, and its nearest neighbours are same-genus:

| query row | organism | top-3 returned |
|---|---|---|
| 0 | *Candidatus* Nealsonbacteria | itself **1.000**, Nealsonbacteria 0.637, F. polymorphum 0.633 |
| 12500 | *Staphylococcus aureus* | itself **1.000**, S. aureus 0.826, S. aureus 0.794 |
| 23500 | *Fusobacterium animalis* | itself **1.000**, F. polymorphum 0.893, F. polymorphum 0.887 |
| 27000 | *Fusobacterium polymorphum* | itself **1.000**, F. animalis 0.917, F. polymorphum 0.866 |

Exact self-match at 1.000 confirms the embedder, the L2 normalisation, the cosine
conversion and the row-ID → record mapping are all correct end to end. Cross-species
neighbours ranking at 0.89–0.92 between the two *Fusobacterium* species confirms the
contrastive training is doing what the benchmark said it would.

### On the real gap1 query it is right once, then drifts

AP012051.1 is a *Fusobacterium*; the corpus carries *F. animalis* and *F. polymorphum*,
4,926 of 43,575 records : an 11.3% prior.

| | rank-1 hit | Fusobacterium among admitted |
|---|---|---|
| unrestricted | **Fusobacterium** (correct) | 1 of 17 = **5.9%**, *below* the 11.3% prior |
| restricted | Fusobacterium | 27 of 27 = 100% |

The top-60 hits for the gap1 query are dominated by the wrong genus: 20 *S. aureus*,
14 *Ca.* Nomurabacteria, 8 *C. psittaci*, 6 *Ca.* Peregrinibacteria. So precision@1 is
fine (consistent with the 0.655 the embedder benchmark measured) and precision@k
collapses below chance as k grows. The agent admits 17 candidates, so it is operating in
the regime where the ranking has already degraded.

### The likely cause is composition, and it is actionable

| | mean GC |
|---|---|
| AP012051.1 contigs (query flanks) | 33.7% left, 36.1% right |
| *Staphylococcus aureus* CDS | **32.5%**: closest of any organism in the corpus |
| *Fusobacterium animalis* CDS | 26.9% |
| *Fusobacterium polymorphum* CDS | 26.1% |

The organism the search prefers is the one whose base composition is nearest the query,
not the one that is homologous. Beyond rank 1 the embedding appears to be tracking
composition more than sequence relatedness.

There is a domain mismatch underneath this that is worth fixing before blaming the model:
**the corpus is coding sequences only, while the query is raw contig** including
intergenic DNA. In an AT-rich organism, coding regions run several points higher in GC
than the genome average, which is exactly the 33.7% vs 26.9% gap seen here. The query and
the corpus are not drawn from the same distribution.

Three things follow, in order of cost:

1. **Restriction rescues it completely** (27 of 27 correct) and is one field in the
   request. When the organism is known, use it.
2. **Query with CDS-like sequence**, or build a corpus that includes intergenic regions,
   so query and corpus come from the same distribution.
3. **The metadata index** measured at precision@1 0.906 against 0.655 for the DNA index
   (`embedder_benchmark/bridge_architecture.md`). The Retrieval agent already carries
   organism and protein metadata per record for exactly this.

This also refines the earlier finding. The generator was handed 2,942 tokens of context
that was, in the unrestricted case, mostly the wrong organism. So "the generator cannot
use retrieved context" is established for *this* context; it has not been tested with
context that is actually homologous.

---

## 7. Test coverage

```
154 passed, 3 skipped in 3.2 s          (no LLM, no models)
```

| File | Covers |
|---|---|
| `test_tool_contracts.py` | Every precondition path: out-of-order calls, mutated `gap_length`, wrong `free_tokens`, exhausted budgets, peer failure, pool reuse |
| `test_a2a.py` | Agent cards, both well-known paths, health, `message/send`, `tasks/get`, all five JSON-RPC error codes, declined methods, terminal-task immutability, store eviction |
| `test_gate.py` | The gate, including a perfect sequence rejected for zero admitted context |
| `test_flank_selection.py` | Coordinate-based flank matching for all 32 interior gaps, plus an explicit assertion of the `contigs[0]`/`contigs[1]` trap |
| `test_corpus_store.py` | Offset round-trip, metadata-without-nucleotides, fingerprint stability |
| `test_naming.py` | Tool-id coupling; that no tool accepts a sequence |
| `test_mlm.py` | Context-first assembly, the measured mask seed, plus two heavy end-to-end tests against the real models |
| `test_deploy_parsing.py` | The two `deploy.sh` shell parsers, run as real bash against the exact salloc output and `scontrol` layout that broke them |
| `test_agent_runtime.py` | Tool confinement (both `tools` and `allowed_tools`, no shell or filesystem tool reachable, model pinned) and retrieval-pool persistence across A2A requests |

Each of the four bring-up defects in section 8 now has a regression test, and each test
was verified by reintroducing the bug and confirming it fails:

| defect reintroduced | result |
|---|---|
| `read -ra` node collapse | 2 tests fail |
| job id scraped from the node range | 4 tests fail |
| `tools` removed from the SDK options | 9 tests fail |
| `persist_state=False` on retrieval | 3 tests fail |

`test.py` gained two end-to-end assertions covering the same ground on a live run:
**11** every tool call across all three agents is an `mcp__` id, and **12** one ranked
pool served the whole negotiation (`pool_size` constant while `served_total` climbs).

The three skipped tests are `heavy`-marked; `GENOMIO_HEAVY=1` runs them and they pass:

```
base=682 free=3236 admitted=1335 tokens (8 records) input=2285 produced=866/870 attempts=10
untrimmed contig31+contig22 = 58776 tokens vs a 4096 window
```

---

## 8. Defects found and fixed during bring-up

Recorded because each one would have been invisible in a passing run.

**The model refused after the tool had already run.** A retrieval turn ended with
`API Error: Sonnet 4.5 can't help with this`, and the task still completed correctly,
because the reply is serialised from state rather than parsed from the model's text.
Removing the 1,800-base query string from the prompt stopped the refusal recurring and cut
the call from 46 s to 13 s. The query now lives in agent state; the model chooses *when*
to search, not *what* to search for.

**`allowed_tools` alone did not confine the model.** The first smoke run shows a
`ToolSearch` call the agent never registered. Setting `tools` as well as `allowed_tools`
restricts the built-in set too.

**`read -ra` silently co-located a three-node fleet on one node.** `scontrol show
hostnames` prints one host per line and `read -ra` reads only the first, so `deploy.sh`
reported "1 node" for a three-node allocation. `mapfile -t` fixes it. The first deployment
looked entirely successful.

**The retrieval pool was being destroyed between calls.** `handle_message` dropped task
state when a request finished, which would have made `extend_context` rebuild the ranking
every time, or fail with `NO_POOL`. Retrieval sessions now persist by `request_id`.

**gap33 has no right-hand flank.** It runs 1557603–1558103 while the assembly's last
contig ends at 1557603. 32 of 33 gaps are reconstructable; the terminal one is declined
rather than filled against an invented flank.

---

## 9. Cost and timing

| | |
|---|---|
| wall clock per gap | 210–250 s |
| assistant turns per run | ~14 across the three agents |
| coordinator startup | ~2 s (no models) |
| retrieval startup | ~30 s cold (DNABERT-S), ~6 s warm |
| reconstruction startup | ~3 s |
| resident memory | ~1.0 GB retrieval, ~1.2 GB reconstruction, ~120 MB coordinator |

Roughly 60–120k input tokens per run. Billing is against the CLI subscription, not API
credits.

---

## 10. What would come next

1. **Fix the generator.** This is now the only thing standing between the system and a
   real result. Block-by-block filling at one mask per pass, so each token is written
   while seeing the previous ones. `docs/gap_filling_known_bugs.md` measures that at
   roughly ten minutes per 870-base gap instead of one.
2. **Score against a composition-matched floor, not a uniform one.** Section 6 shows why:
   random DNA at the true gap's 31.5% GC scores 40.24%, while uniform 50% GC random
   scores 36.00%. A prediction that merely matches base composition can clear a uniform
   floor by four points while recovering nothing. Any future metric module should report
   the composition-matched baseline beside every identity number.
3. **Fix the identity metric.** `pairwise2.align.globalxx` charges nothing for mismatches
   or gaps and rates random DNA at 64%, above every real result. Until that is replaced,
   no quality number in the older results files means anything.
3. **More samples.** Every number here is one run per condition.
4. **The metadata index.** `embedder_benchmark/` measures precision@1 of 0.906 for a text
   index over record metadata against 0.655 for the DNA index. The Retrieval agent already
   carries organism and protein metadata for exactly this.
