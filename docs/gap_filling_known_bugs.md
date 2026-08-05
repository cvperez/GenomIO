# Three bugs in gap filling, and what fixing them would involve

This document describes three problems in the gap filling code. It is written for someone
who has not read the code. Each section says what is wrong, shows the evidence, then shows
the code as it stands today and the code that would replace it.

**Nothing here is applied.** The code in the repository still contains all three bugs. This
is a record of what was found and measured, so the fixes can be made deliberately later.

The short version: the pipeline produces repetitive nonsense, the retrieved reference
sequences are thrown away before the model ever sees them, and the score used to measure
quality rates random DNA higher than every real result. The three are independent, and all
three have to be fixed before any gap filling number can mean anything.

**The numbers in `results_AP012051.1_*.csv` should not be used.** They were produced with the
broken score. Re-scored properly, every one of them lands below the level you would get by
guessing at random. There is a table at the end.

---

## Bug 1: the score rates random DNA above every real result

### What is wrong

To judge a prediction, the code compares it to the true gap sequence and reports a
percentage. It uses `pairwise2.align.globalxx`. The `xx` matters: that function counts
matching letters while charging **nothing** for a mismatch and **nothing** for a gap.

With gaps free, the aligner can slide the two sequences around as much as it likes,
inserting space wherever that helps, until it has collected every letter that happens to
agree anywhere. Two unrelated sequences share a lot of letters that way, simply because DNA
only has four of them.

So the number being reported is not identity. It is the length of the longest common
subsequence, divided by the length of the longer sequence.

### The evidence

Compared against a real 870 bp gap:

| What was compared | Score reported |
|---|---|
| Two random DNA sequences | **64.1%** |
| A shuffled copy of the true gap | **64.9%** |
| The pipeline's actual gap1 prediction | 56.9% |
| The pipeline's actual gap2 prediction | 54.2% |
| The pipeline's actual gap3 prediction | 44.0% |

Random sequence scores higher than anything the pipeline has ever produced. A score whose
floor sits above your results cannot tell you anything.

### The code today

In `src/core/gap_filler.py` and again in `src/core/gap_filler_rag.py`:

```python
from Bio import SeqIO, pairwise2

alignments = pairwise2.align.globalxx(prediction, real_gap_seq)
identity = 0.0
if alignments:
    best = alignments[0]
    score = best.score
    align_len = max(len(prediction), len(real_gap_seq))
    identity = score / align_len * 100
```

There is a second problem hiding in the first line. Without `one_alignment_only=True`,
`globalxx` builds **every** equally good alignment. With free gaps and a four letter alphabet
there are an enormous number of them, so this is both the slowest step in the pipeline and a
way to run out of memory.

### The proposed replacement

A new `src/core/metrics.py`, as the single home for sequence comparison:

```python
from Bio.Align import PairwiseAligner

MATCH_SCORE = 1.0
MISMATCH_SCORE = -1.0
OPEN_GAP_SCORE = -10.0
EXTEND_GAP_SCORE = -1.0

def sequence_identity(predicted, reference):
    """Percent of alignment columns where both sequences carry the same base."""
    alignment = _get_aligner().align(predicted, reference)[0]
    top, bottom = alignment[0], alignment[1]
    matches = sum(1 for a, b in zip(top, bottom) if a == b and a != "-")
    return 100.0 * matches / len(top)
```

and at both call sites:

```python
identity = sequence_identity(prediction, real_gap_seq)
```

Mismatches would cost something, and gaps would cost a lot. The heavy gap penalty is
deliberate: a prediction is compared against a true gap of nearly the same length, so there
is no reason to expect insertions or deletions, and cheap gaps are exactly the loophole that
lets the current score invent matches out of unrelated sequence.

Measured behaviour of the replacement, tested against a real 870 bp gap:

| What was compared | Current score | Proposed score |
|---|---|---|
| Identical sequences | 100.0% | 100.0% |
| 90% identical | not measured | 93.2% |
| 70% identical | not measured | 77.4% |
| Random DNA | **64.1%** | **34.6%** |
| A repeating motif | not measured | 33.9% |

The 70% row is the proof that this is calibrated. If you randomly change 30% of the letters,
70% stay the same and a quarter of the changed ones land back on the original letter by luck,
so the true answer is about 77.5%. The proposed score says 77.4%.

The floor is not zero, and that is expected. With four letters, two unrelated sequences agree
at roughly a quarter of positions before alignment does anything.

Speed is not a concern either: 0.03 s for 870 bp and 0.71 s at 80 MB for 4,867 bp, far
cheaper than the current call.

### Report the floor next to every result

Nobody noticed the current score was broken because nobody knew where its floor was. So the
module should also provide:

```python
def random_baseline_identity(length, trials=3):
    """What random DNA of this length scores. Print it next to any result."""
```

with a `random_baseline_percent` column written beside `identity_percent` in every results
file. If the two are close, the prediction is worth nothing, and that should be visible
without anyone having to go and check.

---

## Bug 2: the retrieved context is thrown away before the model sees it

### What is wrong

The RAG version of the gap filler retrieves reference sequences and passes them to the model
along with the sequence being filled. It adds them at the **end** of the input.

Language models have a maximum input size. GENA-LM takes 4,096 tokens, and anything longer is
cut off **from the end**. So the retrieved context is always first in line to be deleted.

It is worse than that, because the contigs either side of the gap are passed whole. One of
them, contig4, is 41,748 bases, which is 6,792 tokens by itself. The input is already far over
the limit before any context is added.

### The evidence

Counting tokens for each gap against the 4,096 limit:

| Gap | Tokens before context | What happens to the 337 tokens of context |
|---|---|---|
| gap1 | 3,580 | kept in full |
| gap2 | 4,032 | 64 kept, 273 discarded |
| gap3 | 7,392 | all discarded, input already over the limit |

The results confirm it. Gap3's run with RAG and run without RAG came out identical to two
decimal places, 43.95 against 43.96, with the same GC content and the same entropy. They were
identical because for gap3 the two runs were literally the same input.

### The code today

```python
def build_masked_input(left_seq, right_seq):
    return f"{left_seq} [MASKS] {right_seq}"

def build_masked_input_rag(left_seq, right_seq, context=None):
    base = f"{left_seq} [MASKS] {right_seq}"
    return f"{base}\n\n# Context: {context}" if context else base
```

### The proposed change

```python
MODEL_FLANK = 2000

def build_masked_input(left_seq, right_seq):
    return f"{left_seq[-MODEL_FLANK:]} [MASKS] {right_seq[:MODEL_FLANK]}"

def build_masked_input_rag(left_seq, right_seq, context=None):
    base = build_masked_input(left_seq, right_seq)
    return f"{context} {base}" if context else base
```

Three changes:

1. **The context goes first.** Cutting happens at the end, so whatever is at the front
   survives.
2. **The contigs get trimmed** to 2,000 bases either side. Those are the bases that actually
   border the gap, so they are the informative ones. The retrieval query is already trimmed
   this way; the model input is not.
3. **The `# Context:` label goes.** It is English prose being handed to a tokenizer that only
   understands nucleotides, which has no way to read it as anything but broken DNA.

In `src/core/gap_filler.py` the trimming was in fact already written, and then left commented
out:

```python
left = left_seq # [-1000:] if len(left_seq) > 1000 else left_seq
right = right_seq # [:1000] if len(right_seq) > 1000 else right_seq
```

### What it would achieve

Measured with the proposed version:

| Gap | Tokens now | Tokens after, with context | Fits? |
|---|---|---|---|
| gap1 | 3,917 | 1,041 | yes |
| gap2 | 4,369 | 998 | yes |
| gap3 | 7,729 | 992 | yes |

All three gaps would fit with room to spare and all three would keep their context. Comparing
"with RAG" against "without RAG" would become a real comparison for every gap, rather than
only for gap1.

---

## Bug 3: the generator fills every mask in one go

### What is wrong

This is the one that produces the actual nonsense.

To fill a gap of 870 bases the code puts about 282 `[MASK]` markers in a row and asks the
model to fill all of them in a **single pass**.

A masked language model works out what belongs in a masked position by looking at the text
around it. When you put hundreds of masks side by side, each one looks around and mostly sees
other masks. None of them can see what any of the others turned into, because they are all
decided at the same moment from the same snapshot. With nothing to tell them apart, they all
fall towards the same few most likely tokens.

The result is not DNA. It is the same short pattern repeated.

### The evidence

The first prediction for gap1 begins:

```
GTCGTCGTCGACGTTCGTCGTTCGACGTCGTCGTCGTCGAAGACGTTCGTCGACGACGAATTACGTCG...
```

against a true gap that begins:

```
TTTTCATTGATTCTTTAAGTCTTTTGTAGTGGTCATATGCCCTAAAAATTGCAGTGCCCAATTTTGTT...
```

The prediction is 60% GC. The real gap is 31% GC. Measuring how varied a sequence is, using
3-mer entropy where 6.00 is the maximum, random DNA is 5.96 and the real gaps are 5.58 to
5.73:

| Gap | Without RAG | With RAG | The real gap |
|---|---|---|---|
| gap1 | 3.63 | 3.39 | 5.58 |
| gap2 | 2.65 | **0.98** | 5.73 |
| gap3 | 2.36 | 2.36 | 5.63 |

An entropy of 0.98 means the output is very nearly a single repeated letter.

There is a second symptom. Because a decoded token can be anywhere from one to about ten
bases, the only way to hit a target length is to guess a number of masks, see what comes out,
and guess again. For an 870 base gap that settles after about six tries. For the 4,867 base
gap it never settles at all:

```
attempt 6: 1392 masks -> 4275 bases   (target 4867)
attempt 7: 1490 masks -> 5860 bases
attempt 8: 1242 masks -> 4046 bases
attempt 9: 1378 masks -> 4261 bases
attempt 10: 1479 masks -> 5821 bases   gives up, returns the closest attempt
```

Every sample of that gap uses up the full ten attempt budget and still misses.

### The code today

```python
def predict_until_length(masked_input_base, tokenizer, model, gap_length, max_attempts=10):
    attempt = 0
    mask_count = int(gap_length / 4)
    best_seq, best_diff = "", float("inf")

    while attempt < max_attempts:
        masked_gap = " ".join(["[MASK]"] * mask_count)          # all masks at once
        masked_input = masked_input_base.replace("[MASKS]", masked_gap)
        outputs = model(**tokenizer(masked_input, ...))          # one single pass

        predicted_tokens = [sample(outputs.logits[0, i]) for i in mask_positions]
        predicted_seq = "".join(predicted_tokens).replace("▁", "")

        diff = abs(len(predicted_seq) - gap_length)
        if diff <= 3:
            return predicted_seq
        # otherwise guess a different mask_count and try the whole thing again
        mask_count += adjustment if len(predicted_seq) < gap_length else -adjustment
        attempt += 1

    return best_seq
```

### The proposed change

A new `src/core/mlm_filling.py`, replacing all three copies of the function:

```python
BASES_PER_TOKEN = 3.0
DEFAULT_BLOCK_SIZE = 1
MAX_EMPTY_ROUNDS = 10

def fill_gap(masked_input_base, tokenizer, model, gap_length, block_size=1, ...):
    produced = ""
    empty_rounds = 0
    while len(produced) < gap_length:
        remaining = gap_length - len(produced)
        n_masks = max(1, min(block_size, ceil(remaining / BASES_PER_TOKEN)))

        # everything generated so far becomes ordinary left context for this block
        visible = produced[-MAX_PRODUCED_CONTEXT:]
        masks = " ".join(["[MASK]"] * n_masks)
        masked_input = masked_input_base.replace("[MASKS]", f"{visible} {masks}")

        outputs = model(**tokenizer(masked_input, ...))
        block = decode(sample(outputs.logits[0, i]) for i in mask_positions)

        if not block:
            # one empty block is bad luck, not a stall; only stop after several
            empty_rounds += 1
            if empty_rounds >= MAX_EMPTY_ROUNDS:
                break
            continue

        empty_rounds = 0
        produced += block

    return produced[:gap_length]
```

The gap gets filled a block at a time. The important line is the one that puts `produced` back
into the input: **block two is written while looking at block one**, block three while looking
at blocks one and two, and so on. That is what the single pass version cannot do.

The way tokens are chosen would stay exactly as it is, so any difference in output comes from
the block structure and nothing else.

### The block size is the setting that matters, and it has to be 1

Filling in blocks is not on its own enough. A first attempt with 32 masks per pass still
produced a repeat. Measured on a 150 base target, where the real sequence has entropy 5.30 and
34.0% GC:

| Masks per pass | Entropy | GC | What the output looks like |
|---|---|---|---|
| **1** | **4.53** | 25.5% | `TAAAAGAGCTAGTGATACTGATATAGCAATAATGATACAGATAAT` |
| 4 | 3.80 | 44.7% | `GACGACGTCACGACGTTAGGGCGGACGTTAGGGCGACGACGACGC` |
| 8 | 2.42 | 62.7% | `ACGACGACGACGACGACGACGTACGACGACGACGACGACGACGAC` |
| 32 | 2.01 | 66.0% | `GACGACGACGACGACGACGACGACGAAGACGACGACGACGACGAC` |

The trend is steep and goes one way. Only one mask per pass produces anything that reads like
DNA, and it is the only setting that lands near the real GC content instead of drifting GC
rich.

The reason is the same reason the current code fails, just at a smaller scale. Masks in the
same pass cannot see each other. Four masks together is already four bases with no knowledge
of one another, and the errors compound because the model then reads its own output back as
context and continues the pattern it just created. A block of 32 masks was doing a small
version of exactly what a block of 282 does.

There is also a training mismatch. Masked language models are trained with about 15% of tokens
hidden and scattered around. A run of 32 consecutive masks is nothing the model ever saw while
learning.

So one mask per pass, which costs one forward pass per token. That is the honest price of
filling a gap this way.

### What it would achieve

Run against the real 870 base gap1, at one mask per pass:

| | Current code | Proposed |
|---|---|---|
| Length produced | 872 of 870, after 6 tries | **870 of 870, exactly, first time** |
| Entropy | 3.63 | **4.25** (real gap 5.58) |
| Identity, scored properly | 32.83% | **40.52%** |
| Random floor | 35.46% | 35.46% |

That last pair of rows is the point. The current code scores **below** the random floor. The
proposed version scores **above** it, by about five points. That is the first time anything in
this pipeline has produced a gap fill measurably better than guessing.

It is a modest result and it comes from a single sample, so the five point margin needs more
samples before anyone leans on it. Entropy 4.25 against a real 5.58 also says the output is
still not really DNA. But it is above the floor, which nothing here has managed before.

One detail worth keeping: the `MAX_EMPTY_ROUNDS` retry matters more than it looks. An earlier
version stopped at the first block that decoded to no bases, which truncated the same 870 base
gap at 186 bases and dragged the score down to 11.26%. Occasionally the model picks a token
carrying no nucleotides, and that is bad luck rather than a stall.

### It is slower

One forward pass per token means an 870 base gap takes about ten minutes rather than about one.
The 4,867 base gap would take roughly an hour per sample. Running the full three gap, three
sample, two arm comparison would be six to twelve hours rather than the one to two it takes
now.

---

## One implementation, not three

`predict_until_length` exists in three copies, in `src/core/gap_filler.py`,
`src/core/gap_filler_rag.py` and `src/rag/gap_filler.py`. They differ only in their comments
and print statements; the logic is identical in all three. The same is true of the scoring
code, which exists twice.

Fixing a bug three times is how the copies drifted apart in the first place, so the fixed
versions belong in one place each:

| Module | What it would hold | Used by |
|---|---|---|
| `src/core/metrics.py` | `sequence_identity`, `random_baseline_identity` | both core gap fillers |
| `src/core/mlm_filling.py` | `fill_gap` | both core gap fillers and the agent |

The agent path would pick up the fix through the same module, which is the point.

---

## What the existing results really are

The predictions from the last run are still on disk, so they can simply be re-scored with the
corrected metric. No re-running needed:

| Gap | Arm | Score as reported | Corrected score | Random floor |
|---|---|---|---|---|
| gap1 | no RAG | 56.92% | 32.83% | 35.46% |
| gap2 | no RAG | 54.21% | 34.61% | 37.03% |
| gap3 | no RAG | 43.95% | 33.18% | 36.86% |
| gap1 | with RAG | 55.46% | 31.26% | 35.46% |
| gap2 | with RAG | 37.71% | 33.58% | 37.03% |
| gap3 | with RAG | 43.96% | 33.23% | 36.86% |

**Every single one is below its own random floor.** The pipeline as it stands is not filling
gaps at all, in either arm.

It also shows how misleading the current score is about the comparison. Gap2 with RAG looks
dramatically worse than without, 37.71% against 54.21%, which reads like retrieval actively
causing harm. Corrected, the two are 33.58% and 34.61%, both noise. The apparent collapse is an
artefact of scoring a low variety sequence with a metric that rewards variety.

---

## A fourth problem, recorded but not investigated further

`build_masked_input_with_context` in `src/rag/gap_filler.py`, which is the agent's path rather
than the standalone scripts, has a different flaw.

It already places retrieved context in the right spot, immediately either side of the masks,
which is better than what the RAG script does. But it finds that context by **exact text
search**: it needs the query flank to appear character for character inside a retrieved record.

That made sense when retrieval was itself an exact text search, because retrieval had already
guaranteed the flank was in there. Since retrieval became similarity based, a record comes back
because it *resembles* the query, not because it *contains* it. The search therefore fails every
time, the error is caught and ignored, and the function quietly returns no context at all.

The agent still runs. It simply gets no benefit from retrieval on that path. Fixing it means
aligning the flank against the record rather than searching for it.

Separately, `src/core/evaluation.py` cannot be imported at all: it needs the
`python-Levenshtein` package, which is not installed and is not listed in `requirements.txt`.

---

## Suggested order, if these get fixed

1. **The metric first.** Nothing else can be measured until it works, and it is the smallest
   change of the three.
2. **Context placement.** Small, self contained, and it makes the RAG comparison meaningful for
   all three gaps instead of one.
3. **The generator.** The largest change, the one that needs the most compute to verify, and
   the one whose benefit is currently a single sample worth five points above the floor.
