# DNABERT-S retrieval: what changed and why

This document explains a change to how GenomIO finds reference sequences before it fills a
gap. It assumes you know what a DNA sequence is and nothing else.

There are two parts. One is a straightforward swap of one model for another. The other
looks small in the diff but changes how the agent finds anything at all, and that is the
part worth reading carefully.

---

## 1. The idea in one page

To fill a gap in a draft genome, GenomIO first looks for **reference sequences that
resemble the region around the gap**, and then hands those to a language model that
predicts the missing nucleotides. This document is only about the first step, the looking.

To find "similar" sequences quickly, each sequence in the reference library is turned into
a list of 768 numbers, called an **embedding**. Sequences that are alike should get
similar lists. Searching then means finding the closest lists, which a library called
FAISS does in microseconds, instead of comparing millions of letters.

Everything depends on the model that produces those numbers. Two things about it changed.

### The reference library: `rag_corpus` became `rag_corpus_uniform`

The old library, `rag_corpus/`, was assembled by a script that asked NCBI one question per
species and downloaded whichever file came first alphabetically. That mostly worked, and
failed silently twice. One species, *Candidatus* Nealsonbacteria, returned no results at
all and was skipped with no file and no error, so it simply was not in the library. Another,
*Candidatus* Woesebacteria, got a file of whole assembled contigs, 10,000 to 200,000 bases
each, instead of individual genes. Nothing in the log distinguished that from success.

`rag_corpus_uniform/` is built by `scripts/download_genomes_uniform.sh`, which asks three
questions per species instead of one, keeps up to 15 candidate assemblies, and works down
the list until it finds a real gene level file. All 20 species now resolve, and every
record is **one gene, roughly 300 to 3,000 bases**, as annotated by NCBI.

That matters for a reason that is easy to miss: a gap is typically a few hundred to a few
thousand bases. If your reference records are whole chromosomes, you are comparing a gap to
a chromosome. If they are genes, you are comparing like with like.

It also means **no chunking**. The old code cut every record into 1,000 character pieces
with a 200 character overlap, because it had no idea where meaningful boundaries were. Now
every record boundary is already a gene boundary, so the chunking code is gone.

The corpus is committed to the repository, so nobody has to re-run the NCBI download.

### The embedder: `all-MiniLM-L6-v2` became `DNABERT-S`

`all-MiniLM-L6-v2` is a good model for English sentences. It was being used on strings like
`ATGAGCGCCGTGCTGGTGACC`, which are not English sentences. The numbers it produced carried
essentially no biological information.

`embedder_benchmark/` tested 16 DNA models. Measured across 1,000 sequences from 20
species, `DNABERT-S` wins:

| Rank | Model | Silhouette | Inter species distance | Training objective |
|---|---|---|---|---|
| 1 | **DNABERT_S** | **0.0486** | **0.646** | Contrastive |
| 2 | DNABERT_1 | 0.0236 | 0.160 | Masked language modelling |
| 7 | DNABERT_2 | 0.0029 | 0.215 | Masked language modelling |
| 9 to 12 | Nucleotide Transformer v2, 50M to 500M | negative | 0.98 to 0.99 | Masked language modelling |

**Silhouette** answers: is a sequence, on average, closer to sequences from its own species
than to sequences from any other? Above zero means yes. Below zero means the search would
be worse than useless. Thirteen of the sixteen models scored at or below 0.015.

The reason DNABERT-S wins is not that it is bigger. It has 110 million parameters, 23 times
fewer than the largest model tested, which it beats by a factor of five. It wins because of
how it was trained. Most DNA models are trained to predict hidden nucleotides, which makes
them good at nucleotides and indifferent to whether two sequences from the same organism end
up near each other. DNABERT-S was trained **contrastively**: shown pairs of sequences and
taught to pull same species pairs together and push different species pairs apart. That is
precisely the property a search needs.

The DNABERT family makes the point cleanly, since the three variants differ one change at a
time. DNABERT_1 to DNABERT_2 changed only the tokenizer, and the score got **8 times
worse**. DNABERT_2 to DNABERT_S kept that tokenizer and added contrastive training, and the
score got **16 times better**.

---

## 2. The easy half: `src/core/gap_filler_rag.py`

This file compares gap filling with and without retrieval. Here, the change really is just
an embedding swap. There is no hidden complexity, so do not go looking for any.

```
before:  cut records into 1000 char chunks -> all-MiniLM-L6-v2 -> FAISS -> top 2 DNA strings
after:   one record per gene, no chunking  -> DNABERT-S        -> FAISS -> top 2 DNA strings
```

What comes out is the same shape it always was: **plain nucleotides, still two of them**.
The records now carry organism and protein names, but none of that is returned here and
none of it reaches the gap filling model, because none of it was there before. How the gap
is actually predicted is completely untouched.

One real bug got fixed on the way. The query used to be `left_seq + right_seq`, the two
whole contigs, often over 100,000 bases. Every embedding model truncates its input, so what
was actually being searched for was the first fragment of the left contig and nothing at all
from the right one. The query is now the **900 bases either side of the gap**, which is the
part that borders it and matches the length range the benchmark was run on. MiniLM had the
same problem, so this is an old bug rather than a new one, but the swap would have been
meaningless without fixing it.

---

## 3. The hard half: the agent pipeline

This is the part that needs explaining, because **the old agent code contained no embedder
at all**. "Swap the embedder" does not describe what happened here.

### The call chain

```
gap-filler-agents/test.py
  builds "<50 bases>---<50 bases>" from two contigs, calls plan()
    |
    v
  src/agents/planner.py
    a LangChain agent on Gemini 2.5 Flash, forced to call two tools in order
      |
      +-- 1. context_tool ---> src/rag/retriever.py      <-- THIS is what changed
      |
      +-- 2. gap_filler_tool -> src/rag/gap_filler.py    (GENA-LM, untouched)
```

### What `retriever.py` used to do

It read every file in `rag_corpus/`, all 52 MB of it, from disk. On every single call. Then
it used Python's `in` operator: a record was a match only if the query flank appeared inside
it **character for character**.

That is not a similarity search, it is a substring search. A reference gene 98% identical to
the query scored exactly the same as one sharing nothing: no match. In practice the tool
mostly returned "no matching sequence found", and when it did return something, it was
because of an exact repeat.

### What it does now

It embeds the flanks either side of each gap with DNABERT-S, searches the FAISS index by
cosine similarity, and returns the closest records. A record comes back because it **looks
biologically like** the query.

| | before | after |
|---|---|---|
| How a match is found | Python `in`, the flank must appear letter for letter | cosine similarity between DNABERT-S embeddings |
| What it reads | all of `rag_corpus/`, re-read from disk every call | a FAISS index built once and cached |
| What "similar" means | identical substring, or nothing | biologically related, usually same species |
| Speed per call | seconds of file parsing | microseconds of index lookup |
| Return value | `"Match 1:\n{header}\n{sequence}"` | **unchanged, deliberately** |

### Why no agent code changed

That last row is the point. `retrieve_context()` kept its exact signature and its exact
output format, so `planner.py` and `planner_tools.py` did not need a single line changed.
The agent asks the same question and gets an answer in the same shape; only the way the
answer is found is different.

"We replaced retrieval and touched no agent code" is the least obvious thing about this
change, which is why it gets a heading.

### Flanks have to be long enough

Substring search and similarity search want opposite things from the query. A short flank
is *easier* to find verbatim, so 50 bases was a reasonable default before. An embedding has
to characterise a sequence rather than locate it, and 50 bases is not enough to characterise
anything.

Measured over 100 balanced queries against the full corpus, where 0.05 is chance across the
20 species:

| Query flank | Precision at rank 1 |
|---|---|
| 50 bp | 0.170 |
| 120 bp | 0.290 |
| 300 bp | 0.520 |
| **600 bp** | **0.780** |

`gap-filler-agents/test.py` was building its query with 50 base flanks, which would have
left the agent retrieving near noise. Its default is now 600, which also sits inside the
300 to 900 bp band the benchmark was run on, and gives the gap filling model more flanking
context. `MAX_FLANK` in `retriever.py` caps queries at 900, because DNABERT-S truncates
beyond roughly 2,000 bases and the part nearest the gap is the part that matters.

The effect measured here is on retrieval. Whether a longer flank also improves the filled
sequence has not been measured.

### One knock on effect to know about

`build_masked_input_with_context()` in `src/rag/gap_filler.py` takes the retrieved records
and looks for the query flank **inside** them, using `match.index(start)`, so it can trim
the surrounding context and shrink the gap accordingly.

Under substring retrieval that always worked, because retrieval had already guaranteed the
flank was in there. Under similarity retrieval it usually will not be found: a record is
returned for being alike, not for containing the query. The existing error handler catches
that and the gap length stays at its full value.

Nothing needs fixing. That is correct behaviour for a similarity search, and the code
already handles it. But the gap filling model does see something different now, so it is
worth knowing before someone reads it as a regression.

---

## 4. Running it

The index is about 130 MB, past what GitHub accepts in a file, so it is **not committed**.
It is built once and cached under `.cache/rag_index/`, which is ignored by git.

```bash
python3 scripts/build_rag_index.py            # build it, once
python3 scripts/build_rag_index.py --check    # is the cache present and current?
python3 scripts/build_rag_index.py --rebuild  # force a rebuild
```

Expect **one to a few hours on CPU** for all 43,575 records, depending on how busy the
machine is. It only needs doing on a fresh checkout, or after the corpus changes. Every run
after that loads the cache in seconds.

The cache remembers which corpus files it was built from and how large they were. Change or
replace a corpus file and the next run rebuilds instead of quietly serving embeddings that
no longer line up with the records. This matters more than it sounds: a FAISS search returns
row numbers, not sequences, so if the vectors and the record list ever drift apart, every
result is confidently wrong rather than obviously broken.

The corpus itself is committed, so `scripts/download_genomes_uniform.sh` never has to be
re-run. Run it only to refresh the corpus from current NCBI assemblies.

Then, as before:

```bash
python -m gap-filler-agents.test     # the agent pipeline, needs GOOGLE_API_KEY in .env
python src/core/gap_filler_rag.py    # the with and without retrieval comparison
```

---

## 5. What did not ship

Each record has two kinds of content: the DNA, and a metadata line naming the organism, the
gene and the protein. The benchmark found that a **second** index over that metadata text,
searched with a sentence encoder, retrieves the right species considerably more often than
the DNA index does: precision at rank 1 of 0.906 against 0.655.

That is not in this change. It needs a second model, a second index and a way to route
queries between them, all of which is designed in `embedder_benchmark/bridge_architecture.md`
and left for later work. What shipped is the DNA index only.

Also worth knowing: DNABERT-S must **not** be used on the metadata text. Its tokenizer only
understands nucleotides, so English words become unknown tokens and every metadata string
collapses to nearly the same vector. Measured at scale, its silhouette on metadata is
**negative**, meaning records land closer to other species' metadata than to their own.
Experiments 3 and 4 in `embedder_benchmark/` cover this.
