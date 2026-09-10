# rag/index.py
"""
The DNA retrieval index (Index A): DNABERT-S vectors of every CDS record in
rag_corpus_uniform/, stored in FAISS and searched by cosine similarity.

Vectors are L2 normalised before they go in, so ordering by Euclidean distance and
ordering by cosine similarity are the same ordering. IndexHNSWFlat searches with squared
L2 and that is what FAISS hands back, so search_many converts it to a cosine before
returning, since distances are better when smaller and cosines when larger.

HNSW is used rather than exact IndexFlatIP because the corpus has about 43,500 records
and the benchmark measured HNSW overtaking exact search at around 1,000, while still
agreeing with brute force on 100% of top-1 hits at that scale
(embedder_benchmark/experiment_8_faiss_scale/results/experiment_8_results.md and embedder_benchmark/experiment_10_faiss_retrieval_scale/results/experiment_10_results.md).

Building the index means embedding the whole corpus, which takes tens of minutes on CPU.
The result is therefore cached under .cache/rag_index/ (gitignored) and reused. The cache
carries a manifest describing the corpus it was built from, so editing or replacing a
corpus file causes a rebuild instead of a silent mismatch between vectors and records.
"""

import json
import os

import faiss
import numpy as np

from . import dnabert_s
from .corpus import CORPUS_DIR, REPO_ROOT, corpus_files, load_records

CACHE_DIR = os.path.join(REPO_ROOT, ".cache", "rag_index")
INDEX_PATH = os.path.join(CACHE_DIR, "index_dna.faiss")
RECORDS_PATH = os.path.join(CACHE_DIR, "records.jsonl")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")

# HNSW parameters from experiment 8. efSearch trades recall against latency at query
# time and can be raised without rebuilding.
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64

_cached = None


def _fingerprint(corpus_dir=None):
    """Describe the corpus precisely enough to detect any change to it."""
    return {
        "model": dnabert_s.MODEL_NAME,
        "dim": dnabert_s.EMBEDDING_DIM,
        "max_tokens": dnabert_s.MAX_TOKENS,
        "hnsw_m": HNSW_M,
        "files": {
            os.path.basename(p): os.path.getsize(p)
            for p in corpus_files(corpus_dir)
        },
    }


def _normalise(vectors):
    out = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(out)
    return out


def build_index(corpus_dir=None, batch_size=16):
    """Embed the corpus and build the FAISS index. Slow; callers should cache."""
    directory = corpus_dir or CORPUS_DIR
    print(f"[INFO] Building DNA retrieval index from {directory}")

    records = load_records(directory)
    print(f"[INFO] {len(records)} records loaded, embedding with {dnabert_s.MODEL_NAME}")

    vectors = dnabert_s.embed(
        [r["sequence"] for r in records],
        batch_size=batch_size,
        progress_every=1000,
    )

    # Default metric is squared L2. On unit vectors that ranks identically to cosine,
    # and it is what the retrieval benchmark measured, so it is kept.
    index = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    index.add(_normalise(vectors))
    print(f"[INFO] Index built: {index.ntotal} vectors, {vectors.shape[1]} dimensions")

    return index, records


def save_index(index, records, corpus_dir=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(RECORDS_PATH, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    manifest = _fingerprint(corpus_dir)
    manifest["records"] = len(records)
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[INFO] Index cached in {CACHE_DIR}")


def _load_cached(corpus_dir=None):
    """Return (index, records) from the cache, or None if it is absent or stale."""
    if not all(os.path.exists(p) for p in (INDEX_PATH, RECORDS_PATH, MANIFEST_PATH)):
        return None

    with open(MANIFEST_PATH) as handle:
        manifest = json.load(handle)

    expected = _fingerprint(corpus_dir)
    if {k: manifest.get(k) for k in expected} != expected:
        print("[INFO] Cached index does not match the current corpus, rebuilding")
        return None

    index = faiss.read_index(INDEX_PATH)
    with open(RECORDS_PATH) as handle:
        records = [json.loads(line) for line in handle]

    if index.ntotal != len(records):
        print("[INFO] Cached index and record list disagree in length, rebuilding")
        return None

    index.hnsw.efSearch = EF_SEARCH
    print(f"[INFO] Loaded cached index: {index.ntotal} vectors from {CACHE_DIR}")
    return index, records


def load_or_build_index(corpus_dir=None, rebuild=False):
    """
    Main entry point. Returns (faiss index, records) and keeps them for the process.

    The first call on a fresh checkout embeds the whole corpus and takes tens of minutes;
    every later call reads the cache in seconds. Warm it deliberately with
    scripts/build_rag_index.py rather than letting it happen inside an agent run.
    """
    global _cached
    if _cached is not None and not rebuild:
        return _cached

    if not rebuild:
        cached = _load_cached(corpus_dir)
        if cached is not None:
            _cached = cached
            return _cached

    index, records = build_index(corpus_dir)
    save_index(index, records, corpus_dir)
    _cached = (index, records)
    return _cached


def search(query_sequence, k=3, corpus_dir=None):
    """
    Return the k most similar records to a DNA sequence, best first.

    Result is a list of (cosine similarity, record) tuples.
    """
    results = search_many([query_sequence], k=k, corpus_dir=corpus_dir)
    return results[0] if results else []


def _cosine_from_l2(squared_distance):
    """
    IndexHNSWFlat searches with squared L2, which is what FAISS returns.

    For unit vectors ||a - b||^2 = 2 - 2*cos, so the cosine is 1 - d/2. Converting here
    rather than leaving raw distances in the API matters: distances are better when
    smaller and cosines when larger, and anything merging results from several queries
    has to know which way round it is.
    """
    return 1.0 - float(squared_distance) / 2.0


def search_many(query_sequences, k=3, corpus_dir=None):
    """Search several sequences at once. Returns one result list per query."""
    queries = [q for q in query_sequences if q]
    if not queries:
        return []

    index, records = load_or_build_index(corpus_dir)
    vectors = _normalise(dnabert_s.embed(queries))
    distances, ids = index.search(vectors, min(k, index.ntotal))

    out = []
    for row_distances, row_ids in zip(distances, ids):
        hits = [(_cosine_from_l2(d), records[int(i)])
                for d, i in zip(row_distances, row_ids) if i >= 0]
        out.append(hits)
    return out
