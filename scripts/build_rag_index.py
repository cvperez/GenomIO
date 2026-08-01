#!/usr/bin/env python3
"""
Build the DNABERT-S retrieval index over rag_corpus_uniform/.

The index is not committed: about 43,500 records at 768 dimensions is roughly 130 MB,
past GitHub's file size limit. It is built once into .cache/rag_index/ and reused from
there, so this only needs running on a fresh checkout or after the corpus changes.

Expect tens of minutes on CPU. Nothing else in the pipeline needs to be running.

    python3 scripts/build_rag_index.py              # build if missing or stale
    python3 scripts/build_rag_index.py --rebuild    # force a rebuild
    python3 scripts/build_rag_index.py --check      # report cache state and exit
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rag import index as rag_index  # noqa: E402
from rag.corpus import CORPUS_DIR, corpus_files  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild even if a valid cache exists")
    parser.add_argument("--check", action="store_true",
                        help="report whether the cache is present and current, then exit")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="sequences per forward pass (default 16)")
    args = parser.parse_args()

    print(f"[INFO] Corpus:  {CORPUS_DIR} ({len(corpus_files())} files)")
    print(f"[INFO] Cache:   {rag_index.CACHE_DIR}")

    if args.check:
        cached = rag_index._load_cached()
        if cached is None:
            print("[INFO] No usable cache. Run this script without --check to build one.")
            return 1
        print(f"[INFO] Cache is current: {cached[0].ntotal} vectors")
        return 0

    started = time.perf_counter()
    if args.rebuild:
        idx, records = rag_index.build_index(batch_size=args.batch_size)
        rag_index.save_index(idx, records)
    else:
        idx, records = rag_index.load_or_build_index()
    elapsed = time.perf_counter() - started

    print(f"[INFO] Ready: {idx.ntotal} vectors over {len(records)} records "
          f"in {elapsed / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
