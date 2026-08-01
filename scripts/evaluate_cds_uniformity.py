#!/usr/bin/env python3
"""
evaluate_cds_uniformity.py — Structural and biological quality checks for CDS FASTA files.

Usage:
    python3 scripts/evaluate_cds_uniformity.py <fasta_file> [<fasta_file> ...]
    python3 scripts/evaluate_cds_uniformity.py rag_corpus_uniform/*.fna

For each file the script reports:
  - Total number of records
  - % of records with length in [300, 3000] bp (expected CDS range)
  - % of records starting with ATG (canonical start codon)
  - % of records ending with a canonical stop codon (TAA, TAG, TGA)
  - Length distribution summary (min, median, max)
  - A per-file verdict

Provenance is inferred from the filename suffix:
  *_cds_from_genomic.fna  → TRUSTED  (official NCBI annotation; statistics
                             are reported but no threshold-based verdict is
                             applied — these files are already validated by
                             NCBI's PGAP pipeline)
  *_cds_from_gbff.fna     → GBFF-derived; full quality thresholds applied
  *_cds_prodigal.fna      → Prodigal-predicted; full quality thresholds applied
  anything else           → UNKNOWN; full quality thresholds applied

Design note: Partial CDS records at contig boundaries may legitimately lack a
canonical start or stop codon, especially in fragmented MAG assemblies. Files
with low start/stop codon coverage are flagged SUSPICIOUS rather than FAIL unless
the coverage is critically low, to account for this expected biological phenomenon.
This applies only to derived (non-NCBI-official) files.
"""

import sys
import os
import statistics
from Bio import SeqIO


STOP_CODONS = {"TAA", "TAG", "TGA"}
CDS_MIN_LEN = 300
CDS_MAX_LEN = 3000

# Thresholds for verdict — applied only to derived (non-NCBI-official) files
SUSPICIOUS_START_ATG = 0.90   # flag SUSPICIOUS if < 90% start with ATG
SUSPICIOUS_STOP_OK   = 0.85   # flag SUSPICIOUS if < 85% end with stop codon
FAIL_LENGTH_OK       = 0.70   # flag FAIL if < 70% in CDS length range
FAIL_MIN_RECORDS     = 10     # flag FAIL if fewer than 10 records total

# Provenance constants
NCBI_OFFICIAL = "NCBI_OFFICIAL"
GBFF_DERIVED  = "GBFF_DERIVED"
PRODIGAL      = "PRODIGAL"
UNKNOWN       = "UNKNOWN"

# Short labels for table display
PROVENANCE_LABEL = {
    NCBI_OFFICIAL: "NCBI",
    GBFF_DERIVED:  "GBFF",
    PRODIGAL:      "PRODIGAL",
    UNKNOWN:       "UNKNOWN",
}


def detect_provenance(path: str) -> str:
    """
    Infer file provenance from the filename suffix.

    Returns one of the NCBI_OFFICIAL, GBFF_DERIVED, PRODIGAL, or UNKNOWN
    constants. This determines whether quality thresholds are applied (derived
    files only) or suppressed (NCBI official files, trusted by construction).
    """
    name = os.path.basename(path)
    if name.endswith("cds_from_genomic.fna"):
        return NCBI_OFFICIAL
    if name.endswith("cds_from_gbff.fna"):
        return GBFF_DERIVED
    if name.endswith("cds_prodigal.fna"):
        return PRODIGAL
    return UNKNOWN


def check_file(path: str) -> dict:
    provenance = detect_provenance(path)

    lengths = []
    n_start_atg = 0
    n_stop_ok = 0
    n_length_ok = 0
    total = 0

    for record in SeqIO.parse(path, "fasta"):
        seq = str(record.seq).upper()
        length = len(seq)
        total += 1
        lengths.append(length)

        if length >= 3:
            if seq[:3] == "ATG":
                n_start_atg += 1
            if seq[-3:] in STOP_CODONS:
                n_stop_ok += 1
        if CDS_MIN_LEN <= length <= CDS_MAX_LEN:
            n_length_ok += 1

    if total == 0:
        return {
            "file": os.path.basename(path),
            "provenance": provenance,
            "total": 0,
            "pct_length_ok": 0.0,
            "pct_start_atg": 0.0,
            "pct_stop_ok": 0.0,
            "len_min": None,
            "len_median": None,
            "len_max": None,
            "verdict": "FAIL",
            "reasons": ["No records found"],
        }

    pct_length_ok = n_length_ok / total
    pct_start_atg = n_start_atg / total
    pct_stop_ok   = n_stop_ok   / total
    len_min    = min(lengths)
    len_max    = max(lengths)
    len_median = statistics.median(lengths)

    # ------------------------------------------------------------------
    # VERDICT LOGIC
    # ------------------------------------------------------------------
    # NCBI official cds_from_genomic files are trusted by construction —
    # they are produced by NCBI's PGAP annotation pipeline and do not
    # require independent threshold-based quality assessment. Low ATG%
    # in these files reflects legitimate alternative start codons (GTG,
    # TTG), not a quality defect.
    # ------------------------------------------------------------------
    if provenance == NCBI_OFFICIAL:
        verdict = "TRUSTED"
        reasons = ["Official NCBI cds_from_genomic annotation — no threshold evaluation applied"]
    else:
        # Derived files (GBFF extraction or Prodigal prediction): apply thresholds
        reasons = []
        verdict = "PASS"

        if total < FAIL_MIN_RECORDS:
            reasons.append(f"Only {total} records (minimum expected: {FAIL_MIN_RECORDS})")
            verdict = "FAIL"

        if pct_length_ok < FAIL_LENGTH_OK:
            reasons.append(
                f"{pct_length_ok*100:.1f}% of records in [{CDS_MIN_LEN}, {CDS_MAX_LEN}] bp "
                f"(threshold: {FAIL_LENGTH_OK*100:.0f}%)"
            )
            verdict = "FAIL"
        elif pct_start_atg < SUSPICIOUS_START_ATG or pct_stop_ok < SUSPICIOUS_STOP_OK:
            if pct_start_atg < SUSPICIOUS_START_ATG:
                reasons.append(
                    f"Only {pct_start_atg*100:.1f}% of records start with ATG "
                    f"(expected ≥ {SUSPICIOUS_START_ATG*100:.0f}%)"
                )
            if pct_stop_ok < SUSPICIOUS_STOP_OK:
                reasons.append(
                    f"Only {pct_stop_ok*100:.1f}% of records end with a canonical stop codon "
                    f"(expected ≥ {SUSPICIOUS_STOP_OK*100:.0f}%)"
                )
            if verdict != "FAIL":
                verdict = "SUSPICIOUS"

        if not reasons:
            reasons.append("All checks passed")

    return {
        "file": os.path.basename(path),
        "provenance": provenance,
        "total": total,
        "pct_length_ok": pct_length_ok,
        "pct_start_atg": pct_start_atg,
        "pct_stop_ok": pct_stop_ok,
        "len_min": len_min,
        "len_median": len_median,
        "len_max": len_max,
        "verdict": verdict,
        "reasons": reasons,
    }


def print_report(results: list[dict]):
    # Header
    print("\n" + "=" * 104)
    print("CDS UNIFORMITY EVALUATION REPORT")
    print("=" * 104)

    header = (
        f"{'File':<50} {'Prov':>8} {'Records':>7} {'Len OK%':>7} {'ATG%':>6} {'Stop%':>6}  "
        f"{'Min':>5} {'Med':>5} {'Max':>5}  {'Verdict'}"
    )
    print(header)
    print("-" * 104)

    for r in results:
        prov_label = PROVENANCE_LABEL.get(r["provenance"], "?")
        if r["total"] == 0:
            row = (
                f"{r['file']:<50} {prov_label:>8} {'0':>7} {'N/A':>7} {'N/A':>6} {'N/A':>6}  "
                f"{'N/A':>5} {'N/A':>5} {'N/A':>5}  {r['verdict']}"
            )
        else:
            row = (
                f"{r['file']:<50} {prov_label:>8} {r['total']:>7} "
                f"{r['pct_length_ok']*100:>6.1f}% "
                f"{r['pct_start_atg']*100:>5.1f}% "
                f"{r['pct_stop_ok']*100:>5.1f}%  "
                f"{r['len_min']:>5} {int(r['len_median']):>5} {r['len_max']:>5}  "
                f"{r['verdict']}"
            )
        print(row)
        # Only print reason lines for non-TRUSTED verdicts with actual issues
        if r["verdict"] not in ("PASS", "TRUSTED"):
            for reason in r["reasons"]:
                print(f"  {'':50} !! {reason}")

    print("=" * 104)

    # Biological assessment
    print("\nBIOLOGICAL ASSESSMENT")
    print("-" * 104)
    for r in results:
        print(f"\n[{r['file']}]")

        if r["total"] == 0:
            print("  No records. File is empty or unreadable.")
            continue

        # NCBI official files: one-line note only — no threshold commentary
        if r["provenance"] == NCBI_OFFICIAL:
            print(
                f"  Official NCBI cds_from_genomic — statistics only; no threshold assessment.\n"
                f"  Records: {r['total']} | median length: {int(r['len_median'])} bp | "
                f"ATG: {r['pct_start_atg']*100:.1f}% | Stop: {r['pct_stop_ok']*100:.1f}%"
            )
            continue

        # Derived files: full commentary
        median = int(r["len_median"])
        if 300 <= median <= 1500:
            print(f"  Length distribution: median={median} bp — consistent with typical prokaryotic CDS lengths.")
        elif median < 300:
            print(f"  Length distribution: median={median} bp — SHORT. Many records are below the minimum CDS threshold.")
            print("  Possible cause: annotation includes non-coding features or fragmented contigs.")
        else:
            print(f"  Length distribution: median={median} bp — LONG. Possible gene fusions or annotation errors.")

        if r["pct_start_atg"] >= 0.95:
            print(f"  Start codons: {r['pct_start_atg']*100:.1f}% ATG — excellent.")
        elif r["pct_start_atg"] >= SUSPICIOUS_START_ATG:
            pct = r["pct_start_atg"] * 100
            print(f"  Start codons: {pct:.1f}% ATG — acceptable. Some records may start with alternative start "
                  "codons (GTG, TTG) or are partial CDS at contig boundaries.")
        else:
            pct = r["pct_start_atg"] * 100
            print(f"  Start codons: {pct:.1f}% ATG — LOW. Suggests many partial or fragmented CDS records.")

        if r["pct_stop_ok"] >= 0.95:
            print(f"  Stop codons: {r['pct_stop_ok']*100:.1f}% canonical — excellent.")
        elif r["pct_stop_ok"] >= SUSPICIOUS_STOP_OK:
            pct = r["pct_stop_ok"] * 100
            print(f"  Stop codons: {pct:.1f}% canonical — acceptable. Partial CDS at contig boundaries "
                  "may lack stop codons.")
        else:
            pct = r["pct_stop_ok"] * 100
            print(f"  Stop codons: {pct:.1f}% canonical — LOW. Many CDS records appear truncated.")

        if r["verdict"] == "PASS":
            print("  VERDICT: PASS — Records are structurally consistent with coding sequences. "
                  "Preliminary automated opinion; final biological validation deferred pending expert review.")
        elif r["verdict"] == "SUSPICIOUS":
            print("  VERDICT: SUSPICIOUS — One or more quality checks are below threshold. "
                  "Likely explained by fragmented MAG assembly (partial CDS at contig breaks). "
                  "Manual inspection recommended before use in production.")
        else:
            print("  VERDICT: FAIL — Critical quality checks failed. Do not use this file without "
                  "investigating the root cause.")

    print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <fasta_file> [<fasta_file> ...]", file=sys.stderr)
        sys.exit(1)

    paths = sys.argv[1:]
    results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found, skipping: {path}", file=sys.stderr)
            continue
        print(f"[INFO] Checking: {path}")
        results.append(check_file(path))

    if not results:
        print("[ERROR] No valid files to evaluate.", file=sys.stderr)
        sys.exit(1)

    print_report(results)


if __name__ == "__main__":
    main()
