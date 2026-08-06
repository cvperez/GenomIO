# test.py
from src.agents.planner import plan
from pathlib import Path
import csv

GAPS_TSV = Path("simulated_draft_genomes/gaps/AP012051.1_gaps.tsv")
CONTIGS_FASTA = Path("simulated_draft_genomes/contigs/AP012051.1_contigs.fasta")

TARGET_GAP_ID = "AP012051.1_gap1"

def read_tsv_gap_length(tsv_path: Path, gap_id: str) -> dict:
    """Load gap info (including length) from a TSV with header:
       gap_id    start    end    length    sequence
    """
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["gap_id"] == gap_id:
                return {
                    "gap_id": row["gap_id"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "length": int(row["length"]),
                    "gap_sequence": row.get("sequence", "")
                }
    raise ValueError(f"{gap_id} not found in {tsv_path}")

def read_fasta_contigs(fasta_path: Path) -> list[dict]:
    """Parse a FASTA file into a list of dicts: {'header': str, 'sequence': str}."""
    contigs = []
    header = None
    seq_chunks = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous contig
                if header is not None:
                    contigs.append({"header": header, "sequence": "".join(seq_chunks).upper()})
                header = line[1:]  # drop '>'
                seq_chunks = []
            else:
                seq_chunks.append(line)
        # save last contig
        if header is not None:
            contigs.append({"header": header, "sequence": "".join(seq_chunks).upper()})
    if not contigs:
        raise RuntimeError(f"No contigs found in {fasta_path}")
    return contigs

def build_gapped_sequence_full(contig1_seq: str, contig2_seq: str, flank: int = 50) -> str:
    left = contig1_seq[-flank:] if len(contig1_seq) >= flank else contig1_seq
    right = contig2_seq[:flank] if len(contig2_seq) >= flank else contig2_seq
    return f"{left}---{right}"

if __name__ == "__main__":
    # 1) Load gap1 from TSV
    gap = read_tsv_gap_length(GAPS_TSV, TARGET_GAP_ID)
    gap_length = gap["length"]
    print(f"[DEBUG] Gap info: {gap}")

    # 2) Load contigs from FASTA and pick contig #1 and #2
    contigs = read_fasta_contigs(CONTIGS_FASTA)
    contig1 = contigs[0]
    contig2 = contigs[1]
    print("[INFO] Using full contigs:")
    print(f"  contig1: {contig1['header']} (len={len(contig1['sequence'])})")
    print(f"  contig2: {contig2['header']} (len={len(contig2['sequence'])})")

    # 3) Build the input sequence with a gap between full contigs
    input_dna = build_gapped_sequence_full(contig1["sequence"], contig2["sequence"])
    print("[INFO] Sequence passed to the agent:")
    print(f"  Total length (without dashes): {len(contig1['sequence']) + len(contig2['sequence'])}")
    print(f"  Start: {input_dna[:60]}...")
    print(f"  End: ...{input_dna[-60:]}")

    # 4) Call the planner with sequence + gap_length (plus optional meta)
    print(f"[DEBUG] Calling plan() with gap_length={gap_length}")
    output = plan({
        "sequence": input_dna,
        "gap_length": gap_length,
        "meta": {
            "gap_id": gap["gap_id"],
            "contig1_header": contig1["header"],
            "contig2_header": contig2["header"]
        }
    })

    print("[Final Output]:", output)
