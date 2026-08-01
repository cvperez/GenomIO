#!/usr/bin/env python3
"""
extract_cds_from_gbff.py — Extract CDS features from a GenBank flat file (.gbff or .gbff.gz)
and write them as a FASTA file that mirrors the NCBI cds_from_genomic.fna header format.

Usage:
    python3 scripts/extract_cds_from_gbff.py <input.gbff[.gz]> <output.fna>

This script is called by download_genomes_uniform.sh for assemblies that lack a
cds_from_genomic.fna.gz file on the NCBI FTP server but do have a _genomic.gbff.gz.
The GenBank flat file contains NCBI PGAP-generated CDS annotations with coordinates
and translated sequences, making it equivalent in authority to a cds_from_genomic file.
"""

import gzip
import sys
import os
from Bio import SeqIO


def extract_cds(gbff_path: str, out_path: str) -> int:
    """
    Parse a GenBank file and write all CDS features to a FASTA file.

    Returns the number of CDS records written.
    """
    opener = gzip.open if gbff_path.endswith(".gz") else open
    count = 0

    with opener(gbff_path, "rt") as handle, open(out_path, "w") as out:
        for record in SeqIO.parse(handle, "genbank"):
            cds_index = 0
            for feature in record.features:
                if feature.type != "CDS":
                    continue

                # Extract nucleotide sequence using the feature coordinates
                seq = feature.extract(record.seq)

                # Build qualifiers matching NCBI cds_from_genomic.fna header convention
                locus_tag = feature.qualifiers.get("locus_tag", [f"cds_{cds_index}"])[0]
                product = feature.qualifiers.get("product", ["unknown"])[0]
                protein_id = feature.qualifiers.get("protein_id", [""])[0]
                gene = feature.qualifiers.get("gene", [""])[0]
                location = str(feature.location)

                # Compose header; include gene field only when present
                gene_field = f" [gene={gene}]" if gene else ""
                protein_field = f" [protein_id={protein_id}]" if protein_id else ""
                header = (
                    f">{record.id}_cds_{locus_tag}_{cds_index}"
                    f"{gene_field}"
                    f" [locus_tag={locus_tag}]"
                    f" [product={product}]"
                    f"{protein_field}"
                    f" [location={location}]"
                    f" [gbkey=CDS]"
                )

                out.write(f"{header}\n{seq}\n")
                count += 1
                cds_index += 1

    return count


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.gbff[.gz]> <output.fna>", file=sys.stderr)
        sys.exit(1)

    gbff_path, out_path = sys.argv[1], sys.argv[2]

    if not os.path.exists(gbff_path):
        print(f"[ERROR] Input file not found: {gbff_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Parsing GenBank file: {gbff_path}")
    n = extract_cds(gbff_path, out_path)
    print(f"[INFO] Extracted {n} CDS records → {out_path}")

    if n == 0:
        print("[WARNING] No CDS features found. Check that the GBFF contains gene annotations.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
