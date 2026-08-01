#!/bin/bash
# =============================================================================
# download_genomes_uniform.sh
# =============================================================================
# Extended pipeline that produces rag_corpus_uniform/ with uniform, gene-level
# FASTA granularity across all 20 species.
#
# Problem this script solves:
#   The original download_genomes.sh selects .fna.gz files alphabetically, which
#   picks cds_from_genomic.fna.gz (one record per coding sequence, ~300-3000 bp)
#   for most assemblies. However, for 5 metagenome-assembled genomes (MAGs),
#   NCBI provides no cds_from_genomic.fna.gz — only _genomic.fna.gz (one record
#   per assembled contig, 10k-200k+ bp). This mismatch in record granularity
#   breaks the assumption that every document in the RAG corpus is gene-sized.
#
# NCBI assembly query — three-pass candidate collection:
#   The script runs three NCBI queries per species, merges the results, deduplicates
#   by assembly accession, and iterates through all candidates until a usable CDS
#   source is found. Running all three passes upfront (rather than stopping after
#   the first non-empty pass) is intentional: the "latest" assembly for a species
#   may be an unannotated deposit whose GBFF contains 0 CDS features, in which
#   case candidates from the broader passes must be available as alternatives.
#
#   Pass 1: "species"[Organism] AND latest[filter] AND complete genome[filter]
#           Preferred: finished, fully-annotated assemblies.
#   Pass 2: "species"[Organism] AND latest[filter]
#           Includes MAGs and scaffold assemblies, for taxa that lack complete
#           genomes (e.g. CPR bacteria such as Candidatus Woesebacteria).
#   Pass 3: genus[Organism] AND latest[filter]
#           Genus-level fallback for taxa whose species phrase is unresolvable
#           in NCBI (e.g. "Candidatus Nealsonbacteria bacterium" — NCBI stores
#           only strain-specific names such as "...bacterium RBG_13_36_15",
#           making the bare species phrase unrecognised even without filters).
#
# Strategy applied per assembly candidate:
#   A) cds_from_genomic.fna.gz exists in FTP listing
#      → Download and copy as-is. No additional processing needed.
#   B) No CDS FASTA, but _genomic.gbff.gz exists in FTP listing
#      → Download the GenBank flat file (NCBI PGAP annotations embedded) and
#        extract CDS features using extract_cds_from_gbff.py (Biopython).
#        If the GBFF has 0 CDS features (unannotated deposit — observed for
#        Candidatus Woesebacteria bacterium GCA_013374835.1 which has only a
#        'source' feature in its GBFF), the script logs the finding and tries
#        the next candidate rather than giving up.
#   C) No CDS FASTA and no GBFF annotation
#      → Fall back to Prodigal de-novo gene prediction (-p meta mode for MAGs).
#        Not required for the current 20-species corpus (all resolvable species
#        resolve via Strategy A or B), but included for robustness.
#
# Design constraints:
#   - No sliding-window chunking: every record boundary is a biological boundary.
#   - Original rag_corpus/ and download_genomes.sh are never modified.
#   - Script is reproducible: re-running produces the same rag_corpus_uniform/.
#
# Prerequisites:
#   - NCBI E-utilities: esearch, efetch, xtract (same as download_genomes.sh)
#   - curl
#   - Python 3 with Biopython (pip install biopython)
#   - Prodigal (only needed for Strategy C; optional for current corpus)
#
# Usage:
#   bash scripts/download_genomes_uniform.sh
# =============================================================================

set -e

# Determine repo root relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Output directories
OUTPUT_CORPUS="$REPO_ROOT/rag_corpus_uniform"
DOWNLOAD_DIR="$REPO_ROOT/genomes_downloaded_uniform"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_cds_from_gbff.py"

# Accession to organism table, consumed by src/rag/corpus.py. Recorded here rather than
# derived later because this is the only point where the species name that produced an
# assembly is known for certain; the FASTA records themselves carry locus tags, not names.
ORGANISM_TABLE="$OUTPUT_CORPUS/organisms.tsv"

mkdir -p "$OUTPUT_CORPUS"
mkdir -p "$DOWNLOAD_DIR"

{
  echo "# Assembly accession to source organism, as resolved by scripts/download_genomes_uniform.sh."
  echo "# Written by the download script at the moment it resolves each species, so the name here is"
  echo "# the one NCBI answered with. Do not edit by hand; rerun the script instead."
  echo "#"
  printf '# accession\torganism\n'
} > "$ORGANISM_TABLE"

# Append one row for a resolved species. Accession is normalised to the GCF_000000000 form
# (no version suffix) so it matches the prefix of the FASTA filename.
record_organism() {
  local filename="$1" species="$2"
  local accession
  accession="$(echo "$filename" | cut -d_ -f1-2 | cut -d. -f1)"
  printf '%s\t%s\n' "$accession" "$species" >> "$ORGANISM_TABLE"
}

# Same species list as download_genomes.sh (order unchanged)
species_list=(
  "Candidatus Nealsonbacteria bacterium"
  "Candidatus Nomurabacteria bacterium"
  "Staphylococcus aureus"
  "Candidatus Moraniibacteriota bacterium"
  "Salinibacter ruber"
  "Akkermansia muciniphila"
  "Dehalococcoides mccartyi"
  "Chlamydia trachomatis"
  "Campylobacter lari"
  "Candidatus Woesebacteria bacterium"
  "Fusobacterium animalis"
  "Thermus thermophilus"
  "Pseudogemmatithrix spongiicola"
  "Archangium violaceum"
  "Fusobacterium polymorphum"
  "Nitrospira sp."
  "Candidatus Parcubacteria bacterium"
  "Candidatus Saccharibacteria bacterium oral taxon 488"
  "Candidatus Peregrinibacteria bacterium"
  "Chlamydia psittaci"
)

# ---------------------------------------------------------------------------
# query_assemblies QUERY
#   Queries NCBI Assembly and returns up to 5 tab-separated lines:
#     AssemblyAccession  FtpPath_RefSeq  FtpPath_GenBank
#   NCBI errors are suppressed; returns empty on failure.
# ---------------------------------------------------------------------------
query_assemblies() {
  local query="$1"
  esearch -db assembly -query "$query" 2>/dev/null |
  efetch -format docsum 2>/dev/null |
  xtract -pattern DocumentSummary \
         -element AssemblyAccession FtpPath_RefSeq FtpPath_GenBank 2>/dev/null |
  head -n 5 || true
}

# ---------------------------------------------------------------------------
# extract_genus SPECIES_NAME
#   Returns the genus token:
#     "Candidatus Nealsonbacteria bacterium" → "Nealsonbacteria"
#     "Staphylococcus aureus"               → "Staphylococcus"
#     "Nitrospira sp."                       → "Nitrospira"
# ---------------------------------------------------------------------------
extract_genus() {
  local name="$1"
  if [[ "$name" == Candidatus* ]]; then
    echo "$name" | awk '{print $2}'
  else
    echo "$name" | awk '{print $1}'
  fi
}

echo "[INFO] Building rag_corpus_uniform/ with uniform CDS-level granularity..."

for species in "${species_list[@]}"; do
  echo "=============================="
  echo "[INFO] Processing: $species"

  # -------------------------------------------------------------------------
  # COLLECT ASSEMBLY CANDIDATES (three-pass, deduplicated)
  # -------------------------------------------------------------------------
  # All three passes run upfront so that if the "latest" complete assembly
  # turns out to be unannotated (Strategy B → 0 CDS), the broader pass results
  # are already available as alternatives in the same iteration loop.
  GENUS=$(extract_genus "$species")

  P1=$(query_assemblies "\"$species\"[Organism] AND latest[filter] AND complete genome[filter]")
  P2=$(query_assemblies "\"$species\"[Organism] AND latest[filter]")
  P3=$(query_assemblies "$GENUS[Organism] AND latest[filter]")

  # Merge all three passes, remove blank lines, deduplicate by first field
  # (assembly accession), cap at 15 candidates total.
  ALL_CANDIDATES=$(printf "%s\n%s\n%s\n" "$P1" "$P2" "$P3" \
    | grep -v '^[[:space:]]*$' \
    | awk '!seen[$1]++' \
    | head -n 15)

  if [[ -z "$ALL_CANDIDATES" ]]; then
    echo "  [WARNING] No NCBI assembly found for '$species' in any query pass. Skipping."
    continue
  fi

  # -------------------------------------------------------------------------
  # ASSEMBLY ITERATION
  # Try each candidate in order until a usable CDS source is found.
  # -------------------------------------------------------------------------
  RESOLVED=false

  while IFS=$'\t' read -r ASSEMBLY FTP_REF FTP_GEN; do
    [[ "$RESOLVED" == "true" ]] && break
    [[ -z "$FTP_REF" && -z "$FTP_GEN" ]] && continue

    # Prefer RefSeq over GenBank (same policy as original script)
    if [[ -n "$FTP_REF" ]]; then
      FTP_PATH="$FTP_REF"
    else
      FTP_PATH="$FTP_GEN"
    fi

    echo "  [INFO] Trying assembly: $ASSEMBLY"
    FILE_LIST=$(curl -s "$FTP_PATH/" 2>/dev/null || true)
    [[ -z "$FILE_LIST" ]] && { echo "  [WARNING] Empty FTP listing for $ASSEMBLY. Skipping."; continue; }

    # ------------------------------------------------------------------
    # FILE TYPE DETECTION
    # ------------------------------------------------------------------
    # Check explicitly for each file type. We do NOT rely on alphabetic
    # sort order (the original script's implicit mechanism). 'cds_from_genomic
    # .fna.gz' sorts before '_genomic.fna.gz', which is why the original
    # script worked for 16/21 species, but explicit detection is clearer.
    # ------------------------------------------------------------------
    CDS_FILE=$(echo "$FILE_LIST"     | grep 'cds_from_genomic\.fna\.gz' | awk '{print $NF}' | head -n 1)
    GBFF_FILE=$(echo "$FILE_LIST"    | grep '_genomic\.gbff\.gz'         | awk '{print $NF}' | head -n 1)
    GENOMIC_FILE=$(echo "$FILE_LIST" | grep '_genomic\.fna\.gz'          | awk '{print $NF}' | head -n 1)

    TARGET_DIR="$DOWNLOAD_DIR/${species// /_}_data"
    mkdir -p "$TARGET_DIR"

    # ------------------------------------------------------------------
    # STRATEGY A: Official CDS FASTA exists
    # ------------------------------------------------------------------
    if [[ -n "$CDS_FILE" ]]; then
      # File type detected: cds_from_genomic. One record per coding sequence.
      # No additional processing needed — copy directly to rag_corpus_uniform/.
      echo "  [STRATEGY A] cds_from_genomic.fna.gz found for $ASSEMBLY."
      echo "  [DETECTED] Assembly: $ASSEMBLY | File: cds_from_genomic | Strategy: direct copy"

      CDS_URL="${FTP_PATH}/${CDS_FILE}"
      LOCAL_CDS="${TARGET_DIR}/${CDS_FILE}"

      if curl -s -o "$LOCAL_CDS" "$CDS_URL" 2>/dev/null; then
        cp "$LOCAL_CDS" "$OUTPUT_CORPUS/"
        echo "  [OK] Copied: $CDS_FILE"
        record_organism "$CDS_FILE" "$species"
        RESOLVED=true
      else
        echo "  [ERROR] Download failed: $CDS_URL. Trying next assembly."
      fi

    # ------------------------------------------------------------------
    # STRATEGY B: No CDS FASTA, but GenBank annotation file exists
    # ------------------------------------------------------------------
    elif [[ -n "$GBFF_FILE" ]]; then
      # File type detected: _genomic (no cds_from_genomic on FTP).
      # GBFF annotation found: should contain NCBI PGAP CDS annotations.
      # If the GBFF has 0 CDS features (unannotated deposit — observed for
      # GCA_013374835.1 Woesebacteria, GBFF has only 'source' feature),
      # the script logs the finding and moves to the next candidate.
      echo "  [STRATEGY B] No cds_from_genomic for $ASSEMBLY. Trying GBFF CDS extraction."
      echo "  [DETECTED] Assembly: $ASSEMBLY | File: _genomic (GBFF) | Strategy: GBFF CDS extraction"

      GBFF_URL="${FTP_PATH}/${GBFF_FILE}"
      LOCAL_GBFF="${TARGET_DIR}/${GBFF_FILE}"

      if ! curl -s -o "$LOCAL_GBFF" "$GBFF_URL" 2>/dev/null; then
        echo "  [ERROR] GBFF download failed: $GBFF_URL. Trying next assembly."
        continue
      fi

      BASE="${GBFF_FILE%_genomic.gbff.gz}"
      OUT_FASTA="${TARGET_DIR}/${BASE}_cds_from_gbff.fna"

      if python3 "$EXTRACT_SCRIPT" "$LOCAL_GBFF" "$OUT_FASTA" 2>/dev/null; then
        cp "$OUT_FASTA" "$OUTPUT_CORPUS/"
        echo "  [OK] Extracted CDS → $OUTPUT_CORPUS/$(basename "$OUT_FASTA")"
        record_organism "$(basename "$OUT_FASTA")" "$species"
        RESOLVED=true
      else
        # Exit code 2: 0 CDS records extracted — assembly is an unannotated
        # deposit. Move to the next candidate assembly.
        echo "  [INFO] GBFF for $ASSEMBLY is unannotated (0 CDS features). Trying next candidate."
      fi

    # ------------------------------------------------------------------
    # STRATEGY C: No official annotation — Prodigal gene prediction
    # ------------------------------------------------------------------
    elif [[ -n "$GENOMIC_FILE" ]]; then
      # File type detected: _genomic. No cds_from_genomic.fna.gz and no
      # _genomic.gbff.gz found in this assembly's FTP listing.
      # NOTE: Strategy C has not been triggered for any assembly in the
      # current 20-species corpus. Retained for future robustness.
      echo "  [STRATEGY C] No CDS FASTA or GBFF for $ASSEMBLY. Attempting Prodigal."
      echo "  [DETECTED] Assembly: $ASSEMBLY | File: _genomic (no annotation) | Strategy: Prodigal"

      if ! command -v prodigal &>/dev/null; then
        echo "  [INFO] Prodigal not installed — skipping this assembly."
        echo "         Install with: conda install -c bioconda prodigal"
        continue
      fi

      GENOMIC_URL="${FTP_PATH}/${GENOMIC_FILE}"
      LOCAL_GENOMIC="${TARGET_DIR}/${GENOMIC_FILE}"

      if ! curl -s -o "$LOCAL_GENOMIC" "$GENOMIC_URL" 2>/dev/null; then
        echo "  [ERROR] Genomic FASTA download failed. Trying next assembly."
        continue
      fi

      GENOMIC_UNZIPPED="${LOCAL_GENOMIC%.gz}"
      gunzip -c "$LOCAL_GENOMIC" > "$GENOMIC_UNZIPPED"

      BASE="${GENOMIC_FILE%_genomic.fna.gz}"
      OUT_FASTA="${TARGET_DIR}/${BASE}_cds_prodigal.fna"

      # -p meta: metagenome mode (appropriate for MAGs without close relatives)
      # -d: output nucleotide sequences of predicted genes
      if prodigal \
           -i "$GENOMIC_UNZIPPED" \
           -o /dev/null \
           -f gff \
           -d "$OUT_FASTA" \
           -p meta \
           2>/dev/null; then
        cp "$OUT_FASTA" "$OUTPUT_CORPUS/"
        echo "  [OK] Prodigal predicted genes → $OUTPUT_CORPUS/$(basename "$OUT_FASTA")"
        record_organism "$(basename "$OUT_FASTA")" "$species"
        RESOLVED=true
      else
        echo "  [ERROR] Prodigal failed for $ASSEMBLY. Trying next assembly."
      fi

    else
      echo "  [WARNING] No usable .fna.gz or .gbff.gz found for $ASSEMBLY. Trying next."
    fi

  done <<< "$ALL_CANDIDATES"

  if [[ "$RESOLVED" == "false" ]]; then
    echo "  [WARNING] Could not find a usable CDS source for '$species' after all candidates. Skipping."
  fi

done

# Decompress any .fna.gz files copied by Strategy A
echo ""
echo "[INFO] Decompressing .fna.gz files in $OUTPUT_CORPUS/ ..."
gunzip -f "$OUTPUT_CORPUS"/*.fna.gz 2>/dev/null || true

echo ""
echo "[INFO] Done. Files in $OUTPUT_CORPUS/:"
ls -lh "$OUTPUT_CORPUS/"
echo ""
echo "[INFO] Total FASTA files: $(ls "$OUTPUT_CORPUS"/*.fna 2>/dev/null | wc -l)"
echo "[INFO] Organism table: $ORGANISM_TABLE ($(grep -cv '^#' "$ORGANISM_TABLE") species recorded)"
echo ""
echo "[INFO] To evaluate structural quality of the produced files, run:"
echo "       python3 $SCRIPT_DIR/evaluate_cds_uniformity.py $OUTPUT_CORPUS/*.fna"
echo ""
echo "[INFO] To build the DNABERT-S retrieval index over this corpus, run:"
echo "       python3 $SCRIPT_DIR/build_rag_index.py"
