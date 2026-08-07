#!/bin/bash
# deploy.sh -- run the three GenomIO agents on three ARES compute nodes.
#
#   salloc -N 3 -p compute -t 02:00:00 --no-shell -J genomio-mas   # note the JOBID
#   ./multiagent_system/deploy.sh <JOBID>
#   multiagent_system/.venv/bin/python3 multiagent_system/test.py --mode slurm
#   scancel <JOBID>
#
# Or let it allocate for you:
#   ./multiagent_system/deploy.sh --alloc
#
# Environment overrides:
#   BASE_PORT (8100)  PARTITION (compute)  WALLTIME (02:00:00)  THREADS (16)
#   MODEL (whatever config.py defaults to)
#
# Notes specific to ARES:
#   * ssh to a compute node is refused by pam_slurm_adopt unless you hold an allocation
#     on it. salloc first and ssh works; without it nothing does. srun --overlap is the
#     fallback and the script reports which path it used.
#   * $HOME and /mnt/common are shared, so the venv, the FAISS cache and ~/.claude
#     credentials are visible from every node with no staging.
#   * The compute nodes' root filesystem is effectively full; TMPDIR goes to NVMe.
#   * ANTHROPIC_API_KEY is unset so the SDK uses claude CLI subscription auth.
set -uo pipefail

MAS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$MAS_DIR/.." && pwd)"
VENV_PY="$MAS_DIR/.venv/bin/python3"
RUN_DIR="$MAS_DIR/.run"
LOG_DIR="$MAS_DIR/logs"
BASE_PORT="${BASE_PORT:-8100}"
PARTITION="${PARTITION:-compute}"
WALLTIME="${WALLTIME:-02:00:00}"
THREADS="${THREADS:-16}"
ROLES=(coordinator retrieval reconstruction)

die() { echo "FATAL: $*" >&2; exit 1; }
info() { echo "[deploy] $*"; }

# --------------------------------------------------------------- allocation
JOB_ID="${1:-}"
if [[ "$JOB_ID" == "--alloc" ]]; then
  info "requesting 3 nodes on $PARTITION for $WALLTIME"
  OUT="$(salloc -N 3 -p "$PARTITION" -t "$WALLTIME" --no-shell -J genomio-mas 2>&1)" \
    || die "salloc failed: $OUT"
  echo "$OUT"
  # Parse the job id from the "Granted job allocation N" line specifically. Scraping the
  # last number out of the whole output picks up the node range instead --
  # "Nodes ares-comp-[10-12] are ready for job" yields 12, not the job id.
  JOB_ID="$(sed -n 's/.*Granted job allocation \([0-9]\{1,\}\).*/\1/p' <<<"$OUT" | tail -1)"
  [[ -n "$JOB_ID" ]] || die "could not parse a job id out of: $OUT"
  info "allocation $JOB_ID -- remember to 'scancel $JOB_ID' when finished"
fi
[[ -n "$JOB_ID" ]] || die "usage: $0 <SLURM_JOBID> | --alloc"

NODELIST="$(squeue -j "$JOB_ID" -h -o '%N' 2>/dev/null)"
[[ -n "$NODELIST" ]] || die "job $JOB_ID has no nodes (is it still queued? try 'squeue -j $JOB_ID')"
# scontrol prints one hostname per line, so this must be mapfile: `read -ra` would take
# only the first and silently co-locate a three-node fleet on one machine.
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
info "job $JOB_ID holds ${#NODES[@]} node(s): ${NODES[*]}"

# One agent per node when there are three, otherwise co-locate on the first. The
# reconstruction and retrieval agents each want about 1 GB and a CPU to themselves.
if [[ ${#NODES[@]} -ge 3 ]]; then
  PLACEMENT="one-per-node"
  HOSTS=("${NODES[0]}" "${NODES[1]}" "${NODES[2]}")
  PORTS=("$BASE_PORT" "$BASE_PORT" "$BASE_PORT")
else
  PLACEMENT="co-located on ${NODES[0]}"
  HOSTS=("${NODES[0]}" "${NODES[0]}" "${NODES[0]}")
  PORTS=("$BASE_PORT" "$((BASE_PORT + 1))" "$((BASE_PORT + 2))")
  info "WARNING: fewer than 3 nodes; $PLACEMENT"
fi
info "placement: $PLACEMENT"

# --------------------------------------------------------------- preflight
# Check every node before spending a single model token.
LAUNCHER=""
for i in "${!HOSTS[@]}"; do
  NODE="${HOSTS[$i]}"
  if [[ -z "$LAUNCHER" ]]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" true 2>/dev/null; then
      LAUNCHER="ssh"
    else
      LAUNCHER="srun"
      info "ssh refused (pam_slurm_adopt); falling back to srun --overlap"
    fi
  fi
done
info "launcher: $LAUNCHER"

remote() {  # remote <node> <command...>
  local node="$1"; shift
  if [[ "$LAUNCHER" == "ssh" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" "$@"
  else
    srun --jobid="$JOB_ID" --overlap -N1 -n1 -w "$node" --quiet bash -lc "$*"
  fi
}

for NODE in $(printf '%s\n' "${HOSTS[@]}" | sort -u); do
  OUT="$(remote "$NODE" "
    hostname
    [[ -x '$VENV_PY' ]] || { echo 'MISSING venv at $VENV_PY'; exit 1; }
    [[ -d \"\$HOME/.claude\" ]] || { echo 'MISSING ~/.claude credentials'; exit 1; }
    [[ -x \"\$HOME/.local/bin/claude\" ]] || { echo 'MISSING claude CLI'; exit 1; }
    [[ -f '$REPO_ROOT/.cache/rag_index/records.jsonl' ]] || { echo 'MISSING FAISS cache'; exit 1; }
    '$VENV_PY' -c 'import claude_agent_sdk, torch, faiss' || { echo 'IMPORT FAILED'; exit 1; }
    echo OK
  " 2>&1)" || { echo "$OUT"; die "preflight failed on $NODE"; }
  info "preflight $NODE: $(tail -1 <<<"$OUT")"
done

# --------------------------------------------------------------- endpoints
mkdir -p "$RUN_DIR" "$LOG_DIR/$JOB_ID"
declare -A URL
for i in "${!ROLES[@]}"; do
  URL[${ROLES[$i]}]="http://${HOSTS[$i]}:${PORTS[$i]}"
done
{
  echo "{"
  echo "  \"coordinator\": \"${URL[coordinator]}\","
  echo "  \"retrieval\": \"${URL[retrieval]}\","
  echo "  \"reconstruction\": \"${URL[reconstruction]}\""
  echo "}"
} > "$RUN_DIR/endpoints.json"
info "endpoints -> $RUN_DIR/endpoints.json"
for r in "${ROLES[@]}"; do printf '  %-15s %s\n' "$r" "${URL[$r]}"; done

# --------------------------------------------------------------- launch
PIDFILE="$RUN_DIR/pids-$JOB_ID.txt"
: > "$PIDFILE"
for i in "${!ROLES[@]}"; do
  ROLE="${ROLES[$i]}"; NODE="${HOSTS[$i]}"; PORT="${PORTS[$i]}"
  LOG="$LOG_DIR/$JOB_ID/agent-$ROLE-$NODE.log"
  info "launching $ROLE on $NODE:$PORT"
  remote "$NODE" "
    cd '$REPO_ROOT'
    unset ANTHROPIC_API_KEY
    mkdir -p /mnt/nvme/\$USER/genomio-tmp 2>/dev/null || true
    export TMPDIR=/mnt/nvme/\$USER/genomio-tmp
    [[ -d \"\$TMPDIR\" ]] || export TMPDIR=/tmp
    export OMP_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1
    export GENOMIO_URL_COORDINATOR='${URL[coordinator]}'
    export GENOMIO_URL_RETRIEVAL='${URL[retrieval]}'
    export GENOMIO_URL_RECONSTRUCTION='${URL[reconstruction]}'
    ${MODEL:+export GENOMIO_AGENT_MODEL='$MODEL'}
    nohup '$VENV_PY' -m multiagent_system.agents.$ROLE \
      --host 0.0.0.0 --port $PORT --threads $THREADS \
      --retrieval-url '${URL[retrieval]}' \
      --reconstruction-url '${URL[reconstruction]}' \
      > '$LOG' 2>&1 &
    echo \$!
  " >> "$PIDFILE" 2>/dev/null &
done
wait
info "launched; logs in $LOG_DIR/$JOB_ID/"

# --------------------------------------------------------------- health gate
info "waiting for models (retrieval loads DNABERT-S, ~30s cold)"
DEADLINE=$((SECONDS + 480))
while (( SECONDS < DEADLINE )); do
  READY=0
  for r in "${ROLES[@]}"; do
    curl -s --max-time 4 "${URL[$r]}/health" 2>/dev/null | grep -q '"models_loaded": *true' \
      && READY=$((READY + 1))
  done
  if (( READY == 3 )); then
    info "all three agents ready after ${SECONDS}s"
    for r in "${ROLES[@]}"; do
      printf '  %-15s %s\n' "$r" "$(curl -s --max-time 4 "${URL[$r]}/health")"
    done
    echo
    info "now run:  $VENV_PY $MAS_DIR/test.py --mode slurm"
    info "tear down: scancel $JOB_ID"
    exit 0
  fi
  sleep 5
done

echo "FATAL: agents did not become ready in 480s" >&2
for i in "${!ROLES[@]}"; do
  echo "--- ${ROLES[$i]} on ${HOSTS[$i]} ---"
  tail -20 "$LOG_DIR/$JOB_ID/agent-${ROLES[$i]}-${HOSTS[$i]}.log" 2>/dev/null || echo "(no log)"
done
exit 1
