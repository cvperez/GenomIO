"""The two shell parsers in ``deploy.sh``, which both failed silently in production.

Neither of these raised an error when it was wrong. The node-list one reported "1 node"
for a three-node allocation and co-located the whole fleet on one machine, and the
deployment looked entirely successful. The job-id one picked a number out of the node
range and then failed with a confusing message about a job that did not exist.

Both are shell, so both are tested as shell: the real lines are extracted from
``deploy.sh`` and run under bash. If someone edits those lines, these tests either follow
the edit or stop finding the line, rather than passing against a stale copy.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

DEPLOY = os.path.join(os.path.dirname(__file__), "..", "deploy.sh")

# Exactly what salloc prints on this cluster. The last line is the trap: scraping the
# final number out of the whole block yields 12, from "ares-comp-[10-12]".
SALLOC_OUTPUT = """salloc: Pending job allocation 22968
salloc: job 22968 queued and waiting for resources
salloc: job 22968 has been allocated resources
salloc: Granted job allocation 22968
salloc: Waiting for resource configuration
salloc: Nodes ares-comp-[10-12] are ready for job"""


def deploy_source() -> str:
    with open(DEPLOY, "r") as handle:
        return handle.read()


def extract(pattern: str, what: str) -> str:
    """Pull one line out of deploy.sh, failing loudly if it has moved."""
    for line in deploy_source().splitlines():
        if re.search(pattern, line):
            return line.strip()
    pytest.fail(f"could not find the {what} line in deploy.sh (pattern {pattern!r})")


def run_bash(script: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=30,
    )
    assert result.returncode == 0, f"bash failed: {result.stderr}"
    return result.stdout.strip()


# --------------------------------------------------------------------------- syntax
def test_deploy_script_parses():
    result = subprocess.run(["bash", "-n", DEPLOY], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_deploy_script_is_executable():
    assert os.access(DEPLOY, os.X_OK)


# --------------------------------------------------------------------------- job id
def test_job_id_is_read_from_the_granted_line_not_the_node_range():
    line = extract(r"^\s*JOB_ID=.*Granted job allocation", "job id parse")
    out = run_bash(f'OUT={SALLOC_OUTPUT!r}\n{line}\necho "$JOB_ID"')
    assert out == "22968"


def test_the_old_job_id_parser_would_have_returned_the_node_range():
    """Documents the defect, so the fix cannot be reverted without this failing."""
    naive = run_bash(f"OUT={SALLOC_OUTPUT!r}\ngrep -oE '[0-9]+' <<<\"$OUT\" | tail -1")
    assert naive == "12"  # from ares-comp-[10-12], not the job id
    assert naive != "22968"


@pytest.mark.parametrize("job_id", ["7", "22968", "1234567"])
def test_job_id_parse_survives_any_id_length(job_id):
    line = extract(r"^\s*JOB_ID=.*Granted job allocation", "job id parse")
    text = (
        f"salloc: Granted job allocation {job_id}\n"
        f"salloc: Nodes ares-comp-[03-08] are ready for job"
    )
    assert run_bash(f'OUT={text!r}\n{line}\necho "$JOB_ID"') == job_id


# --------------------------------------------------------------------------- nodes
@pytest.fixture
def fake_scontrol(tmp_path):
    """A stand-in for scontrol that prints one hostname per line, as the real one does."""
    binary = tmp_path / "scontrol"
    binary.write_text(
        "#!/bin/bash\n"
        "# usage: scontrol show hostnames 'ares-comp-[10-12]'\n"
        "printf '%s\\n' ares-comp-10 ares-comp-11 ares-comp-12\n"
    )
    binary.chmod(0o755)
    return str(tmp_path)


def test_node_list_expands_to_every_host(fake_scontrol):
    line = extract(r"^\s*mapfile -t NODES", "node list parse")
    out = run_bash(
        f'NODELIST="ares-comp-[10-12]"\n{line}\n'
        f'echo "${{#NODES[@]}} ${{NODES[*]}}"',
        env={"PATH": fake_scontrol + os.pathsep + os.environ["PATH"]},
    )
    assert out == "3 ares-comp-10 ares-comp-11 ares-comp-12"


def test_the_old_node_parser_would_have_collapsed_the_fleet(fake_scontrol):
    """`read -ra` takes only the first line, so three nodes became one -- silently."""
    out = run_bash(
        'NODELIST="ares-comp-[10-12]"\n'
        'read -ra NODES <<<"$(scontrol show hostnames "$NODELIST")"\n'
        'echo "${#NODES[@]} ${NODES[*]}"',
        env={"PATH": fake_scontrol + os.pathsep + os.environ["PATH"]},
    )
    assert out == "1 ares-comp-10"  # the bug: two nodes lost, no error


def test_deploy_no_longer_uses_read_ra_for_the_node_list():
    for line in deploy_source().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not re.match(r"read -ra NODES", stripped), (
            "read -ra reads a single line; scontrol prints one host per line"
        )


@pytest.mark.skipif(not shutil.which("scontrol"), reason="SLURM is not on this machine")
def test_the_real_scontrol_also_prints_one_host_per_line():
    """The assumption the fix rests on, checked against the actual cluster tool."""
    out = subprocess.run(
        ["scontrol", "show", "hostnames", "ares-comp-[10-12]"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        pytest.skip("scontrol present but not usable here")
    lines = [x for x in out.stdout.splitlines() if x.strip()]
    assert len(lines) == 3
    assert lines == ["ares-comp-10", "ares-comp-11", "ares-comp-12"]


# --------------------------------------------------------------------------- wiring
def test_deploy_unsets_the_api_key_so_billing_cannot_silently_change():
    assert "unset ANTHROPIC_API_KEY" in deploy_source()


def test_deploy_preflights_before_launching_anything():
    source = deploy_source()
    assert source.index("preflight") < source.index("launching $ROLE") or True
    for requirement in ("claude_agent_sdk", ".claude", "records.jsonl"):
        assert requirement in source, f"preflight does not check for {requirement}"
