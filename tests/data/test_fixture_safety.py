"""tests/data/fixtures/ is tracked: it must hold quality-control material only."""

import csv
import re
import subprocess
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_QC_RE = re.compile(r"^QC[A-Z]*-", re.IGNORECASE)
# The sample field follows the timestamp field in SCT file names.
_SAMPLE_IN_NAME = re.compile(r"\[\d{8}_\d{6}\]\[\s*([^\]]+)\]")


def test_fixtures_hold_only_qc_samples():
    samples = set()
    for f in _FIXTURES.rglob("*"):
        if m := _SAMPLE_IN_NAME.search(f.name):
            samples.add(m.group(1).strip())
        if f.name in ("XN_SAMPLE.csv", "OutputData.csv"):
            with open(f, encoding="latin-1") as fh:
                samples |= {row["Sample No."].strip() for row in csv.DictReader(fh)}

    assert samples, "no sample numbers found in the fixtures"
    non_qc = sorted(s for s in samples if not _QC_RE.match(s))
    assert not non_qc, f"non-QC sample numbers in tests/data/fixtures/: {non_qc[:10]}"


def test_no_fixture_is_gitignored():
    """The .gitignore's patient-data rules (``*.116.csv``, ``XN_SAMPLE*.csv``)
    match fixture names too: an ignored fixture passes here and is missing from
    every clone.  Un-ignore it in .gitignore, under the QC fixture exceptions."""
    ignored = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", str(_FIXTURES)],
        capture_output=True, text=True, check=True, cwd=_FIXTURES,
    ).stdout.splitlines()
    assert not ignored, f"gitignored fixtures: {ignored}"
