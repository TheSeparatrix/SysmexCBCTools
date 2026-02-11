"""Integration test: verify that importing XNSampleProcessor succeeds
even when optional dependencies (pygam, scipy, sklearn, torch) are absent.

This runs the import in a subprocess with those packages blocked via
a custom meta-path finder, so the test is independent of the current
environment's installed packages.
"""

from __future__ import annotations

import subprocess
import sys

# Shared blocker snippet used by both tests.  Uses the modern
# importlib.abc.MetaPathFinder protocol (find_spec) so it works
# reliably on Python 3.12+.
_BLOCKER = """\
import importlib.abc
import sys

_blocked = {"pygam", "scipy", "sklearn", "torch", "ot"}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _blocked or any(
            fullname.startswith(b + ".") for b in _blocked
        ):
            raise ImportError(f"Blocked for testing: {fullname}")
        return None

sys.meta_path.insert(0, _Blocker())
"""


def test_import_xnsampleprocessor_without_optional_deps():
    """Importing XNSampleProcessor must not require optional packages."""
    script = _BLOCKER + (
        "from sysmexcbctools import XNSampleProcessor\n"
        "assert XNSampleProcessor is not None\n"
        "print('ok')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Import failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ok" in result.stdout


def test_optional_classes_are_none_without_deps():
    """Optional classes should be None when their deps are blocked."""
    script = _BLOCKER + (
        "import sysmexcbctools\n"
        "assert sysmexcbctools.GAMCorrector is None\n"
        "assert sysmexcbctools.FlowTransformer is None\n"
        "assert sysmexcbctools.ImpedanceTransformer is None\n"
        "assert sysmexcbctools.XNSampleTransformer is None\n"
        "assert sysmexcbctools.DisAE is None\n"
        "print('ok')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Import failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ok" in result.stdout
