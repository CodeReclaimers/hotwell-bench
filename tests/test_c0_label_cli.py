"""C0: label_items.py takes the master seed from a file; the test split refuses --master."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "label_items.py"


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def _module():
    spec = importlib.util.spec_from_file_location("label_items", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_test_split_refuses_seed_on_command_line(tmp_path):
    r = _cli("--master", "123", "--preset", "P01", "--split", "test", "--indices", "0",
             "--cache", str(tmp_path / "c"), "--out", str(tmp_path / "o"))
    print(r.returncode, r.stderr.strip().splitlines()[-1])
    assert r.returncode == 2 and "--master-seed-file, not --master" in r.stderr


def test_seed_flags_are_exclusive_and_one_is_required(tmp_path):
    f = tmp_path / "seed"
    f.write_text("123\n")
    both = _cli("--master", "123", "--master-seed-file", str(f), "--preset", "P01",
                "--indices", "0")
    neither = _cli("--preset", "P01", "--indices", "0")
    print(both.stderr.strip().splitlines()[-1], "|", neither.stderr.strip().splitlines()[-1])
    assert both.returncode == 2 and "not allowed with" in both.stderr
    assert neither.returncode == 2 and "one of the arguments" in neither.stderr


def test_read_seed(tmp_path):
    mod = _module()
    f = tmp_path / "seed"
    f.write_text("  20261107\n")
    assert mod.read_seed(str(f)) == 20261107
    for bad in ("", "-5", "12 34", "0x10", "1.5"):
        f.write_text(bad)
        with pytest.raises(SystemExit):
            mod.read_seed(str(f))


def test_preset_is_required_and_must_be_registered(tmp_path):
    """C4: no default preset; an unknown name is refused before anything runs."""
    missing = _cli("--master", "123", "--indices", "0")
    unknown = _cli("--master", "123", "--preset", "P99", "--indices", "0")
    print(missing.stderr.strip().splitlines()[-1], "|", unknown.stderr.strip().splitlines()[-1])
    assert missing.returncode == 2 and "--preset" in missing.stderr
    assert unknown.returncode == 2 and "invalid choice" in unknown.stderr
