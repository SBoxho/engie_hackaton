"""Repository verification entrypoint.

The app is Streamlit based, so there is no static production build artifact.
This command treats bytecode compilation plus Streamlit AppTest smoke coverage
as the build check, and runs optional quality tools when they are installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_TARGETS = ("app", "src", "scripts", "tests")
TEXT_TARGETS = (
    ".github",
    ".streamlit",
    "app",
    "data/config",
    "docs",
    "scripts",
    "src",
    "tests",
)
ROOT_TEXT_FILES = (
    ".env.example",
    ".gitignore",
    "pytest.ini",
    "README.md",
    "requirements-dev.txt",
    "requirements.txt",
    "runtime.txt",
)
TEXT_SUFFIXES = {".md", ".py", ".json", ".toml", ".yml", ".yaml", ".txt", ".ini"}
SECRET_PATTERNS = (
    "BEGIN PRIVATE KEY",
    "RTE_ECOWATT_API_TOKEN=\"",
    "ENTSOE_API_TOKEN=\"",
    "api_key=",
    "apikey=",
    "password=",
)


def _print_header(title: str) -> None:
    print(f"\n== {title} ==", flush=True)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return int(completed.returncode)


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for target in TEXT_TARGETS:
        path = ROOT / target
        if not path.exists():
            continue
        files.extend(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES
        )
    files.extend(ROOT / name for name in ROOT_TEXT_FILES if (ROOT / name).exists())
    return sorted(set(files))


def check_formatting() -> int:
    _print_header("Formatting checks")
    failures: list[str] = []
    files = _iter_text_files()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip() != line:
                failures.append(f"{path.relative_to(ROOT)}:{number}: trailing whitespace")
        if "\t" in text and path.suffix.lower() in {".md", ".yml", ".yaml", ".toml", ".ini"}:
            failures.append(f"{path.relative_to(ROOT)}: contains tab characters")
        if any(
            line.startswith(("<<<<<<< ", ">>>>>>> ")) or line == "======="
            for line in text.splitlines()
        ):
            failures.append(f"{path.relative_to(ROOT)}: possible merge conflict marker")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"Checked {len(files)} text files.")
    return 0


def run_lint() -> int:
    _print_header("Lint checks")
    status = 0
    if _module_available("ruff"):
        status |= _run([sys.executable, "-m", "ruff", "check", *PYTHON_TARGETS])
    else:
        print("ruff not installed; using Python parser/bytecode checks as the available lint gate.")
    status |= _run([sys.executable, "-m", "compileall", "-q", *PYTHON_TARGETS])
    return status


def run_bootstrap_check() -> int:
    _print_header("Bootstrap check")
    return _run([sys.executable, "-m", "scripts.bootstrap"])


def run_security_checks(*, strict_tools: bool) -> int:
    _print_header("Security checks")
    failures: list[str] = []
    for path in _iter_text_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in {".env.example", "scripts/verify.py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lowered = text.lower()
        for pattern in SECRET_PATTERNS:
            if pattern.lower() in lowered:
                failures.append(f"{rel}: possible secret pattern {pattern!r}")
        if rel.startswith("app/") and ("rte_ecowatt_api_token" in lowered or "entsoe_api_token" in lowered):
            failures.append(f"{rel}: frontend bundle references token configuration")
    if failures:
        print("\n".join(failures))
        return 1
    print("No obvious committed secrets or frontend token references found.")

    if _module_available("pip_audit"):
        return _run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                "requirements.txt",
                "--requirement",
                "requirements-dev.txt",
                "--progress-spinner",
                "off",
            ]
        )
    if strict_tools:
        print("pip-audit is not installed and --strict-quality-tools was requested.")
        return 1
    print("pip-audit not installed; dependency vulnerability audit skipped in this environment.")
    return 0


def run_type_checks(*, strict_tools: bool) -> int:
    _print_header("Type checks")
    if _module_available("mypy"):
        return _run([sys.executable, "-m", "mypy", "app", "src", "scripts"])
    if strict_tools:
        print("mypy is not installed and --strict-quality-tools was requested.")
        return 1
    print("mypy not installed; no configured type checker is available in this repo.")
    return 0


def run_tests() -> int:
    _print_header("Unit and Streamlit smoke tests")
    env = os.environ.copy()
    env.setdefault("APP_MODE", "demo")
    env.setdefault("DEMO_ALLOW_EXTERNAL_API", "0")
    return _run([sys.executable, "-m", "pytest", "-q"], env=env)


def run_build_check() -> int:
    _print_header("Production build check")
    print("Streamlit has no static build step; bytecode build and AppTest smoke are covered above.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository verification gates.")
    parser.add_argument(
        "--strict-quality-tools",
        action="store_true",
        help="Fail when optional external quality tools such as mypy are unavailable.",
    )
    args = parser.parse_args()

    status = 0
    status |= run_bootstrap_check()
    status |= check_formatting()
    status |= run_lint()
    status |= run_type_checks(strict_tools=args.strict_quality_tools)
    status |= run_security_checks(strict_tools=args.strict_quality_tools)
    status |= run_tests()
    status |= run_build_check()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
