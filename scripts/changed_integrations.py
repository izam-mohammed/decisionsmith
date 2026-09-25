"""Print the integration jobs to run as a JSON list, given the changed file paths on stdin.

Core changes run every integration; a change inside one integration's module, test or examples runs only that one.
With --all, print every integration.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = (
    "src/decisionsmith/",
    "tests/conftest.py",
    "tests/test_examples.py",
    "tests/integrations/__init__.py",
    "tests/integrations/conftest.py",
    "tests/integrations/openai_mock.py",
    "tests/integrations/pyproject.toml",
    "scripts/integration.sh",
    "scripts/check_coverage.py",
    "scripts/changed_integrations.py",
    ".github/workflows/integrations.yml",
    "pyproject.toml",
    "uv.lock",
)


def names() -> list[str]:
    text = (ROOT / "tests/integrations/pyproject.toml").read_text()
    return sorted(n for n in re.findall(r"^int-([a-z0-9-]+)\s*=", text, re.M) if n != "base")


def owner(path: str, all_names: list[str]) -> str | None:
    for name in all_names:
        module = name.replace("-", "_")
        if path in (f"src/decisionsmith/integrations/{module}.py", f"tests/integrations/test_{module}.py"):
            return name
        if path.startswith(f"examples/04-integrations/{name}/"):
            return name
    return None


def select(paths: list[str], all_names: list[str]) -> list[str]:
    picked: set[str] = set()
    for path in paths:
        name = owner(path, all_names)
        if name:
            picked.add(name)
        elif path.startswith(CORE):
            return all_names
    return sorted(picked)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    all_names = names()
    paths = [p.strip().replace("\\", "/") for p in sys.stdin if p.strip()]
    print(json.dumps(all_names if "--all" in argv else select(paths, all_names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
