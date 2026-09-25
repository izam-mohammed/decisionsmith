"""Rewrite `[tool.uv] conflicts` in pyproject.toml: every framework extra and `int-*` group is mutually exclusive.

The universal lock would otherwise have to find one version set that satisfies every framework at once. Users
installing from PyPI are not affected: pip and uv resolve their own combination of extras freshly.
"""

import re
import sys
from pathlib import Path

CORE_EXTRAS = {"laya", "anthropic", "mcp", "all"}
# frameworks that pin an older mcp than the `mcp` extra needs (mcp >= 2.2)
OLD_MCP = ["crewai", "semantic-kernel"]
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def main() -> int:
    s = PYPROJECT.read_text()
    extras_block = re.search(r"\[project.optional-dependencies\]\n(.*?)\n\n", s, re.S).group(1)
    extras = [e for e in re.findall(r"^([a-z0-9-]+) = ", extras_block, re.M) if e not in CORE_EXTRAS]
    groups = [g for g in re.findall(r"^(int-[a-z0-9-]+) = ", s, re.M) if g != "int-base"]
    items = ['    { extra = "%s" },' % e for e in sorted(extras)] + [
        '    { group = "%s" },' % g for g in sorted(groups)
    ]
    block = "conflicts = [\n  [\n%s\n  ],\n" % "\n".join(items)
    for name in OLD_MCP:
        for core in ("mcp", "all"):
            block += '  [{ group = "int-%s" }, { extra = "%s" }],\n' % (name, core)
            block += '  [{ extra = "%s" }, { extra = "%s" }],\n' % (name, core)
    block += "]"
    new = re.sub(r"conflicts = \[\n.*?\n\]", block, s, flags=re.S)
    if "--check" in sys.argv:
        return 0 if new == s else 1
    PYPROJECT.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
