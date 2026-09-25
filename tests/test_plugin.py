import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"


def front(path):
    text = path.read_text()
    assert text.startswith("---\n"), path
    head, _, body = text[4:].partition("\n---\n")
    return yaml.safe_load(head), body


def test_manifests():
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert market["plugins"][0]["source"] == "./plugin" and plugin["name"] == "decisionsmith"
    mcp = json.loads((PLUGIN / ".mcp.json").read_text())
    assert mcp["mcpServers"]["decisionsmith"]["args"][-1] == "mcp"


def test_skills_commands_agents():
    skills = sorted(PLUGIN.glob("skills/*/SKILL.md"))
    assert len(skills) == 7
    for s in skills:
        meta, body = front(s)
        assert meta["name"] == s.parent.name and 20 < len(meta["description"]) < 1024 and "Laya" in body
    commands = sorted(PLUGIN.glob("commands/*.md"))
    assert len(commands) == 8 and all(front(c)[0]["description"] for c in commands)
    agents = sorted(PLUGIN.glob("agents/*.md"))
    assert len(agents) == 6 and all(front(a)[0]["name"] == a.stem for a in agents)


def test_build_skill_uses_only_real_mcp_tools():
    import re

    from decisionsmith import mcp

    names = {fn.__name__ for fn in mcp.TOOLS}
    files = [PLUGIN / "skills" / "decisionsmith-build" / "SKILL.md", PLUGIN / "commands" / "build.md"]
    files += [PLUGIN / "agents" / ("%s.md" % a) for a in ("labeler", "data-writer", "evaluator")]
    used = {m for f in files for m in re.findall(r"`([a-z_]+)\(", f.read_text())}
    assert used and used <= names, used - names
    skill = files[0].read_text()
    assert skill.count("**Approval.**") == 3 and "## Honesty rules" in skill


def test_build_agents_have_only_the_tools_they_need():
    from decisionsmith import mcp

    prefix = "mcp__plugin_decisionsmith_decisionsmith__"
    names = {fn.__name__ for fn in mcp.TOOLS}
    tools = {
        a: [t.strip() for t in front(PLUGIN / "agents" / ("%s.md" % a))[0]["tools"].split(",")]
        for a in ("labeler", "data-writer", "evaluator")
    }
    for granted in tools.values():
        assert all(t[len(prefix) :] in names for t in granted if t.startswith(prefix)), granted
    assert tools["labeler"] == [prefix + "golden_batch", prefix + "golden_submit"]
    assert tools["data-writer"] == [prefix + "golden_add"]
    assert "Bash" in tools["evaluator"] and prefix + "evaluate" in tools["evaluator"]
    assert not any(t.startswith(prefix + "golden_") for t in tools["evaluator"])
