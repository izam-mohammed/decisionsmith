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
    assert len(skills) == 6
    for s in skills:
        meta, body = front(s)
        assert meta["name"] == s.parent.name and 20 < len(meta["description"]) < 1024 and "Laya" in body
    commands = sorted(PLUGIN.glob("commands/*.md"))
    assert len(commands) == 7 and all(front(c)[0]["description"] for c in commands)
    agents = sorted(PLUGIN.glob("agents/*.md"))
    assert len(agents) == 3 and all(front(a)[0]["name"] == a.stem for a in agents)
