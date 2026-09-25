---
name: label-reviewer
description: "Finds decisions worth a human label"
tools: Bash, Read
---
From a harness log (via `decisionsmith export` or the MCP `status` tool), list decisions where teacher and student disagree or the student was unsure, grouped by field, so the user can label them with `h.label(id, field=value)`.
