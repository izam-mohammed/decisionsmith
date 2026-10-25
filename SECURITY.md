# Security policy

## Supported versions

decisionsmith is young, so only the latest release on [PyPI](https://pypi.org/project/decisionsmith/) gets
security fixes. Please upgrade (`uv add decisionsmith --upgrade-package decisionsmith`) before reporting.

| Version | Supported |
|---|---|
| 0.1.x (latest) | yes |
| older | no |

## Reporting a vulnerability

Please do not open a public issue, pull request or discussion for a security problem.

Report it privately through GitHub:
[Report a vulnerability](https://github.com/izam-mohammed/decisionsmith/security/advisories/new)
(the "Security" tab, then "Report a vulnerability").

Helpful details:

- the version and how you installed it
- what an attacker could do, and what they need first (for example a crafted log file, model folder or HTTP response)
- the smallest steps or script that show it
- any fix you have in mind

Please leave out real API keys and real customer data.

## What happens next

This is a project with one maintainer, so there are no guaranteed response times. We aim to:

- acknowledge the report within a few days
- agree on whether it is a vulnerability and how serious it is
- prepare a fix in a private fork, release it, and publish a GitHub security advisory that credits you (unless you
  would rather not be named)

Please give us a reasonable chance to release a fix before you share details publicly.

## Scope

In scope: the `decisionsmith` package, its CLI and MCP server, and this repository's release workflow.
Problems in the engines it talks to (hosted LLMs, Jev, Laya) should go to those projects; tell us too if
decisionsmith makes them worse.
