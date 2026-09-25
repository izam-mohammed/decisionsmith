---
name: decisionsmith-build
description: "Build a decision model with no LLM API key: you (the coding agent) label a golden dataset through the decisionsmith MCP tools, train Laya, evaluate it and wire it into the app. Use for /decisionsmith:build, or when the user wants a classifier, router or screener trained on their own data."
---
# Build a model (you are the labeller)

You label the data yourself through the decisionsmith MCP tools (`decisionsmith mcp`); no LLM API key is used.
Stop at every **approval** point and wait for the user.

1. **Look at the data.** Read a sample of real texts and run `data_check(path)`. Propose the labels (one answer per
   text) or a Pydantic schema (several answers), with a one-line description per option. **Approval.**
2. **Pick the rows.** `golden_start(source, labels=[...] or schema=..., n=200, strategy="diverse", agent="claude-code")`
   (a harness log `.db` can use `strategy="uncertain"`). CLI alternative:
   `decisionsmith golden tickets.csv --labels a,b,c --teacher agent:claude-code --strategy diverse`.
3. **Label.** Start a *fresh* `labeler` subagent for each batch (it calls `golden_batch(session)` once, then
   `golden_submit`), until `remaining` is 0. A share of the texts comes back in a later batch under a new id for a
   second, independent answer; the labeler has only the two labelling tools (no files, no `golden_status`), so it
   can't see the first answer. Don't label batches yourself and don't pass answers between labelers. Then
   `golden_status(session)` and show the user the disagreements and skipped rows.
4. **Short of data?** If `data_check` or `golden_status` shows an option under 10 rows, ask the user, then have the
   `data-writer` subagent add varied examples with `golden_add`. Show it samples from `split=train` rows only (see
   `split` in golden.session.json), never test rows; texts sharing 80% or more of their words with a test text are
   rejected. Fresh `labeler`s give each written text its second answer (step 3).
5. **Train and evaluate.** `golden_finish(session)` writes golden.csv. `finetune("golden.csv", session, out="runs/v1")`
   (ask before long runs), then the `evaluator` subagent runs `evaluate("runs/v1", "golden.csv", session,
   save="models/<name>")`. Show the report table, the agreement with each labeller's labels (accuracy only on
   rows a person labelled), and every go/no-go reason. **Approval.**
6. **Wire it in.** Write the production code: `model = ds.load("models/<name>-v1")` and
   `ds.harness(model, teacher=..., mode="shadow", collect=0.05)` where the app decides today. Show the diff. **Approval.**

Headless (`claude -p "/decisionsmith:build ..."`): labels given in the prompt count as step 1's approval; if the
prompt says to stop after the report, skip step 6. Never skip the second answers or the report.

## Honesty rules
- Report numbers only from `evaluate`, `finetune`, `bench` or `status` output, exactly as measured. Never claim
  accuracy without running `evaluate`; say "not measured yet" instead.
- A number measured against your own labels is agreement with you, not accuracy: call it that, and suggest the
  user checks a sample and fills the rows the two answers disagreed on (left blank in golden.csv).
- Label from the text alone. When unsure or no option fits, skip with a reason; never guess to finish faster.
- Mark written data as synthetic (`golden_add` does); never put it in the test split; say how many rows are synthetic.
- Whether AI outputs may be used to train a model depends on the provider's terms and the user's plan; tell the user
  to check before training on agent-labelled data.
- Never edit golden.csv labels yourself after finishing, never change harness modes, and never send data to hosted
  engines without the user's approval.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
