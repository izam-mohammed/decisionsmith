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
3. **Label.** Hand batches to the `labeler` subagent: `golden_batch(session)`, then
   `golden_submit(session, answers, agent="claude-code")`, until `pass` becomes 2. Pass 2 is the blind re-check: give it
   to a *fresh* `labeler` that has not seen pass 1. Then `golden_status(session)` and show the user the disagreements
   and skipped rows.
4. **Short of data?** If `data_check` or `golden_status` shows an option under 10 rows, ask the user, then have the
   `data-writer` subagent add varied examples with `golden_add` and a fresh `labeler` re-check them (pass 2).
5. **Train and evaluate.** `golden_finish(session)` writes golden.csv. `finetune("golden.csv", session, out="runs/v1")`
   (ask before long runs), then the `evaluator` subagent runs `evaluate("runs/v1", "golden.csv", session,
   save="models/<name>")`. Show the report table, accuracy by who labelled, and every go/no-go reason. **Approval.**
6. **Wire it in.** Write the production code: `model = ds.load("models/<name>-v1")` and
   `ds.harness(model, teacher=..., mode="shadow", collect=0.05)` where the app decides today. Show the diff. **Approval.**

Headless (`claude -p "/decisionsmith:build ..."`): labels given in the prompt count as step 1's approval; if the
prompt says to stop after the report, skip step 6. Never skip the re-check or the report.

## Honesty rules
- Report numbers only from `evaluate`, `finetune`, `bench` or `status` output, exactly as measured. Never claim
  accuracy without running `evaluate`; say "not measured yet" instead.
- Accuracy against your own labels measures agreement with you, not truth: say so, and suggest the user checks a
  sample and fills the rows the passes disagreed on (left blank in golden.csv).
- Label from the text alone. When unsure or no option fits, skip with a reason; never guess to finish faster.
- Mark written data as synthetic (`golden_add` does); never put it in the test split; say how many rows are synthetic.
- Whether AI outputs may be used to train a model depends on the provider's terms and the user's plan; tell the user
  to check before training on agent-labelled data.
- Never edit golden.csv labels yourself after finishing, never change harness modes, and never send data to hosted
  engines without the user's approval.
- Built on Laya (Apache-2.0, Nandakishor M / Convai Innovations). Not affiliated with TypeSafe AI or Convai.
