# FAQ

**Why not just use the LLM?** Cost and latency per call, forever. The harness keeps the LLM for what the fast model
isn't sure about and measures when that share can shrink.

**Why not just use Laya?** Zero-shot it is weak on many tasks (0.36 accuracy on typed-decisions before fine-tuning
in Laya's own benchmarks). It needs your data; the harness collects it.

**Can I fine-tune Jev?** No, Jev is hosted. `h.adapt()` calibrates it and picks thresholds from your data, and it
can be a teacher or a student.

**Is my data sent anywhere?** Only to the engines you configure. Laya runs locally; an Ollama teacher keeps
everything on your machine. The log is a local file.

**Does the fine-tuned model need decisionsmith?** No. `runs/v1` is a normal Laya checkpoint.

**Isn't this just distillation?** Yes: continuous, measured per field, calibrated on held-out data, and never
switched on without passing its checks.

**Which loss?** Soft cross-entropy by default. The Laya notebook's RL term is available as `--loss rlcd`, and
`--loss proper` adds spherical and ranked-probability scores, for comparison on your data.

**Credits.** Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai
Innovations. Jev and TypeSafe are trademarks of their owners. decisionsmith is not affiliated with either.
