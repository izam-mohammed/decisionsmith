# Notebooks

Each notebook makes its own synthetic data, so it runs anywhere (a laptop, Colab, Kaggle). CI runs every notebook
top to bottom offline (`DS_OFFLINE=1`, a tiny Laya checkpoint) so none of them rot; the committed copies keep the
outputs of real runs where the hardware allowed it. After each run, local absolute paths in the outputs were
replaced (`<venv>`, `.`, `~`); no numbers were changed.

| notebook | what it does | runs where | committed outputs |
|---|---|---|---|
| [01_quickstart](01_quickstart.ipynb) | `ds.model(labels)`, zero-shot, head-only `.train()`, save and load | laptop CPU/MPS, about a minute | real run, Laya English on Apple MPS |
| [02_generate_golden_dataset](02_generate_golden_dataset.ipynb) | `model.generate(n, teacher=...)`, review the CSV, train, test on your own texts | needs an LLM key or Ollama | none (paid API) |
| [03_harness_shadow_to_cascade](03_harness_shadow_to_cascade.ipynb) | shadow mode, `status()`, `adapt()`, `finetune()`, move ready fields to cascade | laptop, a few minutes | real run, stand-in teacher, real Laya student |
| [04_finetune_full_gpu](04_finetune_full_gpu.ipynb) | full fine-tune, DDP on 2x T4, report, download | Kaggle / Colab GPU | none (needs a GPU) |
| [05_multilingual_heads](05_multilingual_heads.ipynb) | `laya:multilingual` + head-only training ([laya#320](https://github.com/NandhaKishorM/laya/issues/320)) | laptop or GPU | real run, Laya multilingual on Apple MPS |
| [06_typed_decisions_reproduction](06_typed_decisions_reproduction.ipynb) | three losses x three seeds on typed-decisions | Kaggle 2x T4 | none (needs GPUs and the data) |
| [07_bench_llm_vs_jev_vs_laya](07_bench_llm_vs_jev_vs_laya.ipynb) | `ds.bench` base vs fine-tuned Laya (plus an LLM and Jev when keys are set), coverage curve | laptop (+ keys) | real run, Laya parts only |

The stand-in teacher in 03 answers from the synthetic labels, the way a perfect LLM would; set `DS_TEACHER` to use
a real one. Numbers in the outputs come from these small synthetic sets and say nothing about your data.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations.
decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
