import json
import os
import re
import socket
from pathlib import Path
from typing import Annotated, Literal

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest
from pydantic import BaseModel, Field

import decisionsmith as ds

TEXTS = {
    "billing": ["you charged me twice", "my invoice is wrong", "the payment failed", "refund my card charge"],
    "technical": ["the app crashes on login", "the page shows an error", "sync is broken", "uploads time out"],
    "sales": ["how much is the plan", "can I get a quote", "do you offer discounts", "I want to upgrade"],
}


class Ticket(BaseModel):
    """A support ticket."""

    team: Annotated[
        Literal["billing", "technical", "sales"], ds.Options(billing="payments and refunds", sales="pricing")
    ] = Field(description="Which team should handle this?")
    wants_refund: bool = Field(description="Does the customer ask for a refund?")


def truth(text: str) -> dict:
    team = next(t for t, xs in TEXTS.items() if any(x in text for x in xs))
    return {"team": team, "wants_refund": "refund" in text}


def corpus(n: int) -> list[str]:
    out = []
    i = 0
    while len(out) < n:
        for team, xs in TEXTS.items():
            out.append("%s please %d" % (xs[i % len(xs)], i))
        i += 1
    return out[:n]


def make_checkpoint(path: Path, words: list[str] | None = None, head_layers: int = 1, seed: int = 0) -> Path:
    import torch
    from laya.common import DecisionModel
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(seed)
    base = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", ",", ".", "?", "!", "_", "`"]
    vocab_words = set(words or [])
    for xs in TEXTS.values():
        for x in xs:
            vocab_words.update(re.findall(r"\w+", x.lower()))
    vocab_words.update(
        [
            "a",
            "support",
            "ticket",
            "which",
            "team",
            "should",
            "handle",
            "this",
            "does",
            "the",
            "customer",
            "ask",
            "for",
            "refund",
            "choice",
            "score",
            "noul",
            "question",
            "level",
            "no",
            "yes",
            "statement",
            "holds",
            "not",
            "hold",
            "true",
            "false",
            "billing",
            "technical",
            "sales",
            "payments",
            "and",
            "refunds",
            "pricing",
            "please",
            "is",
            "it",
            "what",
            "rate",
            "low",
            "medium",
            "high",
            "urgency",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "connectivity",
            "test",
        ]
    )
    vocab = {w: i for i, w in enumerate(base + sorted(vocab_words - set(base)))}
    tk = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        max_position_embeddings=128,
    )
    path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=head_layers)
    save_file(model.state_dict(), path / "model.safetensors")
    cfg = {
        "encoder": "unused/offline",
        "head_layers": head_layers,
        "act_costs": {"act": 0},
        "max_len": 64,
        "head_max_len": 48,
        "temperature": [1.0, 1.0, 1.0],
    }
    (path / "rl_agent_config.json").write_text(json.dumps(cfg))
    return path


@pytest.fixture
def no_network(monkeypatch):
    """Every connection outside this machine fails: examples, notebooks and integration tests run offline."""
    real = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost") and not str(host).startswith("/"):
            raise OSError("network is blocked in this test (tried %r)" % (address,))
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory) -> Path:
    return make_checkpoint(tmp_path_factory.mktemp("tiny"))


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "decisions.db")


@pytest.fixture
def teacher():
    return ds.testing.FakeEngine(truth, confidence=1.0, name="teacher")


@pytest.fixture
def toy_csv(tmp_path) -> str:
    import csv

    path = tmp_path / "toy.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "team", "wants_refund", "group"])
        w.writeheader()
        for i, text in enumerate(corpus(60)):
            t = truth(text)
            w.writerow(
                {
                    "id": "r%d" % i,
                    "text": text,
                    "team": t["team"],
                    "wants_refund": str(t["wants_refund"]),
                    "group": str(i // 3),
                }
            )
    return str(path)
