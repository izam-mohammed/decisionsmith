"""Shared bits for the examples: the toy data and a stand-in teacher that needs no API key."""

import csv
import os

import decisionsmith as ds

HERE = os.path.dirname(os.path.abspath(__file__))
TOY = os.path.join(HERE, "data", "toy.csv")


def toy_rows() -> list[dict]:
    with open(TOY, newline="") as f:
        return list(csv.DictReader(f))


def teacher():
    """`DS_TEACHER` (e.g. claude-haiku-4-5), or a stand-in that answers from the toy labels."""
    if os.environ.get("DS_TEACHER"):
        return os.environ["DS_TEACHER"]
    labels = {r["text"]: {"team": r["team"], "wants_refund": r["wants_refund"]} for r in toy_rows()}
    return ds.testing.FakeEngine(labels, confidence=1.0, name="stand-in-llm")
