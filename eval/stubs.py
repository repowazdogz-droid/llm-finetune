"""Stub generate-functions. Shipped, not test-only.

These exist so that the harness can be verified end to end with no GPU and no
model — and so that the PC phase has a smoke test it can run *before* spending
an hour loading weights. If `run_eval(perfect_generator, ...)` does not score
100%, the harness is broken and no amount of fine-tuning will show it.

Each stub is a valid implementation of the harness contract:

    generate(prompts: Sequence[str]) -> Sequence[str]

They read the reference answer back out of the prompt where they need it, which
is a legitimate thing for a stub to do and an illegitimate thing for a model to
do — the distinction is why these live behind names that say what they are.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"

QUESTION_RE = re.compile(r"Problem:\s*(.*?)\n\nSolution:", re.DOTALL)


def _questions_from(prompts: Sequence[str]) -> list[str]:
    out = []
    for p in prompts:
        m = QUESTION_RE.search(p)
        out.append(m.group(1).strip() if m else p)
    return out


def _answer_lookup() -> dict[str, dict]:
    """Map normalised question text -> record, over train + eval."""
    from eval.harness import load_jsonl

    table = {}
    for name in ("train.jsonl", "eval.jsonl"):
        path = PROCESSED / name
        if path.exists():
            for r in load_jsonl(path):
                table[" ".join(r["question"].lower().split())] = r
    return table


class PerfectGenerator:
    """Answers every question correctly, in the requested format.

    The upper bound. Scores 100% by construction, so a run that does not is a
    harness bug rather than a model result.
    """

    def __init__(self):
        self.table = _answer_lookup()

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        out = []
        for q in _questions_from(prompts):
            r = self.table.get(" ".join(q.lower().split()))
            if r is None:
                out.append("I don't know.")
            else:
                out.append(f"{r['solution']}\n#### {r['answer']}")
        return out


class MemorisingGenerator(PerfectGenerator):
    """Reproduces the *training* solution verbatim, whatever it was asked.

    The positive control for the memorisation check. It emits real training
    documents, so a detector that fails to flag this output is not working.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        from eval.harness import load_jsonl

        self.train = load_jsonl(PROCESSED / "train.jsonl")
        self.rng = random.Random(seed)

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        return [
            f"{r['solution']}\n#### {r['answer']}"
            for r in (self.rng.choice(self.train) for _ in prompts)
        ]


class ConstantGenerator:
    """Always answers 42. The floor: near-zero accuracy, perfect format."""

    def __init__(self, value: str = "42"):
        self.value = value

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        return [f"The answer is obvious.\n#### {self.value}"] * len(prompts)


class UnformattedGenerator(PerfectGenerator):
    """Correct answers, but never uses the `####` marker.

    Separates format compliance from correctness: this should score high under
    the last-number fallback and near zero under `strict=True`.
    """

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        out = []
        for q in _questions_from(prompts):
            r = self.table.get(" ".join(q.lower().split()))
            out.append(
                f"{r['solution']}\nSo the answer is {r['answer']}."
                if r
                else "No idea whatsoever."
            )
        return out


class RaisingGenerator:
    """Raises on every call. Verifies the harness records rather than crashes."""

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        raise RuntimeError("simulated inference failure")


class WrongLengthGenerator:
    """Returns the wrong number of outputs. Must be rejected, not silently zipped.

    A short return from a real batched inference call would otherwise misalign
    every prediction after it against the wrong reference — scoring nonsense
    that looks like a bad model rather than a broken pipeline.
    """

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        return ["only one output"]
