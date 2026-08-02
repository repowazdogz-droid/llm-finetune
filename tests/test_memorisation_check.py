"""Floor B0 gate, part 2: the memorisation check detects what it claims to.

A detector is only worth running if it fires on the thing it is for and stays
quiet on the thing it is not for. Both directions are tested here with
constructed inputs whose correct verdict is known:

  positive control — output copied verbatim from the training corpus MUST flag.
  negative control — held-out text that was never in the corpus MUST NOT flag.

Without the negative control a detector that flags everything would pass; without
the positive control one that flags nothing would pass. Neither alone is a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.harness import load_jsonl
from eval.memorisation_check import (
    DEFAULT_FLAG_TOKENS,
    MemorisationIndex,
    build_index_from_training,
    check_outputs,
    format_report,
    ngrams,
    tokenise,
)
from eval.stubs import MemorisingGenerator

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"


@pytest.fixture(scope="module")
def train_rows():
    return load_jsonl(PROCESSED / "train.jsonl")


@pytest.fixture(scope="module")
def eval_rows():
    return load_jsonl(PROCESSED / "eval.jsonl")


@pytest.fixture(scope="module")
def index(train_rows):
    """A partial index — the first 2000 documents. Fast, and enough for most tests."""
    return MemorisationIndex((r["solution"] for r in train_rows[:2000]), n=8)


@pytest.fixture(scope="module")
def full_index(train_rows):
    """An index over the whole training corpus.

    Needed wherever the candidate text is drawn from *anywhere* in training.
    `MemorisingGenerator` samples across all 6973 documents, so scoring it
    against a 2000-document index flags only ~5/30 — not because the detector
    failed but because 70% of the candidates genuinely are not in that index.
    Matching the index to the corpus the candidates came from is the difference
    between testing the detector and testing the fixture.
    """
    return MemorisationIndex((r["solution"] for r in train_rows), n=8)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def test_tokenise_is_case_insensitive_and_splits_punctuation():
    assert tokenise("Natalia sold 48 clips.") == ["natalia", "sold", "48", "clips", "."]
    assert tokenise("ABC") == tokenise("abc")


def test_ngrams_shape():
    toks = ["a", "b", "c", "d"]
    assert ngrams(toks, 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert ngrams(toks, 5) == []


def test_index_reports_its_own_size(index):
    assert len(index) > 0
    assert index.n_documents == 2000
    assert index.n_tokens > 0


# ---------------------------------------------------------------------------
# POSITIVE CONTROL — verbatim training text must flag
# ---------------------------------------------------------------------------

def test_verbatim_training_text_is_flagged(index, train_rows):
    """The detector's whole job. If this fails, it is not a detector."""
    candidates = [r["solution"] for r in train_rows[:50]]
    r = check_outputs(candidates, index, flag_tokens=DEFAULT_FLAG_TOKENS)
    assert r["flagged_rate"] > 0.9, (
        f"only {r['n_flagged']}/50 verbatim training solutions were flagged"
    )


def test_memorising_stub_is_caught_end_to_end(full_index):
    """The full path: a generate-function that regurgitates gets flagged.

    Asserted at 28/30 rather than 30/30 on purpose. Calibration measured a 4.7%
    false-negative rate at this threshold — very short training solutions do not
    contain a 25-token span — so demanding 30/30 would be demanding the detector
    exceed its own measured accuracy, and the test would be flaky by design.
    """
    gen = MemorisingGenerator(seed=0)
    outputs = gen([f"Problem: x{i}\n\nSolution:" for i in range(30)])
    r = check_outputs(outputs, full_index, flag_tokens=DEFAULT_FLAG_TOKENS)
    assert r["n_flagged"] >= 28, f"only {r['n_flagged']}/30 regurgitations flagged"
    assert r["longest_match_tokens"]["max"] > DEFAULT_FLAG_TOKENS


def test_a_long_copied_span_inside_novel_text_is_still_caught(index, train_rows):
    """Copying buried in original prose must not hide from the detector."""
    stolen = train_rows[0]["solution"]
    candidate = (
        "Let me think about this from first principles for a while. "
        + stolen
        + " And that concludes my entirely original reasoning."
    )
    r = check_outputs([candidate], index, flag_tokens=DEFAULT_FLAG_TOKENS)
    assert r["n_flagged"] == 1


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL — the detector must not flag everything
# ---------------------------------------------------------------------------

def test_held_out_solutions_are_not_flagged(index, eval_rows):
    """Never-seen text must stay quiet, or the detector is a rubber stamp."""
    candidates = [r["solution"] for r in eval_rows[:100]]
    r = check_outputs(candidates, index, flag_tokens=DEFAULT_FLAG_TOKENS)
    assert r["flagged_rate"] < 0.05, (
        f"{r['n_flagged']}/100 held-out solutions were falsely flagged"
    )


def test_completely_unrelated_text_is_not_flagged(index):
    candidates = [
        "The mitochondrion is the powerhouse of the cell, as every schoolchild knows.",
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
        "The quick brown fox jumps over the lazy dog repeatedly and without complaint.",
    ]
    r = check_outputs(candidates, index, flag_tokens=DEFAULT_FLAG_TOKENS)
    assert r["n_flagged"] == 0
    assert r["longest_match_tokens"]["max"] < DEFAULT_FLAG_TOKENS


def test_short_incidental_overlap_does_not_flag(index):
    """A common phrase is not copying."""
    r = check_outputs(["How many did she have left?"], index,
                      flag_tokens=DEFAULT_FLAG_TOKENS)
    assert r["n_flagged"] == 0


# ---------------------------------------------------------------------------
# separation, and the reported limit
# ---------------------------------------------------------------------------

def test_positive_and_negative_populations_separate(index, train_rows, eval_rows):
    """The distributions must not overlap in their bulk, or the threshold is arbitrary."""
    seen = check_outputs([r["solution"] for r in train_rows[:100]], index)
    unseen = check_outputs([r["solution"] for r in eval_rows[:100]], index)
    assert seen["longest_match_tokens"]["median"] > (
        unseen["longest_match_tokens"]["median"] + DEFAULT_FLAG_TOKENS
    )


def test_empty_and_tiny_inputs_do_not_crash(index):
    r = check_outputs(["", "x", "one two three"], index)
    assert r["n_flagged"] == 0
    assert r["n_candidates"] == 3


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_clean_report_refuses_to_claim_no_memorisation(index, eval_rows):
    """A zero result must be stated as 'no verbatim copying', never as 'no memorisation'."""
    r = check_outputs([r["solution"] for r in eval_rows[:20]], index)
    text = format_report(r)
    assert "verbatim copying only" in text
    assert "Do not report it as an absence of memorisation." in text


def test_flagged_report_lists_the_offenders(index, train_rows):
    r = check_outputs([r["solution"] for r in train_rows[:5]], index)
    text = format_report(r)
    assert "## Flagged" in text
    assert "a verdict" in text  # "A flag is a prompt to look, not a verdict."
