"""Floor B0 gate, part 1: the evaluation harness works, verified with stubs.

The harness is the instrument every later number depends on, so it is tested
against generators whose correct score is known in advance. A harness that
reported 60% for the perfect generator would make every B1/B2 result
uninterpretable, and nothing downstream would reveal it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.prepare import normalise_answer as prepare_normalise
from eval.harness import (
    DEFAULT_PROMPT,
    extract_answer,
    load_eval_set,
    normalise_answer,
    run_eval,
    wilson_interval,
)
from eval.stubs import (
    ConstantGenerator,
    PerfectGenerator,
    RaisingGenerator,
    UnformattedGenerator,
    WrongLengthGenerator,
)

REPO = Path(__file__).resolve().parent.parent
N = 40  # enough for the rates to be meaningful, small enough to stay fast


@pytest.fixture(scope="module")
def examples():
    return load_eval_set(limit=N)


# ---------------------------------------------------------------------------
# the two normalisers must not drift apart
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    ["72", " 72 ", "1,000", "$5", "5%", "72.", "-3", "0", "3.5", "72.0",
     "", "abc", "1,234,567", "  "],
)
def test_harness_and_prepare_normalisers_agree(raw):
    """If these drift, every accuracy number silently becomes uninterpretable."""
    assert normalise_answer(raw) == prepare_normalise(raw)


def test_normaliser_canonicalises_equivalent_forms():
    assert normalise_answer("1,000") == normalise_answer("1000") == "1000"
    assert normalise_answer("72.0") == normalise_answer("72") == "72"
    assert normalise_answer("$5") == "5"
    assert normalise_answer("abc") is None


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def test_extract_prefers_the_marker():
    got, how = extract_answer("blah 999 blah\n#### 72")
    assert (got, how) == ("72", "marker")


def test_extract_falls_back_to_last_number():
    got, how = extract_answer("I think it is 12, no wait, 34")
    assert (got, how) == ("34", "last_number")


def test_strict_mode_refuses_the_fallback():
    got, how = extract_answer("I think it is 34", strict=True)
    assert (got, how) == (None, "none")


def test_extract_returns_none_when_there_is_no_number():
    assert extract_answer("no numbers here at all") == (None, "none")


def test_marker_wins_even_when_a_later_number_exists():
    """A trailing sentence after the marker must not hijack the answer."""
    got, how = extract_answer("#### 72\nThanks for reading, see problem 99.")
    assert (got, how) == ("72", "marker")


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------

def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_wilson_stays_in_bounds_at_the_extremes():
    for k, n in [(0, 10), (10, 10), (0, 1), (1, 1)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_narrows_with_n():
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# ---------------------------------------------------------------------------
# end-to-end against stubs whose scores are known in advance
# ---------------------------------------------------------------------------

def test_perfect_generator_scores_one_hundred_percent(examples):
    r = run_eval(PerfectGenerator(), examples, run_name="stub_perfect")
    assert r["accuracy"] == 1.0
    assert r["format_compliance"] == 1.0
    assert r["extraction_strategies"].get("marker") == len(examples)
    assert r["n_correct"] == len(examples)


def test_constant_generator_scores_near_zero(examples):
    """Answering 42 every time: format-perfect, essentially always wrong."""
    r = run_eval(ConstantGenerator("42"), examples, run_name="stub_constant")
    assert r["accuracy"] < 0.10
    assert r["format_compliance"] == 1.0


def test_unformatted_generator_shows_the_fallback_doing_work(examples):
    """Right answers, wrong format: high under fallback, low under strict."""
    lenient = run_eval(UnformattedGenerator(), examples, run_name="stub_lenient")
    strict = run_eval(
        UnformattedGenerator(), examples, strict=True, run_name="stub_strict"
    )

    assert lenient["extraction_strategies"].get("last_number") == len(examples)
    assert lenient["accuracy"] > 0.9
    assert strict["accuracy"] == 0.0
    assert strict["format_compliance"] == 0.0
    # The gap between them is exactly what the report warns about.
    assert lenient["accuracy"] > strict["accuracy"]


def test_generation_failure_is_recorded_not_raised(examples):
    r = run_eval(RaisingGenerator(), examples, run_name="stub_raising", batch_size=16)
    assert r["generation_errors"], "a failing generator must be recorded"
    assert r["accuracy"] == 0.0
    assert "simulated inference failure" in r["generation_errors"][0]["error"]


def test_wrong_output_count_is_rejected(examples):
    """Silent misalignment would score nonsense that looks like a bad model."""
    with pytest.raises(ValueError, match="one output per prompt"):
        run_eval(WrongLengthGenerator(), examples, run_name="stub_wrong_len")


def test_empty_example_set_is_rejected():
    with pytest.raises(ValueError, match="no examples"):
        run_eval(PerfectGenerator(), [], run_name="stub_empty")


def test_malformed_examples_are_rejected():
    with pytest.raises(ValueError, match="lacks"):
        run_eval(PerfectGenerator(), [{"question": "x"}], run_name="stub_bad")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_report_is_written_and_reloadable(tmp_path, examples):
    r = run_eval(
        PerfectGenerator(), examples, report_dir=tmp_path, run_name="stub_report",
        model_description="PerfectGenerator (stub)",
    )
    md = tmp_path / "stub_report.md"
    js = tmp_path / "stub_report.json"
    assert md.exists() and js.exists()

    reloaded = json.loads(js.read_text())
    assert reloaded["accuracy"] == r["accuracy"]

    text = md.read_text()
    assert "100.00%" in text
    assert "PerfectGenerator (stub)" in text
    assert "Scope" in text  # the limits section must survive into the report


def test_report_warns_when_the_fallback_carried_the_score(tmp_path, examples):
    run_eval(
        UnformattedGenerator(), examples, report_dir=tmp_path, run_name="stub_fb"
    )
    text = (tmp_path / "stub_fb.md").read_text()
    assert "last-number fallback" in text
    assert "upper bound" in text


def test_accuracy_by_steps_covers_every_example(examples):
    r = run_eval(PerfectGenerator(), examples, run_name="stub_steps")
    total = sum(m["n"] for m in r["accuracy_by_steps"].values())
    assert total == len(examples)


def test_prompt_template_is_recorded(examples):
    """Two reports are only comparable if the template matches; record it."""
    r = run_eval(PerfectGenerator(), examples, run_name="stub_tmpl")
    assert r["prompt_template"] == DEFAULT_PROMPT
