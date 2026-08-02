"""Verbatim-regurgitation detector: n-gram overlap between outputs and training data.

The question this answers is narrow and worth stating exactly, because the
loose version ("did the model memorise the training set?") is not something
n-gram overlap can answer:

    Does a generated output contain spans of text that appear verbatim in the
    fine-tuning corpus, and how long are the longest such spans?

That is a detector for *copying*, not for memorisation in general. What it
catches and what it does not:

  CATCHES   a model reproducing a training solution word-for-word; a model that
            has collapsed onto emitting template text from its training data;
            an eval item that leaked into train (the output will match a
            training document at high overlap because it *is* one).

  MISSES    a paraphrase; the same reasoning with different wording; a
            memorised *answer* with freshly generated reasoning; memorisation
            of facts rather than strings. A low score here is evidence about
            verbatim copying and nothing more, and reporting it as "the model
            did not memorise" would be an overclaim.

  FALSE-    on this corpus especially. GSM8K solutions are short, formulaic
  POSITIVES arithmetic sentences, so phrases like "how many did she have left"
            recur across unrelated problems. Any correct answer will share some
            n-grams with training data. This is why the primary metric is the
            LONGEST matching span rather than the fraction of matching n-grams:
            incidental overlap produces many short matches, copying produces
            one long one, and the two look completely different under a
            longest-span metric while looking similar under a coverage metric.

Two metrics are reported per candidate:

    longest_match_tokens  — length, in tokens, of the longest contiguous span
                            that appears somewhere in the training corpus.
                            This is the one to look at.
    ngram_coverage        — fraction of the candidate's n-grams that appear in
                            training. Context, not a verdict.

The flag threshold is on longest_match_tokens and defaults to 25, which is
long enough that an entire GSM8K solution sentence plus its neighbour has been
reproduced. It is a chosen number, not a derived one — `calibrate()` prints the
distribution over real training documents so the choice can be checked against
this corpus rather than taken on faith.

Usage:
    .venv/bin/python -m eval.memorisation_check --calibrate
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"

DEFAULT_N = 8
DEFAULT_FLAG_TOKENS = 25

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenise(text: str) -> list[str]:
    """Lowercased word/punctuation tokens.

    Case-insensitive because a model reproducing a training sentence with
    different capitalisation has still reproduced it. Punctuation is kept as
    separate tokens so that span lengths are not inflated by attaching
    punctuation to words.
    """
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class MemorisationIndex:
    """An n-gram index over the training corpus.

    Built once and queried per candidate. Memory is the obvious cost: the GSM8K
    train split is roughly 400k tokens, so the 8-gram set is a few hundred
    thousand tuples. That is fine here and would not be at corpus scale, where
    a Bloom filter or suffix automaton would replace this.
    """

    def __init__(self, documents: Iterable[str], n: int = DEFAULT_N):
        self.n = n
        self._grams: set[tuple[str, ...]] = set()
        self.n_documents = 0
        self.n_tokens = 0
        for doc in documents:
            toks = tokenise(doc)
            self.n_documents += 1
            self.n_tokens += len(toks)
            self._grams.update(ngrams(toks, n))

    def __len__(self) -> int:
        return len(self._grams)

    def contains(self, gram: tuple[str, ...]) -> bool:
        return gram in self._grams

    def longest_match(self, tokens: Sequence[str]) -> int:
        """Length in tokens of the longest span present in the corpus.

        Works by walking the candidate's n-grams and extending a run while
        consecutive n-grams are all present. A run of `r` consecutive matching
        n-grams corresponds to a contiguous span of `r + n - 1` tokens.

        This is an approximation, and the direction of the error matters: two
        adjacent n-grams that each appear in the corpus but in *different*
        documents will be counted as one longer span. So the metric can
        overstate, never understate — which is the right direction for a
        detector whose job is to raise a flag for a human to look at.
        """
        grams = ngrams(tokens, self.n)
        if not grams:
            # Too short to contain even one n-gram; report the token count if
            # the whole thing happens to appear, else 0.
            return 0

        best = 0
        run = 0
        for g in grams:
            if self.contains(g):
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best + self.n - 1 if best else 0

    def coverage(self, tokens: Sequence[str]) -> float:
        grams = ngrams(tokens, self.n)
        if not grams:
            return 0.0
        return sum(self.contains(g) for g in grams) / len(grams)


def check_outputs(
    candidates: Sequence[str],
    index: MemorisationIndex,
    flag_tokens: int = DEFAULT_FLAG_TOKENS,
) -> dict:
    """Score candidate generations for verbatim overlap with the corpus."""
    per_item = []
    for i, text in enumerate(candidates):
        toks = tokenise(text)
        longest = index.longest_match(toks)
        per_item.append(
            {
                "index": i,
                "n_tokens": len(toks),
                "longest_match_tokens": longest,
                "ngram_coverage": round(index.coverage(toks), 4),
                "flagged": longest >= flag_tokens,
                "excerpt": text[:200],
            }
        )

    flagged = [r for r in per_item if r["flagged"]]
    longests = [r["longest_match_tokens"] for r in per_item]
    covs = [r["ngram_coverage"] for r in per_item]

    return {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_candidates": len(candidates),
        "n_gram_size": index.n,
        "flag_threshold_tokens": flag_tokens,
        "corpus": {
            "documents": index.n_documents,
            "tokens": index.n_tokens,
            "distinct_ngrams": len(index),
        },
        "n_flagged": len(flagged),
        "flagged_rate": len(flagged) / len(candidates) if candidates else 0.0,
        "longest_match_tokens": {
            "max": max(longests) if longests else 0,
            "mean": round(sum(longests) / len(longests), 2) if longests else 0.0,
            "median": sorted(longests)[len(longests) // 2] if longests else 0,
        },
        "ngram_coverage": {
            "max": max(covs) if covs else 0.0,
            "mean": round(sum(covs) / len(covs), 4) if covs else 0.0,
        },
        "flagged_items": flagged,
        "per_item": per_item,
    }


def format_report(r: dict) -> str:
    lm = r["longest_match_tokens"]
    lines = [
        "# Memorisation check (verbatim n-gram overlap)",
        "",
        f"- checked: {r['checked_at']}",
        f"- candidates: {r['n_candidates']}",
        f"- n-gram size: {r['n_gram_size']}",
        f"- flag threshold: longest verbatim span >= {r['flag_threshold_tokens']} tokens",
        f"- corpus: {r['corpus']['documents']} documents, "
        f"{r['corpus']['tokens']} tokens, {r['corpus']['distinct_ngrams']} distinct n-grams",
        "",
        "## Result",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| flagged | **{r['n_flagged']} / {r['n_candidates']}** ({r['flagged_rate'] * 100:.1f}%) |",
        f"| longest verbatim span, max | {lm['max']} tokens |",
        f"| longest verbatim span, median | {lm['median']} tokens |",
        f"| longest verbatim span, mean | {lm['mean']} tokens |",
        f"| n-gram coverage, mean | {r['ngram_coverage']['mean']:.4f} |",
        "",
    ]

    if r["flagged_items"]:
        lines += [
            "## Flagged",
            "",
            "| # | longest span | coverage | excerpt |",
            "|---:|---:|---:|---|",
        ]
        for f in r["flagged_items"][:20]:
            excerpt = f["excerpt"].replace("\n", " ").replace("|", "\\|")[:110]
            lines.append(
                f"| {f['index']} | {f['longest_match_tokens']} | "
                f"{f['ngram_coverage']:.3f} | {excerpt}… |"
            )
        lines += [
            "",
            "A flag is a prompt to look, not a verdict. Read the flagged output",
            "against the training corpus before concluding anything.",
            "",
        ]
    else:
        lines += [
            "No candidate reproduced a span of "
            f"{r['flag_threshold_tokens']} or more tokens from the training corpus.",
            "",
            "This is evidence about **verbatim copying only**. It is not evidence",
            "that the model did not memorise: a paraphrase, the same reasoning in",
            "different words, or a memorised answer with fresh reasoning would all",
            "pass this check. Do not report it as an absence of memorisation.",
            "",
        ]

    lines += [
        "## Reading the numbers",
        "",
        "GSM8K solutions are short formulaic arithmetic sentences, so unrelated",
        "correct answers share n-grams routinely. Non-zero coverage is the",
        "expected state and means nothing on its own. The longest-span metric is",
        "the one that separates copying from incidental overlap: incidental",
        "overlap produces many short matches, copying produces one long one.",
        "",
        "The span metric can overstate. Adjacent n-grams that each occur in the",
        "corpus but in *different* documents are counted as one longer span. The",
        "error is one-directional by design — a detector should over-flag rather",
        "than under-flag — so a flagged item may on inspection be nothing.",
        "",
    ]
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL, splitting on newlines only — never `.splitlines()`.

    See eval/harness.py:load_jsonl for why. GSM8K contains a U+2028 inside a
    question, and `.splitlines()` breaks on it while JSON does not escape it.
    """
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").split("\n")
        if l.strip()
    ]


def build_index_from_training(n: int = DEFAULT_N, field: str = "solution") -> MemorisationIndex:
    rows = load_jsonl(PROCESSED / "train.jsonl")
    return MemorisationIndex((r[field] for r in rows), n=n)


def calibrate(n: int = DEFAULT_N, sample: int = 300) -> dict:
    """Print the metric's distribution on real data, so the threshold is checkable.

    Two populations are compared:

      held-out eval solutions — genuinely not in the training corpus. Whatever
        overlap they show is the incidental-overlap floor for this task.
      training solutions themselves — the ceiling. A document that IS in the
        corpus should match itself completely, so this shows what a true
        positive looks like.

    A threshold that does not sit clearly between those two distributions is
    not a useful threshold, and this is how you find that out before trusting it.
    """
    index = build_index_from_training(n=n)

    eval_rows = load_jsonl(PROCESSED / "eval.jsonl")
    train_rows = load_jsonl(PROCESSED / "train.jsonl")

    held_out = [r["solution"] for r in eval_rows[:sample]]
    in_corpus = [r["solution"] for r in train_rows[:sample]]

    def dist(texts):
        vals = sorted(index.longest_match(tokenise(t)) for t in texts)
        return {
            "n": len(vals),
            "min": vals[0],
            "p50": vals[len(vals) // 2],
            "p90": vals[int(len(vals) * 0.9)],
            "p99": vals[int(len(vals) * 0.99)],
            "max": vals[-1],
        }

    ho_vals = [index.longest_match(tokenise(t)) for t in held_out]
    ic_vals = [index.longest_match(tokenise(t)) for t in in_corpus]

    # The error rates the threshold actually produces on this corpus. A
    # threshold that separates the medians can still miss badly in the tails,
    # and the tails are where a short memorised solution would live.
    false_positives = sum(v >= DEFAULT_FLAG_TOKENS for v in ho_vals)
    false_negatives = sum(v < DEFAULT_FLAG_TOKENS for v in ic_vals)

    out = {
        "n_gram_size": n,
        "corpus_documents": index.n_documents,
        "corpus_distinct_ngrams": len(index),
        "held_out_eval_solutions": dist(held_out),
        "training_solutions_in_corpus": dist(in_corpus),
        "default_flag_threshold": DEFAULT_FLAG_TOKENS,
        "at_default_threshold": {
            "false_positive_rate": round(false_positives / len(ho_vals), 4),
            "false_positives": false_positives,
            "false_negative_rate": round(false_negatives / len(ic_vals), 4),
            "false_negatives": false_negatives,
            "note": (
                "False negatives are documents that ARE in the corpus but whose "
                "longest verbatim span falls under the threshold — short solutions. "
                "A model reproducing one of those verbatim would not be flagged."
            ),
        },
    }

    print(json.dumps(out, indent=2))
    print()
    ho = out["held_out_eval_solutions"]
    ic = out["training_solutions_in_corpus"]
    print(f"held-out (true negatives)  p99 = {ho['p99']:>4} tokens, max = {ho['max']} tokens")
    print(f"in-corpus (true positives) p50 = {ic['p50']:>4} tokens, min = {ic['min']} tokens")
    print(f"threshold {DEFAULT_FLAG_TOKENS} sits "
          f"{'BETWEEN the two — usable' if ho['p99'] < DEFAULT_FLAG_TOKENS <= ic['p50'] else 'OUTSIDE the gap — review it'}")
    at = out["at_default_threshold"]
    print(f"  false positives (held-out flagged) : "
          f"{at['false_positives']}/{ho['n']}  ({at['false_positive_rate'] * 100:.1f}%)")
    print(f"  false negatives (in-corpus missed) : "
          f"{at['false_negatives']}/{ic['n']}  ({at['false_negative_rate'] * 100:.1f}%)")
    print("  the false-negative rate is the honest limit: short training "
          "solutions reproduced verbatim would not be flagged.")
    return out


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        calibrate()
    else:
        print(__doc__)
        print("Run with --calibrate to see the metric's distribution on this corpus.")
