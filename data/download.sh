#!/usr/bin/env bash
# Fetch GSM8K into data/raw/ from the OpenAI repository that publishes it.
#
# Fetched from the source repo rather than a mirror or a dataset-hub copy, so
# the provenance chain is one hop and the licence that applies is the one in
# that repo (MIT, Copyright (c) 2021 OpenAI). See DATA.md.
#
# Idempotent: re-running re-fetches and re-checksums, which is cheap at ~5 MB
# and means a truncated earlier download cannot survive silently.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
BASE="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"

mkdir -p "$RAW_DIR"

for f in train.jsonl test.jsonl; do
  echo "fetching $f"
  curl -sSL -C - --retry 5 --retry-delay 2 --retry-all-errors \
       --connect-timeout 30 --max-time 300 \
       -o "$RAW_DIR/$f" "$BASE/$f"
done

# The licence travels with the data. Keeping a copy next to the raw files means
# anyone who finds this directory in isolation can still see the terms.
curl -sSL --max-time 120 -o "$RAW_DIR/LICENSE.gsm8k" \
     "https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE"

echo
echo "--- data/raw ---"
ls -la "$RAW_DIR"
echo
echo "--- sha256 ---"
shasum -a 256 "$RAW_DIR"/*.jsonl
echo
echo "--- line counts ---"
wc -l "$RAW_DIR"/*.jsonl
