#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 dev_tools/summarize_run.py \
  --results outputs/dev_results.json \
  --review outputs/dev_review_summary.json \
  --output outputs/dev_report.md
