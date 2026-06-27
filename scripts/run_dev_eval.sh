#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 runner.py \
  --input data/dev_questions.json \
  --output outputs/dev_results.json \
  --sleep 8 \
  --attempts 1

python3 dev_tools/run_dev_review.py \
  --results outputs/dev_results.json \
  --questions data/dev_questions.json \
  --output outputs/dev_review_summary.json

cat outputs/dev_review_summary.json
