#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 runner.py \
  --input data/official_questions.json \
  --output outputs/official_results.json \
  --sleep 8 \
  --attempts 1

python3 dev_tools/run_dev_review.py \
  --results outputs/official_results.json \
  --questions data/official_questions.json \
  --output outputs/official_review_summary.json

python3 dev_tools/extract_failed_questions.py \
  --results outputs/official_results.json \
  --questions data/official_questions.json \
  --output outputs/official_failed_questions.json

cat outputs/official_review_summary.json
