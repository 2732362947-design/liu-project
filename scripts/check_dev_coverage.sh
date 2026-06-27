#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 dev_tools/check_domain_coverage.py \
  --questions data/dev_questions.json \
  --output outputs/domain_coverage.md
