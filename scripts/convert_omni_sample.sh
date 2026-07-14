#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python3 dev_tools/convert_omni_math.py \
  --input "${HOME}/.cache/modelscope/hub/datasets/AI-ModelScope/Omni-MATH/test.jsonl" \
  --output data/omni_math_sample.json \
  --max-per-domain 3 \
  --max-total 90

python3 dev_tools/check_domain_coverage.py \
  --questions data/omni_math_sample.json \
  --output outputs/omni_domain_coverage.md
