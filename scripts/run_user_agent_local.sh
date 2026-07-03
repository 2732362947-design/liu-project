#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 dev_tools/run_user_agent_local.py \
  --input data/user_agent_smoke.jsonl \
  --output-dir outputs_user_agent
