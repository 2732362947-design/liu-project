#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TEMP_ROOT}"' EXIT
export PIP_CACHE_DIR="${TEMP_ROOT}/pip-cache"

WORKTREE="${TEMP_ROOT}/project"
VENV="${TEMP_ROOT}/venv"
mkdir -p "${WORKTREE}"

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.env' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='outputs/*' \
  --exclude='outputs_user_agent*' \
  --exclude='logs/*' \
  -C "${ROOT}" -cf - . | tar -C "${WORKTREE}" -xf -

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -r "${WORKTREE}/requirements.txt"

cd "${WORKTREE}"
"${VENV}/bin/python" - <<'PY'
import json

from user_agent import ReasoningAgent


class FakeClient:
    def chat(self, **kwargs):
        return "计算过程：1+1=2。\n最终答案：2"


result = ReasoningAgent(client=FakeClient()).solve("1+1=?", {"answer": "blocked"})
assert isinstance(result, dict)
assert isinstance(result.get("final_response"), str)
assert result["final_response"].strip()
json.dumps(result, ensure_ascii=False)
print("clean environment fake-client smoke: passed")
PY
