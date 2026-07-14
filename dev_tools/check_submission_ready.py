import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RISK_SK_PREFIX = "sk" + "-"
RISK_AUTH = "Authori" + "zation"
RISK_BEARER = "Bear" + "er"
RISK_HOME = "/" + "home" + "/" + "ubuntu"
RISK_MAC_USERS = "/" + "Users" + "/"
RISK_WINDOWS_USERS = "C:" + chr(92) + "Users" + chr(92)
RISK_LOAD_DOTENV = "load_" + "dotenv"
RISK_CALL_INTERN = "call_" + "intern_s1"
RISK_ENV_OPEN = "open(" + "\".env\""
RISK_ENV_OPEN_SINGLE = "open(" + "'.env'"
RISK_META_ANSWER_PATTERNS = (
    "metadata" + '["answer"]',
    "metadata" + "['answer']",
    "metadata.get(" + '"answer"',
    "metadata.get(" + "'answer'",
)
ONLINE_MODEL_IMPORTS = {
    "anthropic",
    "cohere",
    "deep" + "seek",
    "google.generativeai",
    "httpx",
    "openai",
    "requests",
}
REQUIRED_METADATA_DENYLIST = {
    "answer",
    "expected_answer",
    "gold_answer",
    "reference_answer",
    "solution",
    "gold",
    "reference",
    "ground_truth",
    "expected",
    "expected_solution",
    "official_answer",
    "label",
    "target",
}
IMPORT_TO_DISTRIBUTION = {"dotenv": "python-dotenv"}


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return "推理过程...\n最终答案：2"


def _result() -> dict:
    return {"warnings": [], "failures": []}


def _add(items: list[dict], path: Path | str, code: str, message: str) -> None:
    try:
        display_path = str(Path(path).resolve().relative_to(ROOT))
    except (TypeError, ValueError):
        display_path = str(path)
    items.append({"path": display_path, "code": code, "message": message})


def check_response_schema(response: Any, metadata: dict | None = None) -> dict:
    result = _result()
    metadata = metadata if isinstance(metadata, dict) else {}
    if not isinstance(response, dict):
        _add(result["failures"], "response", "response_not_dict", "solve() must return a dict")
        return result

    final_response = response.get("final_response")
    if not isinstance(final_response, str) or not final_response.strip():
        _add(result["failures"], "response", "missing_final_response", "final_response must be a nonempty string")
    if metadata.get("answer") is not None and final_response == metadata.get("answer"):
        _add(result["failures"], "response", "metadata_answer_used", "final_response must not directly use metadata answer")

    trace = response.get("trace")
    if trace is not None and not isinstance(trace, list):
        _add(result["failures"], "response", "trace_not_list", "trace must be a list when present")

    try:
        json.dumps(response, ensure_ascii=False)
    except TypeError:
        _add(result["failures"], "response", "response_not_json_serializable", "solve() response must be JSON serializable")
    return result


def check_user_agent_entrypoint() -> dict:
    result = _result()
    user_agent_path = ROOT / "user_agent.py"
    if not user_agent_path.exists():
        _add(result["failures"], user_agent_path, "missing_user_agent", "repository root must contain user_agent.py")
        return result

    try:
        from user_agent import ReasoningAgent
    except Exception as exc:
        _add(result["failures"], user_agent_path, "import_failed", f"ReasoningAgent import failed: {type(exc).__name__}")
        return result

    metadata = {"idx": 0, "answer": "999"}
    try:
        client = FakeClient()
        agent = ReasoningAgent(client=client)
        response = agent.solve("1+1=?", metadata)
        if not client.calls:
            _add(result["failures"], user_agent_path, "client_not_called", "ReasoningAgent must call client.chat")
    except Exception as exc:
        _add(result["failures"], user_agent_path, "solve_failed", f"solve() raised {type(exc).__name__}")
        return result

    schema = check_response_schema(response, metadata)
    result["warnings"].extend(schema["warnings"])
    result["failures"].extend(schema["failures"])
    return result


def _module_scope_imports(nodes: list[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    imports = []
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.If):
            imports.extend(_module_scope_imports(node.body))
            imports.extend(_module_scope_imports(node.orelse))
        elif isinstance(node, ast.Try):
            imports.extend(_module_scope_imports(node.body))
            imports.extend(_module_scope_imports(node.orelse))
            imports.extend(_module_scope_imports(node.finalbody))
            for handler in node.handlers:
                imports.extend(_module_scope_imports(handler.body))
    return imports


def _local_module_path(module_name: str) -> Path | None:
    if not module_name:
        return None
    module_path = ROOT.joinpath(*module_name.split("."))
    file_candidate = module_path.with_suffix(".py")
    if file_candidate.exists():
        return file_candidate
    package_candidate = module_path / "__init__.py"
    return package_candidate if package_candidate.exists() else None


def inspect_official_import_chain() -> tuple[list[Path], set[str]]:
    pending = [ROOT / "user_agent.py"]
    visited: set[Path] = set()
    third_party: set[str] = set()
    while pending:
        path = pending.pop()
        if path in visited or not path.exists():
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _module_scope_imports(tree.body):
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            else:
                module_names = [node.module or ""]
            for module_name in module_names:
                local_path = _local_module_path(module_name)
                if local_path is not None:
                    pending.append(local_path)
                    continue
                top_level = module_name.split(".", 1)[0]
                if top_level and top_level not in sys.stdlib_module_names:
                    third_party.add(module_name)
    return sorted(visited), third_party


def check_official_import_chain() -> dict:
    result = _result()
    try:
        paths, third_party = inspect_official_import_chain()
    except (OSError, SyntaxError) as exc:
        _add(result["failures"], ROOT / "user_agent.py", "import_chain_scan_failed", type(exc).__name__)
        return result
    for module_name in sorted(third_party):
        if module_name in ONLINE_MODEL_IMPORTS or module_name.split(".", 1)[0] in ONLINE_MODEL_IMPORTS:
            _add(
                result["failures"],
                ROOT / "user_agent.py",
                "external_model_import",
                f"official import chain must not load external client module: {module_name}",
            )
    for path in paths:
        if ("deep" + "seek") in path.read_text(encoding="utf-8").lower():
            _add(result["failures"], path, "external_provider_reference", "official import chain references a forbidden provider")
    scan = scan_sensitive_patterns(paths)
    result["warnings"].extend(scan["warnings"])
    result["failures"].extend(scan["failures"])
    return result


def _submission_source_paths() -> list[Path]:
    paths = [
        ROOT / "user_agent.py",
        ROOT / "runner.py",
        ROOT / "intern_s1_client.py",
        ROOT / "config.py",
        ROOT / "README.md",
        ROOT / "SUBMISSION.md",
        ROOT / ".env.example",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
    ]
    for directory in (ROOT / "agents", ROOT / "dev_tools", ROOT / "scripts"):
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return [path for path in paths if path.exists() and "__pycache__" not in path.parts]


def check_repository_source_safety() -> dict:
    result = _result()
    credential_pattern = re.compile(r"sk-[A-Za-z0-9_-]{12,}")
    personal_path_markers = (RISK_HOME + "/", RISK_MAC_USERS, RISK_WINDOWS_USERS)
    for path in _submission_source_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if credential_pattern.search(text):
            _add(result["failures"], path, "hardcoded_api_key", "possible hardcoded API key")
        if any(marker in text for marker in personal_path_markers):
            _add(result["failures"], path, "absolute_personal_path", "absolute personal path appears in source")
    return result


def check_metadata_safety() -> dict:
    result = _result()
    try:
        from user_agent import METADATA_DENYLIST, ReasoningAgent, _safe_metadata
    except Exception as exc:
        _add(result["failures"], ROOT / "user_agent.py", "metadata_check_import_failed", type(exc).__name__)
        return result

    missing = REQUIRED_METADATA_DENYLIST - set(METADATA_DENYLIST)
    if missing:
        _add(
            result["failures"],
            ROOT / "user_agent.py",
            "metadata_denylist_incomplete",
            f"missing normalized metadata keys: {sorted(missing)}",
        )

    canaries = {
        "answer": "blocked_answer_value",
        "Answer": "blocked_case_value",
        "expected-answer": "blocked_expected_value",
        "Ground Truth": "blocked_ground_value",
        "gold": "blocked_gold_value",
        "reference": "blocked_reference_value",
        "official_answer": "blocked_official_value",
        "label": "blocked_label_value",
        "target": "blocked_target_value",
    }
    safe = _safe_metadata({**canaries, "idx": "kept_idx", "subject": "kept_subject"})
    blocked_values = set(canaries.values())
    if blocked_values & set(safe.values()):
        _add(result["failures"], ROOT / "user_agent.py", "metadata_filter_failed", "denylisted metadata survived filtering")

    client = FakeClient()
    response = ReasoningAgent(client=client).solve("1+1=?", canaries)
    serialized_response = json.dumps(response, ensure_ascii=False)
    serialized_prompts = json.dumps(client.calls, ensure_ascii=False)
    for value in blocked_values:
        if value in serialized_response or value in serialized_prompts:
            _add(result["failures"], ROOT / "user_agent.py", "metadata_value_leaked", "blocked metadata entered prompt or trace")
            break
    return result


def _literal_set_assignment(path: Path, variable_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            value = ast.literal_eval(node.value)
            return {str(item) for item in value}
    return set()


def check_runner_metadata_allowlists() -> dict:
    result = _result()
    for path in (ROOT / "dev_tools" / "run_user_agent_local.py", ROOT / "dev_tools" / "run_user_agent_real_smoke.py"):
        try:
            allowlist = _literal_set_assignment(path, "METADATA_ALLOWLIST")
        except (OSError, SyntaxError, ValueError) as exc:
            _add(result["failures"], path, "metadata_allowlist_scan_failed", type(exc).__name__)
            continue
        normalized = {re.sub(r"[\s-]+", "_", key.strip().lower()) for key in allowlist}
        overlap = normalized & REQUIRED_METADATA_DENYLIST
        if overlap:
            _add(result["failures"], path, "unsafe_metadata_allowlist", f"answer fields allowed: {sorted(overlap)}")
    return result


def _default_scan_paths(root: Path = ROOT) -> list[Path]:
    paths = [root / "user_agent.py", root / "runner.py"]
    paths.extend(sorted((root / "agents").glob("*.py")))
    paths.extend(sorted((root / "dev_tools").glob("*.py")))
    return [path for path in paths if path.exists() and path.name != ".env"]


def scan_sensitive_patterns(paths: list[str | Path]) -> dict:
    result = _result()
    for raw_path in paths:
        path = Path(raw_path)
        if path.name == ".env":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        if RISK_SK_PREFIX in text:
            _add(result["failures"], path, "hardcoded_api_key", "possible hardcoded API key prefix")
        if RISK_AUTH in text:
            _add(result["warnings"], path, "authorization_marker", "auth marker appears in scanned file")
        if RISK_BEARER in text:
            _add(result["warnings"], path, "bearer_marker", "bearer marker appears in scanned file")
        if RISK_HOME in text:
            _add(result["warnings"], path, "absolute_home_path", "local absolute path appears in scanned file")

        if path.name == "user_agent.py":
            if RISK_LOAD_DOTENV in text:
                _add(result["failures"], path, "load_dotenv_in_user_agent", "user_agent.py must not load .env")
            if RISK_CALL_INTERN in text:
                _add(result["failures"], path, "call_intern_s1_in_user_agent", "user_agent.py must not call local Intern-S1 client")
            if RISK_ENV_OPEN in text or RISK_ENV_OPEN_SINGLE in text:
                _add(result["failures"], path, "open_env_in_user_agent", "user_agent.py must not open .env")
            for pattern in RISK_META_ANSWER_PATTERNS:
                if pattern in text:
                    _add(result["failures"], path, "metadata_answer_used", "user_agent.py must not directly read metadata answer")
                    break
    return result


def check_requirements() -> dict:
    result = _result()
    path = ROOT / "requirements.txt"
    if not path.exists():
        _add(result["failures"], path, "missing_requirements", "requirements.txt must exist")
        return result

    _paths, third_party = inspect_official_import_chain()
    requirement_names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
        if match:
            requirement_names.add(match.group(1).lower().replace("_", "-"))
    required_distributions = {
        IMPORT_TO_DISTRIBUTION.get(module.split(".", 1)[0], module.split(".", 1)[0]).lower().replace("_", "-")
        for module in third_party
    }
    missing = required_distributions - requirement_names
    if missing:
        _add(result["failures"], path, "missing_runtime_requirement", f"missing runtime dependencies: {sorted(missing)}")
    if not required_distributions and requirement_names:
        _add(result["failures"], path, "unexpected_runtime_requirement", "official runtime is standard-library-only")
    return result


def run_checks() -> dict:
    summary = _result()
    for check in (
        check_user_agent_entrypoint(),
        check_official_import_chain(),
        check_repository_source_safety(),
        check_metadata_safety(),
        check_runner_metadata_allowlists(),
        check_requirements(),
    ):
        summary["warnings"].extend(check["warnings"])
        summary["failures"].extend(check["failures"])
    summary["passed"] = not summary["failures"]
    return summary


def main() -> None:
    summary = run_checks()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
