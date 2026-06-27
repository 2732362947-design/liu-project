import json
import time

import requests

import config


INTERN_S1_API_KEY = config.INTERN_S1_API_KEY
INTERN_S1_BASE_URL = config.INTERN_S1_BASE_URL
INTERN_S1_MODEL = config.INTERN_S1_MODEL
INTERN_S1_CONNECT_TIMEOUT_SECONDS = getattr(config, "INTERN_S1_CONNECT_TIMEOUT_SECONDS", 10)
INTERN_S1_READ_TIMEOUT_SECONDS = getattr(config, "INTERN_S1_READ_TIMEOUT_SECONDS", 60)
INTERN_S1_MAX_RETRIES = getattr(config, "INTERN_S1_MAX_RETRIES", 0)

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
NO_RETRY_STATUS_CODES = {401, 403}
BACKOFF_SECONDS = (3, 8)
DNS_ERROR_KEYWORDS = (
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "getaddrinfo failed",
)
TIMEOUT_KEYWORDS = (
    "read timed out",
    "connecttimeout",
    "readtimeout",
    "timeout",
)
CONNECTION_ERROR_KEYWORDS = (
    "connectionerror",
    "newconnectionerror",
    "failed to establish a new connection",
    "max retries exceeded",
)


def _safe_error_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if INTERN_S1_API_KEY:
        message = message.replace(INTERN_S1_API_KEY, "[redacted]")
    return message.replace("Authorization", "[redacted-header]")


def _safe_response_error(response: requests.Response) -> str:
    text = response.text[:300] if response.text else ""
    if INTERN_S1_API_KEY:
        text = text.replace(INTERN_S1_API_KEY, "[redacted]")
    text = text.replace("Authorization", "[redacted-header]")
    return f"HTTP {response.status_code}: {text}"


def _http_status_from_error(exc: Exception | str) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    text = str(exc).lower()
    for status in (401, 403):
        if f"http {status}" in text or f"status {status}" in text:
            return status
    for token in ("http ", "status "):
        marker = text.find(token)
        if marker == -1:
            continue
        rest = text[marker + len(token) :].lstrip(":=")
        digits = "".join(char for char in rest[:3] if char.isdigit())
        if len(digits) == 3:
            return int(digits)
    return None


def classify_http_status(status_code: int) -> str:
    if status_code == 401:
        return "auth_error"
    if status_code == 403:
        return "forbidden"
    if 400 <= status_code <= 599:
        return "http_error"
    return "unknown_error"


def classify_error(exc: Exception | str) -> str:
    text = str(exc).lower()
    if isinstance(exc, requests.exceptions.Timeout) or any(
        keyword in text for keyword in TIMEOUT_KEYWORDS
    ):
        return "timeout"
    if any(keyword in text for keyword in DNS_ERROR_KEYWORDS):
        return "dns_error"

    status_code = _http_status_from_error(exc)
    if status_code is not None:
        status_type = classify_http_status(status_code)
        if status_type != "unknown_error":
            return status_type

    if isinstance(exc, requests.exceptions.ConnectionError) or any(
        keyword in text for keyword in CONNECTION_ERROR_KEYWORDS
    ):
        return "connection_error"
    return "unknown_error"


def classify_intern_s1_error(error_text: str) -> str:
    return classify_error(error_text)


def call_intern_s1(prompt: str) -> str:
    if not INTERN_S1_API_KEY or not INTERN_S1_BASE_URL or not INTERN_S1_MODEL:
        return "mock_response: Intern-S1 API is not configured."

    url = f"{INTERN_S1_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {INTERN_S1_API_KEY}",
    }
    data = {
        "model": INTERN_S1_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    timeout = (INTERN_S1_CONNECT_TIMEOUT_SECONDS, INTERN_S1_READ_TIMEOUT_SECONDS)
    max_retries = max(0, INTERN_S1_MAX_RETRIES)
    max_attempts = max_retries + 1
    last_error = ""
    session = requests.Session()
    session.trust_env = False

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.post(
                url,
                headers=headers,
                data=json.dumps(data),
                timeout=timeout,
            )
            if response.status_code == 401:
                return "[intern-s1 error] auth_error: unauthorized"
            if response.status_code == 403:
                return "[intern-s1 error] forbidden: forbidden"
            if response.status_code in RETRY_STATUS_CODES:
                last_error = f"http_error: {_safe_response_error(response)}"
                if attempt <= max_retries:
                    time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
                    continue
                return f"[intern-s1 error] {last_error}"
            if response.status_code >= 400:
                return f"[intern-s1 error] http_error: {_safe_response_error(response)}"

            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout as exc:
            last_error = f"timeout: {_safe_error_message(exc)}"
        except requests.exceptions.ConnectionError as exc:
            last_error = f"connection_error: {_safe_error_message(exc)}"
        except Exception as exc:
            error_type = classify_error(exc)
            return f"[intern-s1 error] {error_type}: {_safe_error_message(exc)}"

        if attempt >= max_attempts:
            return f"[intern-s1 error] {last_error}"
        time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])

    return f"[intern-s1 error] {last_error or 'unknown request failure'}"
