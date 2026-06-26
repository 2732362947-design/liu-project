import requests
import time

from config import (
    INTERN_S1_API_KEY,
    INTERN_S1_BASE_URL,
    INTERN_S1_CONNECT_TIMEOUT_SECONDS,
    INTERN_S1_MAX_RETRIES,
    INTERN_S1_MODEL,
    INTERN_S1_READ_TIMEOUT_SECONDS,
)


RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
NO_RETRY_STATUS_CODES = {401, 403}
BACKOFF_SECONDS = (3, 8)
DNS_ERROR_KEYWORDS = (
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "getaddrinfo failed",
)


def _safe_error_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if INTERN_S1_API_KEY:
        message = message.replace(INTERN_S1_API_KEY, "[redacted]")
    return message


def _safe_response_error(response: requests.Response) -> str:
    text = response.text[:300] if response.text else ""
    if INTERN_S1_API_KEY:
        text = text.replace(INTERN_S1_API_KEY, "[redacted]")
    return f"HTTP {response.status_code}: {text}"


def classify_intern_s1_error(error_text: str) -> str:
    text = error_text.lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(keyword in text for keyword in DNS_ERROR_KEYWORDS):
        return "dns_error"
    if "http 401" in text or "status 401" in text or "auth_error" in text:
        return "auth_error"
    if "http 403" in text or "status 403" in text or "forbidden" in text:
        return "forbidden"
    if "http " in text or "status " in text or "http_error" in text:
        return "http_error"
    if (
        "connection_error" in text
        or "connectionerror" in text
        or "failed to establish a new connection" in text
    ):
        return "connection_error"
    return "unknown_error"


def call_intern_s1(prompt: str) -> str:
    if not INTERN_S1_API_KEY or INTERN_S1_API_KEY.startswith("your_"):
        return (
            "[mock intern-s1] No Intern-S1 credentials are configured. "
            f"Prompt length: {len(prompt)}."
        )

    url = f"{INTERN_S1_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {INTERN_S1_API_KEY}",
    }
    data = {
        "model": INTERN_S1_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    session = requests.Session()
    session.trust_env = False

    last_error = ""
    max_retries = max(0, INTERN_S1_MAX_RETRIES)
    max_attempts = max_retries + 1
    timeout = (INTERN_S1_CONNECT_TIMEOUT_SECONDS, INTERN_S1_READ_TIMEOUT_SECONDS)
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.post(url, headers=headers, json=data, timeout=timeout)
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
        except requests.exceptions.ProxyError as exc:
            return f"[intern-s1 error] proxy_error: {_safe_error_message(exc)}"
        except requests.exceptions.Timeout as exc:
            last_error = f"timeout: read timeout after {INTERN_S1_READ_TIMEOUT_SECONDS:g} seconds"
        except requests.exceptions.ConnectionError as exc:
            last_error = f"connection_error: {_safe_error_message(exc)}"
        except Exception as exc:
            return f"[intern-s1 error] {_safe_error_message(exc)}"

        if attempt >= max_attempts:
            return f"[intern-s1 error] {last_error}"
        time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])

    return f"[intern-s1 error] {last_error or 'unknown request failure'}"
