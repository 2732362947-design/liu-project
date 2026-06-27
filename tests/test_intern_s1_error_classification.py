import requests

from intern_s1_client import classify_error, classify_http_status


def test_dns_name_or_service_not_known():
    error = "ConnectionError: Name or service not known"

    assert classify_error(error) == "dns_error"


def test_dns_temporary_failure():
    error = "ConnectionError: Temporary failure in name resolution"

    assert classify_error(error) == "dns_error"


def test_connection_error_without_dns_keyword():
    error = "NewConnectionError: Failed to establish a new connection"

    assert classify_error(error) == "connection_error"


def test_requests_connection_error_object():
    error = requests.exceptions.ConnectionError("Max retries exceeded")

    assert classify_error(error) == "connection_error"


def test_timeout_read_timed_out():
    error = "HTTPSConnectionPool: Read timed out"

    assert classify_error(error) == "timeout"


def test_requests_timeout_object():
    error = requests.exceptions.ReadTimeout("Read timed out")

    assert classify_error(error) == "timeout"


def test_http_401_auth_error():
    assert classify_http_status(401) == "auth_error"
    assert classify_error("HTTP 401: unauthorized") == "auth_error"


def test_http_403_forbidden():
    assert classify_http_status(403) == "forbidden"
    assert classify_error("HTTP 403: forbidden") == "forbidden"


def test_other_http_error():
    assert classify_http_status(500) == "http_error"
    assert classify_error("HTTP 500: server error") == "http_error"
