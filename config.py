import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

INTERN_S1_API_KEY = os.getenv("INTERN_S1_API_KEY", "")
INTERN_S1_BASE_URL = os.getenv("INTERN_S1_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
INTERN_S1_MODEL = os.getenv("INTERN_S1_MODEL", "intern-latest")
INTERN_S1_TIMEOUT_SECONDS = float(os.getenv("INTERN_S1_TIMEOUT_SECONDS", "45"))
INTERN_S1_MAX_RETRIES = int(os.getenv("INTERN_S1_MAX_RETRIES", "1"))
INTERN_S1_CONNECT_TIMEOUT_SECONDS = float(os.getenv("INTERN_S1_CONNECT_TIMEOUT_SECONDS", "10"))
INTERN_S1_READ_TIMEOUT_SECONDS = float(os.getenv("INTERN_S1_READ_TIMEOUT_SECONDS", "45"))
PIPELINE_MAX_ATTEMPTS = int(os.getenv("PIPELINE_MAX_ATTEMPTS", "2"))
