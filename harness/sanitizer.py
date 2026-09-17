"""Secret sanitization utilities for Phase A4.4 Probing Harness.

Guarantees that credentials, tokens, and sensitive values are never logged,
serialized, printed, or included in probe telemetry or error snippets.
"""

import re
from typing import Any, Dict, List, Optional


# Regex patterns matching common provider API key formats and authorization headers
SECRET_PATTERNS = [
    re.compile(r"Bearer\s+([A-Za-z0-9_\-\.]{8,})", re.IGNORECASE),
    re.compile(r"gsk_[A-Za-z0-9]{10,}", re.IGNORECASE),
    re.compile(r"tvly-[A-Za-z0-9]{10,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{10,}", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|secret|password|token|auth)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{8,})['\"]?", re.IGNORECASE),
]


def extract_secrets_from_env(
    environ: Optional[Dict[str, str]] = None,
    auth_env_var: Optional[str] = None,
) -> List[str]:
    """Extracts sensitive token strings from environment for redaction matching."""
    if environ is None:
        return []

    secrets: List[str] = []
    if auth_env_var and auth_env_var in environ:
        val = str(environ[auth_env_var]).strip()
        if len(val) >= 4:
            secrets.append(val)

    for k, v in environ.items():
        k_upper = str(k).upper()
        if any(s in k_upper for s in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH"]):
            val = str(v).strip()
            if len(val) >= 6 and val not in secrets:
                secrets.append(val)

    return secrets


def sanitize_text(
    text: Optional[str],
    secrets: Optional[List[str]] = None,
) -> Optional[str]:
    """Redacts any known or pattern-detected secret strings from text."""
    if text is None:
        return None

    sanitized = str(text)

    # 1. Exact string redactions for known secrets
    if secrets:
        for secret in secrets:
            if secret and len(secret) >= 4:
                sanitized = sanitized.replace(secret, "[REDACTED_SECRET]")

    # 2. Pattern-based redactions
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)

    return sanitized


def sanitize_dict(
    data: Dict[str, Any],
    secrets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Recursively sanitizes string values and keys in a dictionary."""
    clean: Dict[str, Any] = {}
    for k, v in data.items():
        k_clean = str(sanitize_text(str(k), secrets))
        if isinstance(v, str):
            clean[k_clean] = sanitize_text(v, secrets)
        elif isinstance(v, dict):
            clean[k_clean] = sanitize_dict(v, secrets)
        elif isinstance(v, list):
            clean[k_clean] = [
                sanitize_text(x, secrets) if isinstance(x, str)
                else sanitize_dict(x, secrets) if isinstance(x, dict)
                else x
                for x in v
            ]
        else:
            clean[k_clean] = v
    return clean

