"""Base Provider Adapter interface for Phase A4.4.

Defines the common adapter contract, transport encapsulation, error classification,
and sanitization boundaries for all provider probing operations.
"""

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from harness.models import (
    AccessMode,
    CandidateRecord,
    CandidateType,
    FailureClass,
    FailureClassification,
    ProbeRequest,
    ProbeResult,
    ProbeStatus,
)
from harness.sanitizer import extract_secrets_from_env, sanitize_dict, sanitize_text
from harness.taxonomy import classify_failure


class TransportResponse:
    """Encapsulates transport response status, body, and headers."""

    def __init__(
        self,
        status_code: int,
        text: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        return json.loads(self.text)


TransportCallable = Callable[
    [str, str, Optional[Dict[str, str]], Optional[Dict[str, Any]], float],
    TransportResponse,
]


def default_http_transport(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> TransportResponse:
    """Default HTTP transport using Python standard library urllib."""
    req_headers = headers.copy() if headers else {}
    req_data = None
    if json_data is not None:
        req_data = json.dumps(json_data).encode("utf-8")
        if "Content-Type" not in req_headers:
            req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url=url,
        data=req_data,
        headers=req_headers,
        method=method.upper(),
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.getcode()
            body_bytes = response.read()
            body_text = body_bytes.decode("utf-8", errors="replace")
            resp_headers = dict(response.info())
            return TransportResponse(status_code=status, text=body_text, headers=resp_headers)
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        body_text = body_bytes.decode("utf-8", errors="replace")
        resp_headers = dict(e.headers or {})
        return TransportResponse(status_code=e.code, text=body_text, headers=resp_headers)


class BaseProviderAdapter(ABC):
    """Abstract base class for candidate provider adapters."""

    def __init__(self, candidate: CandidateRecord):
        self.candidate = candidate
        self.candidate_id = candidate.candidate_id
        self.candidate_type = candidate.candidate_type
        self.access_mode = candidate.access_mode

    @abstractmethod
    def execute_probe(
        self,
        request: ProbeRequest,
        environ: Optional[Dict[str, str]] = None,
        transport: Optional[TransportCallable] = None,
    ) -> ProbeResult:
        """Executes a controlled capability probe against the candidate provider.

        Args:
            request: The structured probe request.
            environ: Optional environment dictionary (defaults to os.environ).
            transport: Optional injectable HTTP transport callable for offline testing.

        Returns:
            Normalized ProbeResult with zero secrets.
        """
        pass

    def _sanitize(
        self,
        text: Optional[str],
        environ: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Sanitizes text using known environment secrets."""
        secrets = extract_secrets_from_env(environ, self.candidate.auth_env_var)
        return sanitize_text(text, secrets)

    def _classify_adapter_error(
        self,
        http_status: Optional[int] = None,
        error_code: Optional[str] = None,
        raw_message: Optional[str] = None,
        environ: Optional[Dict[str, str]] = None,
    ) -> FailureClassification:
        """Sanitizes and deterministically categorizes an adapter-level error."""
        clean_msg = self._sanitize(raw_message, environ)
        return classify_failure(
            http_status=http_status,
            error_code=error_code,
            raw_message=clean_msg,
        )

