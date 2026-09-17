"""Phase A4.7: Baseline Profile Archival & Registry Capability Verification.

Provides deterministic conversion of empirical CandidateCapabilityProfile evidence
into registry capability metadata and immutable, versioned snapshots.
"""

from harness.registry.archival import (
    list_registry_snapshots,
    load_registry_snapshot,
    save_registry_snapshot,
)
from harness.registry.builder import (
    build_candidate_registry_entry,
    build_registry_snapshot,
)
from harness.registry.models import (
    CandidateRegistryEntry,
    CapabilityVerificationRecord,
    RegistrySnapshot,
    RegistryVerificationStatus,
)
from harness.registry.policy import (
    POLICY_VERSION,
    evaluate_capability_verification,
)

__all__ = [
    "RegistryVerificationStatus",
    "CapabilityVerificationRecord",
    "CandidateRegistryEntry",
    "RegistrySnapshot",
    "POLICY_VERSION",
    "evaluate_capability_verification",
    "build_candidate_registry_entry",
    "build_registry_snapshot",
    "save_registry_snapshot",
    "load_registry_snapshot",
    "list_registry_snapshots",
]

