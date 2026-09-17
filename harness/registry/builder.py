"""Registry Entry and Snapshot Builder for Phase A4.7.

Translates empirical CandidateCapabilityProfile artifacts into structured
CandidateRegistryEntry objects and versioned RegistrySnapshot documents.
One-directional: A4.6 evidence -> Registry capability metadata.
Contains zero runtime routing, zero model arbitration, and zero ranking.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from harness.models import (
    AccessMode,
    BillingStatus,
    CandidateRecord,
    CandidateType,
    VerificationState,
)
from harness.registry.models import (
    CandidateRegistryEntry,
    CapabilityVerificationRecord,
    RegistrySnapshot,
    RegistryVerificationStatus,
)
from harness.registry.policy import POLICY_VERSION, evaluate_capability_verification
from harness.reporting.profiles import CandidateCapabilityProfile, TaskObservationResult


def build_candidate_registry_entry(
    profile: CandidateCapabilityProfile,
    manifest: Optional[CandidateRecord] = None,
    policy_version: str = POLICY_VERSION,
) -> CandidateRegistryEntry:
    """Builds a CandidateRegistryEntry from empirical capability profile evidence.

    Preserves exact evidence traceability and records all claimed vs verified capabilities.
    """
    candidate_id = profile.candidate_id.strip()

    # Determine metadata defaults from manifest if available
    display_name = manifest.display_name if manifest else candidate_id
    candidate_type = manifest.candidate_type if manifest else CandidateType.INFERENCE_MODEL
    provider_id = manifest.provider_id if manifest else "unknown_provider"
    access_mode = manifest.access_mode if manifest else AccessMode.API
    billing_status = manifest.billing_status if manifest else BillingStatus.UNKNOWN
    claimed_caps = manifest.claimed_capabilities if manifest else []

    # 1. Group task results by capability (task_category)
    tasks_by_cap: Dict[str, List[TaskObservationResult]] = {}
    for task in profile.task_results:
        cap = task.task_category
        if cap not in tasks_by_cap:
            tasks_by_cap[cap] = []
        tasks_by_cap[cap].append(task)

    # 2. Evaluate verification status for each empirically tested capability
    verified_caps: Dict[str, CapabilityVerificationRecord] = {}
    for cap, tasks in tasks_by_cap.items():
        rec = evaluate_capability_verification(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            capability=cap,
            tasks=tasks,
            benchmark_id=profile.benchmark_suite_id,
            benchmark_version=profile.benchmark_suite_version,
            policy_version=policy_version,
        )
        verified_caps[cap] = rec

    # 3. Handle manifest claims that were NOT evaluated in benchmark suite
    for claim in claimed_caps:
        if claim not in verified_caps:
            verified_caps[claim] = CapabilityVerificationRecord(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                capability=claim,
                verification_status=RegistryVerificationStatus.UNVERIFIED,
                benchmark_id=profile.benchmark_suite_id,
                benchmark_version=profile.benchmark_suite_version,
                observed_task_ids=[],
                passed_task_ids=[],
                failed_task_ids=[],
                blocked_task_ids=[],
                evidence_ids=[],
                observation_count=0,
                contract_satisfied_count=0,
                contract_satisfaction_rate=0.0,
                policy_version=policy_version,
                notes="Claimed in candidate manifest but not evaluated in benchmark suite.",
            )

    # Sort verified capabilities deterministically by name
    sorted_verified_caps = {k: verified_caps[k] for k in sorted(verified_caps.keys())}

    # 4. Compute aggregate coarse verification state
    if not profile.task_results:
        overall_state = VerificationState.UNVERIFIED
    else:
        tested_records = [r for r in sorted_verified_caps.values() if r.observation_count > 0]
        has_fails = any(r.verification_status == RegistryVerificationStatus.VERIFIED_FAIL for r in tested_records)
        all_passes = all(r.verification_status == RegistryVerificationStatus.VERIFIED_PASS for r in tested_records)
        all_blocked = all(r.verification_status == RegistryVerificationStatus.BLOCKED for r in tested_records)

        if has_fails:
            overall_state = VerificationState.VERIFIED_FAIL
        elif all_passes and tested_records:
            overall_state = VerificationState.VERIFIED_PASS
        elif all_blocked and tested_records:
            overall_state = VerificationState.REQUIRES_REVIEW
        else:
            overall_state = VerificationState.REQUIRES_REVIEW

    return CandidateRegistryEntry(
        candidate_id=candidate_id,
        display_name=display_name,
        candidate_type=candidate_type,
        provider_id=provider_id,
        access_mode=access_mode,
        billing_status=billing_status,
        claimed_capabilities=sorted(claimed_caps),
        verified_capabilities=sorted_verified_caps,
        overall_verification_state=overall_state,
        last_verified_at=datetime.now(timezone.utc).isoformat() if profile.task_results else None,
    )


def build_registry_snapshot(
    profiles: List[CandidateCapabilityProfile],
    manifests: Optional[Dict[str, CandidateRecord]] = None,
    snapshot_id: Optional[str] = None,
    benchmark_suite_id: Optional[str] = None,
    benchmark_suite_version: Optional[str] = None,
    policy_version: str = POLICY_VERSION,
) -> RegistrySnapshot:
    """Builds a complete, deterministic RegistrySnapshot from candidate capability profiles."""
    manifests_map = manifests or {}

    # Deterministic sorting of input profiles by candidate_id
    sorted_profiles = sorted(profiles, key=lambda p: p.candidate_id)

    suite_id = benchmark_suite_id or (sorted_profiles[0].benchmark_suite_id if sorted_profiles else "unknown_suite")
    suite_ver = benchmark_suite_version or (sorted_profiles[0].benchmark_suite_version if sorted_profiles else "1.0.0")

    entries: Dict[str, CandidateRegistryEntry] = {}
    for p in sorted_profiles:
        manifest = manifests_map.get(p.candidate_id)
        entry = build_candidate_registry_entry(p, manifest=manifest, policy_version=policy_version)
        entries[entry.candidate_id] = entry

    # Deterministic sorting of entries by key
    sorted_entries = {k: entries[k] for k in sorted(entries.keys())}

    sid = snapshot_id or f"reg_snap_{suite_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return RegistrySnapshot(
        snapshot_id=sid,
        benchmark_suite_id=suite_id,
        benchmark_suite_version=suite_ver,
        policy_version=policy_version,
        candidate_count=len(sorted_entries),
        entries=sorted_entries,
    )

