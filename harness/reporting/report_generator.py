"""Deterministic Report Generator for Phase A4.6.

Generates human-readable Markdown reports and machine-readable JSON exports
from CandidateCapabilityProfile artifacts.
Strictly observational: zero model rankings, zero leaderboards, zero subjective ratings.
"""

import json
from typing import List

from harness.reporting.profiles import CandidateCapabilityProfile


def export_profile_json(profile: CandidateCapabilityProfile, indent: int = 2) -> str:
    """Exports a CandidateCapabilityProfile as deterministic, indented JSON."""
    return profile.model_dump_json(indent=indent)


def generate_candidate_markdown_report(profile: CandidateCapabilityProfile) -> str:
    """Generates a structured, evidence-oriented Markdown report for a candidate."""
    lines: List[str] = [
        f"# Candidate Capability Baseline Profile: `{profile.candidate_id}`",
        "",
        "> **Notice**: This profile represents empirical benchmark observations collected under fixed,",
        "> deterministic conditions. It contains zero subjective quality scores, zero model rankings,",
        "> and zero winner designations. It does not grant runtime production routing eligibility.",
        "",
        "## 1. Execution Overview",
        "",
        f"- **Candidate Identifier**: `{profile.candidate_id}`",
        f"- **Benchmark Suite**: `{profile.benchmark_suite_id}` (v`{profile.benchmark_suite_version}`)",
        f"- **Generated At (UTC)**: `{profile.generated_at}`",
        f"- **Total Tasks Evaluated**: {profile.evaluated_task_count}",
        f"- **Successful Invocations**: {profile.successful_task_count}",
        f"- **Failed Invocations**: {profile.failed_task_count}",
        f"- **Blocked Invocations**: {profile.blocked_task_count}",
        "",
        "## 2. Contract Compliance Summary",
        "",
        f"- **Contracts Satisfied**: {profile.contract_satisfaction_observations.satisfied_count} / {profile.contract_satisfaction_observations.total_evaluated}",
        f"- **Contracts Unfulfilled**: {profile.contract_satisfaction_observations.unsatisfied_count}",
        f"- **Contract Satisfaction Rate**: {profile.contract_satisfaction_observations.satisfaction_rate * 100.0:.2f}%",
        "",
        "## 3. Capability Observations by Category",
        "",
        "| Capability Category | Total Tasks | Successful | Contracts Satisfied | Compliance Rate | Supporting Evidence |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    if profile.capability_observations:
        for obs in profile.capability_observations:
            rate_pct = f"{obs.contract_satisfaction_rate * 100.0:.1f}%"
            lines.append(
                f"| `{obs.task_category}` | {obs.total_tasks} | {obs.successful_tasks} | {obs.contracts_satisfied} | {rate_pct} | {obs.evidence_count} records |"
            )
    else:
        lines.append("| *No observations recorded* | - | - | - | - | - |")

    lines.extend([
        "",
        "## 4. Task-Level Empirical Observations",
        "",
        "| Task ID | Category | Status | Contract Met | Latency (ms) | Provenance State | Evidence ID |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    if profile.task_results:
        for tr in profile.task_results:
            sat_str = "YES" if tr.contract_satisfied else "NO"
            lines.append(
                f"| `{tr.task_id}` | `{tr.task_category}` | `{tr.execution_status.value}` | **{sat_str}** | {tr.latency_ms:.2f} | `{tr.provenance_state.value}` | `{tr.evidence_id}` |"
            )
    else:
        lines.append("| *No task results available* | - | - | - | - | - | - |")

    lines.extend([
        "",
        "## 5. Observed Failure Classifications",
        "",
    ])

    if profile.observed_failure_classes:
        lines.append("| Failure Class | Occurrences |")
        lines.append("| :--- | :---: |")
        for fc, cnt in profile.observed_failure_classes.items():
            lines.append(f"| `{fc}` | {cnt} |")
    else:
        lines.append("- *No provider or contract failures observed.*")

    lat = profile.observed_latency_statistics
    lines.extend([
        "",
        "## 6. Observed Latency Statistics",
        "",
        f"- **Valid Measurements**: {lat.sample_count}",
        f"- **Minimum Latency**: {lat.min_ms:.2f} ms",
        f"- **Maximum Latency**: {lat.max_ms:.2f} ms",
        f"- **Mean Latency**: {lat.mean_ms:.2f} ms",
        f"- **Median Latency**: {lat.median_ms:.2f} ms",
        "",
        "## 7. Source Provenance Audit",
        "",
        "> **Provenance Invariant**: Citations asserted by an LLM are tagged as `MODEL_CLAIM_ONLY`",
        "> and are never treated as verified source evidence. Only active system fetches are `RETRIEVED_BY_SYSTEM`.",
        "",
        f"- **Total Provenance Observations**: {profile.provenance_observations.total_observations}",
        f"- **Tasks with `RETRIEVED_BY_SYSTEM`**: {profile.provenance_observations.retrieved_by_system_count}",
        f"- **Tasks with `MODEL_CLAIM_ONLY`**: {profile.provenance_observations.model_claim_only_count}",
        f"- **Tasks with `NOT_AVAILABLE`**: {profile.provenance_observations.not_available_count}",
        f"- **Total Model-Claimed Citation Tokens**: {profile.provenance_observations.total_model_claimed_sources}",
        f"- **Total System-Retrieved Documents**: {profile.provenance_observations.total_system_retrieved_sources}",
        "",
        "## 8. Evidence Traceability",
        "",
        f"- **Associated Benchmark Versions**: {', '.join(f'`{v}`' for v in profile.benchmark_versions)}",
        f"- **Supporting Evidence Records ({len(profile.evidence_references)})**:",
    ])

    for ref in profile.evidence_references:
        lines.append(f"  - `{ref}`")

    lines.append("")
    return "\n".join(lines)


def generate_multi_candidate_comparison_report(
    profiles: List[CandidateCapabilityProfile],
) -> str:
    """Generates an objective, non-ranking comparative matrix across evaluated candidates."""
    sorted_profiles = sorted(profiles, key=lambda p: p.candidate_id)

    lines: List[str] = [
        "# Multi-Candidate Capability Observation Matrix",
        "",
        "> **Strict Boundary**: This matrix presents objective, empirical benchmark observations.",
        "> It does NOT declare winners, ranks, or 'best models'. It provides comparative baseline",
        "> evidence for future informed tool and capability selection policies.",
        "",
        "## 1. Candidate Execution & Contract Summary",
        "",
        "| Candidate ID | Evaluated | Succeeded | Failed | Blocked | Contracts Met | Compliance Rate | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for p in sorted_profiles:
        c_rate = f"{p.contract_satisfaction_observations.satisfaction_rate * 100.0:.1f}%"
        mean_lat = f"{p.observed_latency_statistics.mean_ms:.2f}"
        lines.append(
            f"| `{p.candidate_id}` | {p.evaluated_task_count} | {p.successful_task_count} | {p.failed_task_count} | {p.blocked_task_count} | {p.contract_satisfaction_observations.satisfied_count} | {c_rate} | {mean_lat} |"
        )

    # Collect all unique categories across profiles
    all_categories: List[str] = sorted(
        list({obs.task_category for p in sorted_profiles for obs in p.capability_observations})
    )

    lines.extend([
        "",
        "## 2. Category Compliance Matrix",
        "",
        "| Candidate ID | " + " | ".join(f"`{cat}`" for cat in all_categories) + " |",
        "| :--- | " + " | ".join(":---:" for _ in all_categories) + " |",
    ])

    for p in sorted_profiles:
        cat_map = {obs.task_category: obs for obs in p.capability_observations}
        row_cells = []
        for cat in all_categories:
            if cat in cat_map:
                c_obs = cat_map[cat]
                row_cells.append(f"{c_obs.contracts_satisfied}/{c_obs.total_tasks}")
            else:
                row_cells.append("-")
        lines.append(f"| `{p.candidate_id}` | " + " | ".join(row_cells) + " |")

    lines.extend([
        "",
        "## 3. Source Provenance Standings",
        "",
        "| Candidate ID | System Retrieved Tasks | Model Claim Only Tasks | Total Model Citations | Total System Docs |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for p in sorted_profiles:
        po = p.provenance_observations
        lines.append(
            f"| `{p.candidate_id}` | {po.retrieved_by_system_count} | {po.model_claim_only_count} | {po.total_model_claimed_sources} | {po.total_system_retrieved_sources} |"
        )

    lines.append("")
    return "\n".join(lines)

