"""Deterministic Registry Snapshot Archival for Phase A4.7.

Manages persistent filesystem storage and retrieval of immutable RegistrySnapshot
artifacts without database dependencies.
Ensures atomic writes, deterministic formatting, and validation integrity.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List

from harness.registry.models import RegistrySnapshot


def save_registry_snapshot(snapshot: RegistrySnapshot, target_path_or_dir: Path) -> Path:
    """Deterministically serializes and saves a RegistrySnapshot to JSON file.

    If target_path_or_dir is a directory, writes to '{directory}/{snapshot_id}.json'.
    Uses atomic rename to guarantee filesystem consistency.
    """
    if target_path_or_dir.is_dir() or target_path_or_dir.suffix != ".json":
        target_path_or_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_path_or_dir / f"{snapshot.snapshot_id}.json"
    else:
        target_path_or_dir.parent.mkdir(parents=True, exist_ok=True)
        final_path = target_path_or_dir

    # Format JSON deterministically
    json_content = snapshot.model_dump_json(indent=2)

    # Atomic write pattern
    dir_path = final_path.parent
    with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8") as tf:
        tf.write(json_content)
        tf.flush()
        temp_name = tf.name

    os.replace(temp_name, final_path)
    return final_path


def load_registry_snapshot(filepath: Path) -> RegistrySnapshot:
    """Loads and validates a RegistrySnapshot artifact from JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Registry snapshot file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RegistrySnapshot.model_validate(data)


def list_registry_snapshots(directory: Path) -> List[Path]:
    """Lists all available JSON snapshot files in the specified directory, sorted alphabetically."""
    if not directory.exists() or not directory.is_dir():
        return []

    return sorted(list(directory.glob("*.json")))

