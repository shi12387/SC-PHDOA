"""Single source of truth for the released paper protocol."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "final_protocol.json"
DATA_PATHS_PATH = REPOSITORY_ROOT / "configs" / "data_paths.json"


@lru_cache(maxsize=1)
def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    required = {
        "random_seed", "context_scenes", "guard_scenes", "target_scenes",
        "maximum_sources", "als_iterations", "als_ridge", "als_blend",
        "final_iq_refiner", "guard_non_residual_branch", "datasets",
    }
    missing = sorted(required.difference(protocol))
    if missing:
        raise RuntimeError(f"Final protocol is missing fields: {missing}")
    expected = {
        "maximum_sources": DEFAULT_CONFIG.max_sources,
        "als_iterations": DEFAULT_CONFIG.als_iterations,
        "als_ridge": DEFAULT_CONFIG.als_ridge,
        "als_blend": DEFAULT_CONFIG.als_blend,
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise RuntimeError(
                f"Protocol/config mismatch for {key}: "
                f"{protocol[key]!r} != {value!r}"
            )
    if protocol["final_iq_refiner"] is not False:
        raise RuntimeError("The released final protocol must disable the IQ refiner")
    if protocol["guard_non_residual_branch"] != "nominal_manifold":
        raise RuntimeError("The guard fallback must be the nominal manifold")
    return protocol


def protocol_fingerprint() -> str:
    canonical = json.dumps(load_protocol(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_path(name: str) -> Path:
    protocol = load_protocol()
    try:
        filename = protocol["datasets"][name]
    except KeyError as error:
        raise KeyError(f"Unknown paper dataset: {name}") from error
    primary = REPOSITORY_ROOT / "data" / filename
    if not primary.is_file() and DATA_PATHS_PATH.is_file():
        alternatives = json.loads(DATA_PATHS_PATH.read_text(encoding="utf-8"))
        candidate = alternatives.get(name)
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return primary


def output_path(name: str) -> Path:
    return REPOSITORY_ROOT / "outputs" / name
