"""Config loader for configs/cassi.yaml — paper_plan_v2 §17 (single source of truth)."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cassi.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with Path(path or DEFAULT_CONFIG_PATH).open() as f:
        return yaml.safe_load(f)


def require_pilot_calibration(cfg: dict, domain: str) -> None:
    """Scripts past P2 must call this: refuse to run while pilot-frozen fields are null (§17)."""
    allow = cfg["label"]["allowances"][domain]
    med = cfg["label"]["cost_normalization"][f"{domain}_median_pilot_spend"]
    missing = [k for k, v in allow.items() if v is None] + ([f"{domain}_median_pilot_spend"] if med is None else [])
    if missing:
        raise RuntimeError(
            f"Pilot calibration missing for domain '{domain}': {missing}. "
            "Run scripts/p2_pilot_and_collect.sh first and write the values into configs/cassi.yaml "
            "(paper_plan_v2 §16 P2: small=P25 spend, medium=P75, large=2×P90; "
            "median spend is the cost-normalization constant)."
        )
