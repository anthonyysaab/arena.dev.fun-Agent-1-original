"""Export the trained tiny-policy checkpoint to torch-free JSON weights.

Runs wherever torch is installed (for example the poker-nn-training venv):

    python tools/export_pure_weights.py
    python tools/export_pure_weights.py --checkpoint path\\to\\tiny-policy.pt

The export carries the full contract metadata so
``devfun_poker_playground.pure_model`` re-validates it at load time, plus the
source checkpoint's SHA-256 for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from devfun_poker_playground.model_contract import (
    validate_checkpoint_contract,
)
from devfun_poker_playground.playground import PlaygroundPolicy
from devfun_poker_playground.pure_model import validate_pure_weights


def export_weights(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_contract(checkpoint)
    state = checkpoint["model_state_dict"]
    document = {
        "format": "devfun-poker-playground-pure-weights",
        "format_version": 1,
        "source_checkpoint": checkpoint_path.name,
        "source_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "input_size": int(checkpoint["input_size"]),
        "hidden_size": int(checkpoint["hidden_size"]),
        "feature_names": list(checkpoint["feature_names"]),
        "labels": list(checkpoint["labels"]),
        "w1": state["network.0.weight"].double().tolist(),
        "b1": state["network.0.bias"].double().tolist(),
        "w2": state["network.2.weight"].double().tolist(),
        "b2": state["network.2.bias"].double().tolist(),
    }
    table_sizes = checkpoint.get("table_sizes")
    if isinstance(table_sizes, (list, tuple)) and table_sizes:
        document["table_sizes"] = [int(size) for size in table_sizes]
    validate_pure_weights(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="tiny-policy.pt path (default: POKER_POLICY_CHECKPOINT or repo candidates)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tiny-policy-pure.json",
        help="output JSON path (default: artifacts/tiny-policy-pure.json)",
    )
    args = parser.parse_args()

    checkpoint_path = PlaygroundPolicy._checkpoint_path(args.checkpoint)
    document = export_weights(checkpoint_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
    size_kb = args.out.stat().st_size / 1024
    print(f"exported {checkpoint_path} -> {args.out} ({size_kb:.0f} KiB)")
    print(f"source sha256: {document['source_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
