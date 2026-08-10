"""The pure-Python forward pass must match the torch checkpoint's decisions."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from devfun_poker_playground.contract import FEATURE_NAMES, LABELS
from devfun_poker_playground.model_contract import (
    TinyPolicy,
    mask_illegal_logits,
)
from devfun_poker_playground.playground import PlaygroundPolicy
from devfun_poker_playground.pure_model import (
    TinyPolicyForward,
    masked_family,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_pure_weights import export_weights

_LEGALITY = ("legal_fold", "legal_check_call", "legal_aggress")


def _checkpoint_path() -> Path:
    try:
        return PlaygroundPolicy._checkpoint_path(None)
    except FileNotFoundError:
        pytest.skip("tiny-policy.pt checkpoint not available")


def _random_features(rng: random.Random) -> tuple[float, ...]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    hole = rng.sample(range(52), 2)
    board = rng.sample([i for i in range(52) if i not in hole], rng.randint(0, 5))
    for index in hole:
        values[FEATURE_NAMES[index]] = 1.0
    for index in board:
        values[FEATURE_NAMES[52 + index]] = 1.0
    street = rng.randint(0, 3)
    values[f"street_{('preflop', 'flop', 'turn', 'river')[street]}"] = 1.0
    for name in (
        "player_count",
        "active_player_count",
        "log_pot_bb",
        "log_stack_bb",
        "log_effective_stack_bb",
        "log_to_call_bb",
        "log_street_contribution_bb",
        "log_current_bet_bb",
        "log_min_raise_to_bb",
        "spr",
        "raises_current_street",
    ):
        values[name] = rng.uniform(0.0, 6.0)
    values["position"] = rng.random()
    values["pot_odds"] = rng.random()
    values["hole_known_fraction"] = 1.0
    legality = [rng.random() < 0.7 for _ in _LEGALITY]
    if not any(legality):
        legality[rng.randint(0, 2)] = True
    for name, legal in zip(_LEGALITY, legality):
        values[name] = float(legal)
    return tuple(values[name] for name in FEATURE_NAMES)


def test_exported_weights_reproduce_torch_families() -> None:
    checkpoint_path = _checkpoint_path()
    document = export_weights(checkpoint_path)
    pure = TinyPolicyForward(document)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TinyPolicy(hidden_size=int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(20260809)
    mismatches = 0
    for _ in range(500):
        features = _random_features(rng)
        batch = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            torch_logits = model(batch)
            masked = mask_illegal_logits(torch_logits, batch)
        torch_family = LABELS[int(masked.argmax(dim=1).item())]

        pure_logits = pure.logits(features)
        pure_family = masked_family(pure_logits, features)

        for torch_value, pure_value in zip(torch_logits[0].tolist(), pure_logits):
            assert abs(torch_value - pure_value) < 5e-4

        if torch_family != pure_family:
            # Only acceptable when float32 vs float64 rounding flips a
            # near-exact tie between two legal families.
            ranked = sorted(pure_logits, reverse=True)
            assert ranked[0] - ranked[1] < 1e-3
            mismatches += 1
    assert mismatches <= 2


def test_exported_weights_validate_contract() -> None:
    document = export_weights(_checkpoint_path())
    assert document["input_size"] == len(FEATURE_NAMES)
    assert tuple(document["labels"]) == LABELS
    assert len(document["w1"]) == document["hidden_size"]
    assert all(len(row) == len(FEATURE_NAMES) for row in document["w1"])
    assert len(document["w2"]) == len(LABELS)
