"""Inference-only contract shared with the separately trained checkpoint."""

from __future__ import annotations

import torch
from torch import Tensor, nn

LABELS: tuple[str, ...] = ("fold", "check_call", "aggress")

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD_CODES = tuple(f"{rank}{suit}" for rank in _RANKS for suit in _SUITS)
_SCALAR_FEATURE_NAMES = (
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "player_count",
    "active_player_count",
    "position",
    "log_pot_bb",
    "log_stack_bb",
    "log_effective_stack_bb",
    "log_to_call_bb",
    "log_street_contribution_bb",
    "log_current_bet_bb",
    "log_min_raise_to_bb",
    "pot_odds",
    "spr",
    "raises_current_street",
    "legal_fold",
    "legal_check_call",
    "legal_aggress",
    "hole_known_fraction",
)
FEATURE_NAMES: tuple[str, ...] = (
    *(f"hole_{card}" for card in _CARD_CODES),
    *(f"board_{card}" for card in _CARD_CODES),
    *_SCALAR_FEATURE_NAMES,
)


class TinyPolicy(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(LABELS)),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def mask_illegal_logits(logits: Tensor, features: Tensor) -> Tensor:
    legal_columns = [
        FEATURE_NAMES.index(name)
        for name in ("legal_fold", "legal_check_call", "legal_aggress")
    ]
    legal = features[..., legal_columns] > 0.5
    if not torch.all(legal.any(dim=-1)):
        raise ValueError("each decision must have at least one legal action")
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def validate_checkpoint_contract(checkpoint: dict[str, object]) -> None:
    if int(checkpoint.get("input_size", -1)) != len(FEATURE_NAMES):
        raise ValueError("checkpoint input size does not match the Playground feature contract")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature names do not match the Playground feature contract")
    if tuple(checkpoint.get("labels", ())) != LABELS:
        raise ValueError("checkpoint labels do not match the Playground action contract")


__all__ = [
    "FEATURE_NAMES",
    "LABELS",
    "TinyPolicy",
    "mask_illegal_logits",
    "validate_checkpoint_contract",
]
