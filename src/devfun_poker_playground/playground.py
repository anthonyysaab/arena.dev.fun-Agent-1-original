"""Pure decision adapter for dev.fun Poker Playground table snapshots.

This module never performs network I/O.  A runner may pass each fresh Arena
table snapshot to :func:`decide`, then submit the returned payload itself.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from devfun_poker_playground.equity import estimate_equity
from devfun_poker_playground.model_contract import (
    FEATURE_NAMES,
    LABELS,
    TinyPolicy,
    mask_illegal_logits,
    validate_checkpoint_contract,
)

_STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_AGGRESSIVE_ACTIONS = ("bet", "raise", "all-in")


class ArenaSnapshotError(ValueError):
    """Raised when an untrusted Arena snapshot is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ArenaAction:
    action: str
    amount: int | None
    message: str
    reasoning: str | None = None

    def to_payload(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "action": self.action,
            "message": self.message,
        }
        if self.amount is not None:
            payload["amount"] = self.amount
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaSnapshotError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArenaSnapshotError(f"{name} must be an array")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArenaSnapshotError(f"{name} must be a number")
    integer = int(value)
    if integer != value or integer < minimum:
        raise ArenaSnapshotError(f"{name} must be an integer >= {minimum}")
    return integer


def _normalize_card(value: object) -> str:
    if not isinstance(value, str):
        raise ArenaSnapshotError("cards must be strings")
    token = value.strip()
    if len(token) == 3 and token[:2] == "10":
        token = f"T{token[2]}"
    if len(token) != 2:
        raise ArenaSnapshotError(f"invalid card {value!r}")
    card = f"{token[0].upper()}{token[1].lower()}"
    if card[0] not in "23456789TJQKA" or card[1] not in "cdhs":
        raise ArenaSnapshotError(f"invalid card {value!r}")
    return card


def _cards(values: object, name: str, expected: int | None = None) -> tuple[str, ...]:
    cards = tuple(_normalize_card(value) for value in _sequence(values, name))
    if expected is not None and len(cards) != expected:
        raise ArenaSnapshotError(f"{name} must contain exactly {expected} cards")
    if len(set(cards)) != len(cards):
        raise ArenaSnapshotError(f"{name} contains duplicate cards")
    return cards


def _hero_and_seats(
    table: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    self_seat_number = _integer(table.get("selfSeatNumber"), "selfSeatNumber", minimum=1)
    seats = [
        _mapping(value, f"seats[{index}]")
        for index, value in enumerate(_sequence(table.get("seats"), "seats"))
    ]
    hero = next(
        (
            seat
            for seat in seats
            if seat.get("seatNumber") is not None
            and _integer(seat.get("seatNumber"), "seatNumber", minimum=1)
            == self_seat_number
        ),
        None,
    )
    if hero is None:
        raise ArenaSnapshotError("selfSeatNumber does not match any seat")
    return hero, seats


def _blind_seats(table: Mapping[str, Any]) -> tuple[int | None, int | None]:
    small_blind = _integer(table.get("smallBlindChips"), "smallBlindChips", minimum=1)
    big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
    small_seat: int | None = None
    big_seat: int | None = None
    for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
        event = _mapping(raw_event, "recentEvent")
        if event.get("type") != "BlindPosted":
            continue
        summary_value = event.get("summary")
        if summary_value is None:
            continue
        summary = _mapping(summary_value, "recentEvent.summary")
        if summary.get("seatNumber") is None or summary.get("amount") is None:
            continue
        seat_number = _integer(summary["seatNumber"], "blind seat", minimum=1)
        amount = _integer(summary["amount"], "blind amount", minimum=0)
        if amount == small_blind:
            small_seat = seat_number
        if amount == big_blind:
            big_seat = seat_number
    return small_seat, big_seat


def _position(table: Mapping[str, Any], seats: list[Mapping[str, Any]], hero: Mapping[str, Any]) -> float:
    seat_numbers = sorted(
        _integer(seat["seatNumber"], "seatNumber", minimum=1)
        for seat in seats
        if seat.get("seatNumber") is not None
    )
    hero_number = _integer(hero.get("seatNumber"), "hero seatNumber", minimum=1)
    if hero_number not in seat_numbers:
        raise ArenaSnapshotError("hero seat is not in the seated player list")
    if len(seat_numbers) <= 1:
        return 0.0

    small_seat, big_seat = _blind_seats(table)
    if big_seat in seat_numbers:
        if len(seat_numbers) == 2:
            other = next(number for number in seat_numbers if number != big_seat)
            ordered = [big_seat, other]
        elif small_seat in seat_numbers and small_seat != big_seat:
            pivot = seat_numbers.index(big_seat)
            clockwise = seat_numbers[pivot + 1 :] + seat_numbers[:pivot]
            ordered = [small_seat, big_seat, *(n for n in clockwise if n != small_seat)]
        else:
            ordered = seat_numbers
    else:
        ordered = seat_numbers
    return ordered.index(hero_number) / (len(ordered) - 1)


def _log_bb(amount: float, big_blind: int) -> float:
    return math.log1p(max(0.0, amount) / big_blind)


def _aggression_count(table: Mapping[str, Any], street: str) -> int:
    count = 0
    for raw_event in _sequence(table.get("recentEvents") or [], "recentEvents"):
        event = _mapping(raw_event, "recentEvent")
        if str(event.get("street") or "").casefold() != street:
            continue
        summary_value = event.get("summary")
        if summary_value is None:
            continue
        action = str(_mapping(summary_value, "recentEvent.summary").get("action") or "")
        if action.casefold() in _AGGRESSIVE_ACTIONS:
            count += 1
    return count


def features_from_table(
    table: Mapping[str, Any],
    *,
    position: float | None = None,
) -> tuple[float, ...]:
    """Convert one fresh Arena table snapshot to the training feature contract."""

    hero, seats = _hero_and_seats(table)
    allowed = _mapping(table.get("allowedActions"), "allowedActions")
    available = {
        str(value)
        for value in _sequence(allowed.get("availableActions"), "availableActions")
    }
    if not available:
        raise ArenaSnapshotError("no actions are available")

    street = str(table.get("street") or "").casefold()
    if street not in _STREET_INDEX:
        raise ArenaSnapshotError(f"unsupported decision street {table.get('street')!r}")
    hole_cards = _cards(hero.get("holeCards"), "hero holeCards", expected=2)
    board_cards = _cards(table.get("boardCards"), "boardCards")
    if len(board_cards) > 5 or len({*hole_cards, *board_cards}) != len(hole_cards) + len(
        board_cards
    ):
        raise ArenaSnapshotError("hole and board cards must be unique and board length <= 5")

    big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
    pot = _integer(table.get("potChips"), "potChips")
    stack = _integer(hero.get("stackChips"), "hero stackChips")
    contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
    current_bet = _integer(table.get("currentBet"), "currentBet")
    to_call = _integer(allowed.get("callChips", 0), "allowedActions.callChips")
    min_raise_to = allowed.get("minRaiseTo")
    if min_raise_to is None:
        min_raise_to = allowed.get("minBet") or 0
    min_raise_to = _integer(min_raise_to, "minimum aggressive amount")

    active_seats = [
        seat
        for seat in seats
        if str(seat.get("status") or "").casefold() not in {"folded", "settled"}
    ]
    opponent_stacks = [
        _integer(seat.get("stackChips"), "opponent stackChips")
        for seat in active_seats
        if seat is not hero
    ]
    effective_stack = min(stack, max(opponent_stacks, default=stack))

    legal_fold = bool(allowed.get("canFold")) and "fold" in available
    legal_check_call = (
        (bool(allowed.get("canCheck")) and "check" in available)
        or (bool(allowed.get("canCall")) and "call" in available)
    )
    legal_aggress = any(action in available for action in _AGGRESSIVE_ACTIONS)
    if not (legal_fold or legal_check_call or legal_aggress):
        raise ArenaSnapshotError("allowedActions contains no supported legal action")

    values = {name: 0.0 for name in FEATURE_NAMES}
    for card in hole_cards:
        values[f"hole_{card}"] = 1.0
    for card in board_cards:
        values[f"board_{card}"] = 1.0
    for index in range(4):
        values[f"street_{('preflop', 'flop', 'turn', 'river')[index]}"] = float(
            index == _STREET_INDEX[street]
        )
    if position is None:
        position = _position(table, seats, hero)
    if not math.isfinite(position) or not 0.0 <= position <= 1.0:
        raise ArenaSnapshotError("position must be between 0 and 1")

    values.update(
        {
            "player_count": float(len(seats)),
            "active_player_count": float(len(active_seats)),
            "position": position,
            "log_pot_bb": _log_bb(pot, big_blind),
            "log_stack_bb": _log_bb(stack, big_blind),
            "log_effective_stack_bb": _log_bb(effective_stack, big_blind),
            "log_to_call_bb": _log_bb(to_call, big_blind),
            "log_street_contribution_bb": _log_bb(contribution, big_blind),
            "log_current_bet_bb": _log_bb(current_bet, big_blind),
            "log_min_raise_to_bb": _log_bb(min_raise_to, big_blind),
            "pot_odds": 0.0 if to_call <= 0 else to_call / max(pot + to_call, 1),
            "spr": 0.0 if pot <= 0 else effective_stack / pot,
            "raises_current_street": float(_aggression_count(table, street)),
            "legal_fold": float(legal_fold),
            "legal_check_call": float(legal_check_call),
            "legal_aggress": float(legal_aggress),
            "hole_known_fraction": 1.0,
        }
    )
    features = tuple(values[name] for name in FEATURE_NAMES)
    if not all(math.isfinite(value) for value in features):
        raise ArenaSnapshotError("snapshot produced non-finite model features")
    return features


class PlaygroundPolicy:
    """Tiny neural proposal policy with deterministic Arena safety rails."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: nn.Module | None = None,
        equity_trials: int = 100,
        seed: int = 7,
    ) -> None:
        if equity_trials < 0:
            raise ValueError("equity_trials cannot be negative")
        self.equity_trials = equity_trials
        self.seed = seed
        if model is None:
            path = self._checkpoint_path(checkpoint_path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            validate_checkpoint_contract(checkpoint)
            model = TinyPolicy(hidden_size=int(checkpoint["hidden_size"]))
            model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.eval()

    @staticmethod
    def _checkpoint_path(value: str | Path | None) -> Path:
        configured = value or os.environ.get("POKER_POLICY_CHECKPOINT")
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file():
                return path
            raise FileNotFoundError(f"policy checkpoint not found at {path}")

        project_root = Path(__file__).resolve().parents[2]
        candidates = (
            project_root / "artifacts" / "tiny-policy.pt",
            project_root.parent / "poker-nn-training" / "artifacts" / "tiny-policy.pt",
        )
        for path in candidates:
            if path.is_file():
                return path
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "policy checkpoint not found; set POKER_POLICY_CHECKPOINT "
            f"or place it at one of: {checked}"
        )

    def _family(self, features: tuple[float, ...]) -> str:
        batch = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            logits: Tensor = self.model(batch)
            logits = mask_illegal_logits(logits, batch)
        return LABELS[int(logits.argmax(dim=1).item())]

    def _equity(self, table: Mapping[str, Any]) -> float | None:
        if self.equity_trials == 0:
            return None
        hero, seats = _hero_and_seats(table)
        hole_cards = _cards(hero.get("holeCards"), "holeCards", 2)
        hole = (hole_cards[0], hole_cards[1])
        board = _cards(table.get("boardCards"), "boardCards")
        opponents = sum(
            1
            for seat in seats
            if seat is not hero
            and str(seat.get("status") or "").casefold() not in {"folded", "settled"}
        )
        table_id = str(table.get("tableId") or table.get("id") or "")
        digest = hashlib.sha256(f"{self.seed}:{table_id}".encode()).digest()
        trial_seed = int.from_bytes(digest[:8], "big")
        return estimate_equity(
            hole,
            board,
            opponents,
            trials=self.equity_trials,
            seed=trial_seed,
        )

    def decide(
        self,
        table: Mapping[str, Any],
        deadline_s: float = 10.0,
        research_context: Mapping[str, Any] | None = None,
    ) -> dict[str, str | int]:
        context = research_context or {}
        cached_position = context.get("position")
        if cached_position is not None:
            try:
                cached_position = float(cached_position)
            except (TypeError, ValueError) as exc:
                raise ArenaSnapshotError("research_context.position must be a number") from exc
        features = features_from_table(table, position=cached_position)
        allowed = _mapping(table.get("allowedActions"), "allowedActions")
        available = {
            str(value)
            for value in _sequence(allowed.get("availableActions"), "availableActions")
        }
        if deadline_s < 2.0:
            action = self._deadline_action(table, allowed, available)
            return self._render(action, table, allowed, equity=None).to_payload()

        equity = self._equity(table)
        _, seats = _hero_and_seats(table)
        family = (
            self._short_handed_family(table, allowed, available, equity)
            if len(seats) < 6 and equity is not None
            else self._family(features)
        )
        if family == "aggress":
            action = self._aggressive_action(table, allowed, available, equity)
        elif family == "check_call":
            action = self._passive_action(table, allowed, available, equity)
        else:
            action = self._fold_action(table, allowed, available, equity)
        return self._render(action, table, allowed, equity).to_payload()

    def _short_handed_family(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float,
    ) -> str:
        _, seats = _hero_and_seats(table)
        opponent_count = max(1, len(seats) - 1)
        aggression_floor = min(0.72, 0.52 + 0.05 * max(0, opponent_count - 1))
        if any(action in available for action in ("bet", "raise")) and equity >= aggression_floor:
            return "aggress"
        if "check" in available:
            return "check_call"
        if "call" in available and equity + 0.03 >= self._pot_odds(table, allowed):
            return "check_call"
        return "fold"

    def _fold_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        pot_odds = self._pot_odds(table, allowed)
        if "call" in available and equity is not None and equity >= max(0.60, pot_odds + 0.15):
            return "call", None
        if "fold" in available:
            return "fold", None
        return self._passive_action(table, allowed, available, equity)

    def _passive_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        if "call" in available:
            pot_odds = self._pot_odds(table, allowed)
            if equity is None or equity + 0.03 >= pot_odds:
                return "call", None
        if "fold" in available:
            return "fold", None
        if "call" in available:
            return "call", None
        return self._first_legal_aggression(table, allowed, available)

    def _aggressive_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
        equity: float | None,
    ) -> tuple[str, int | None]:
        hero, seats = _hero_and_seats(table)
        opponent_count = sum(
            1
            for seat in seats
            if seat is not hero
            and str(seat.get("status") or "").casefold() not in {"folded", "settled"}
        )
        aggression_floor = min(0.74, 0.50 + 0.04 * max(0, opponent_count - 1))
        if equity is not None and equity < aggression_floor:
            return self._passive_action(table, allowed, available, equity)

        if "bet" in available:
            sized = self._sized_action("bet", table, allowed, equity)
            if sized is not None:
                return sized
        if "raise" in available:
            sized = self._sized_action("raise", table, allowed, equity)
            if sized is not None:
                return sized
        # The warm-start model never controls an optional all-in in v0.
        return self._passive_action(table, allowed, available, equity)

    def _sized_action(
        self,
        action: str,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> tuple[str, int] | None:
        hero, _ = _hero_and_seats(table)
        pot = _integer(table.get("potChips"), "potChips")
        big_blind = _integer(table.get("bigBlindChips"), "bigBlindChips", minimum=1)
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        contribution = _integer(hero.get("currentBetChips"), "hero currentBetChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")

        range_name = "betRange" if action == "bet" else "raiseRange"
        amount_range = _mapping(allowed.get(range_name), range_name)
        minimum = _integer(amount_range.get("min"), f"{range_name}.min")
        maximum = _integer(amount_range.get("max"), f"{range_name}.max")
        base = contribution if action == "bet" else _integer(
            allowed.get("callToAmount"), "callToAmount"
        )
        desired = base + max(big_blind, round(0.5 * (pot + call_chips)))

        if equity is None or equity < 0.72:
            risk_cap = contribution + max(big_blind, round(0.35 * stack))
            maximum = min(maximum, risk_cap)
        if maximum < minimum:
            return None
        return action, min(max(desired, minimum), maximum)

    def _first_legal_aggression(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[str, int | None]:
        for action in ("bet", "raise"):
            if action in available:
                sized = self._sized_action(action, table, allowed, equity=None)
                if sized is not None:
                    return sized
        if "all-in" in available:
            return "all-in", _integer(allowed.get("allInToAmount"), "allInToAmount")
        raise ArenaSnapshotError("no legal fallback action")

    def _deadline_action(
        self,
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        available: set[str],
    ) -> tuple[str, int | None]:
        if "check" in available:
            return "check", None
        hero, _ = _hero_and_seats(table)
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        if "call" in available and call_chips <= max(1, round(0.05 * stack)):
            return "call", None
        if "fold" in available:
            return "fold", None
        if "call" in available:
            return "call", None
        return self._first_legal_aggression(table, allowed, available)

    @staticmethod
    def _pot_odds(table: Mapping[str, Any], allowed: Mapping[str, Any]) -> float:
        pot = _integer(table.get("potChips"), "potChips")
        call_chips = _integer(allowed.get("callChips", 0), "callChips")
        return 0.0 if call_chips == 0 else call_chips / max(pot + call_chips, 1)

    def _render(
        self,
        action: tuple[str, int | None],
        table: Mapping[str, Any],
        allowed: Mapping[str, Any],
        equity: float | None,
    ) -> ArenaAction:
        del equity  # Keep private-card estimates out of public table chat.
        action_name, amount = action
        templates = {
            "fold": (
                "the price and line make this a clean release",
                "this branch is too expensive to continue",
            ),
            "check": (
                "keeping the pot controlled on this texture",
                "taking the free card and preserving flexibility",
            ),
            "call": (
                "the price leaves enough room to continue",
                "continuing without inflating the pot",
            ),
            "bet": (
                "a measured size pressures the weaker range",
                "using a controlled size to deny cheap realization",
            ),
            "raise": (
                "applying pressure while keeping stack risk bounded",
                "this line supports a measured pressure raise",
            ),
            "all-in": (
                "stack geometry makes full commitment cleaner than a partial size",
                "the remaining stack works better as one decision",
            ),
        }
        table_id = str(table.get("tableId") or table.get("id") or "")
        digest = hashlib.sha256(f"{table_id}:{action_name}".encode()).digest()
        choices = templates[action_name]
        message = choices[digest[0] % len(choices)]

        reasoning: str | None = None
        if bool(allowed.get("reasoningRequired")):
            pot_odds = round(100 * self._pot_odds(table, allowed))
            fields = [f'ke: "pot odds {pot_odds}%"', 'pp: "risk-controlled line"']
            if action_name in _AGGRESSIVE_ACTIONS:
                fields.append('sr: "bounded pot pressure"')
            reasoning = "{" + ", ".join(fields) + "}"
        return ArenaAction(action_name, amount, message, reasoning)


_DEFAULT_POLICY: PlaygroundPolicy | None = None


def decide(
    table: Mapping[str, Any],
    deadline_s: float = 10.0,
    research_context: Mapping[str, Any] | None = None,
) -> dict[str, str | int]:
    """Drop-in strategy function; network registration/submission lives elsewhere."""

    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = PlaygroundPolicy()
    return _DEFAULT_POLICY.decide(table, deadline_s, research_context)


__all__ = [
    "ArenaAction",
    "ArenaSnapshotError",
    "PlaygroundPolicy",
    "decide",
    "features_from_table",
]
