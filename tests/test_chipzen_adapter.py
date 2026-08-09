"""Chipzen adapter tests built on the protocol spec's worked hand example.

The turn_request fixtures reproduce Messages 4, 11, and 17 from
POKER-GAME-STATE-PROTOCOL.md section 4, so the mapping is exercised against
the platform's own documented payloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

chipzen_models = pytest.importorskip("chipzen.models")
GameState = chipzen_models.GameState
Action = chipzen_models.Action

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "chipzen"))

import bot as chipzen_bot

from devfun_poker_playground.snapshots import features_from_table

_BLINDS = [
    {"seat": 0, "action": "post_small_blind", "amount": 5, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "post_big_blind", "amount": 10, "phase": "preflop", "is_timeout": False},
]


def _message_4() -> GameState:
    return GameState.from_turn_request(
        {
            "type": "turn_request",
            "match_id": "m_abc123",
            "seq": 4,
            "seat": 0,
            "round_id": "r_550e8400",
            "request_id": "q_1",
            "timeout_ms": 10000,
            "state": {
                "hand_number": 1,
                "phase": "preflop",
                "board": [],
                "your_hole_cards": ["As", "Kh"],
                "pot": 15,
                "your_stack": 995,
                "opponent_stacks": [990],
                "to_call": 5,
                "min_raise": 20,
                "max_raise": 995,
                "action_history": list(_BLINDS),
            },
            "valid_actions": ["fold", "call", "raise"],
        },
        your_seat=0,
        dealer_seat=0,
    )


def _message_11() -> GameState:
    history = [
        *_BLINDS,
        {"seat": 0, "action": "raise", "amount": 30, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "call", "amount": 30, "phase": "preflop", "is_timeout": False},
    ]
    return GameState.from_turn_request(
        {
            "type": "turn_request",
            "seat": 1,
            "round_id": "r_550e8400",
            "request_id": "q_2",
            "state": {
                "hand_number": 1,
                "phase": "flop",
                "board": ["Qs", "7h", "3d"],
                "your_hole_cards": ["Jd", "Tc"],
                "pot": 60,
                "your_stack": 970,
                "opponent_stacks": [970],
                "to_call": 0,
                "min_raise": 10,
                "max_raise": 970,
                "action_history": history,
            },
            "valid_actions": ["check", "raise"],
        },
        your_seat=1,
        dealer_seat=0,
    )


def _message_17() -> GameState:
    history = [
        *_BLINDS,
        {"seat": 0, "action": "raise", "amount": 30, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "call", "amount": 30, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
        {"seat": 0, "action": "raise", "amount": 40, "phase": "flop", "is_timeout": False},
    ]
    return GameState.from_turn_request(
        {
            "type": "turn_request",
            "seat": 1,
            "round_id": "r_550e8400",
            "request_id": "q_3",
            "state": {
                "hand_number": 1,
                "phase": "flop",
                "board": ["Qs", "7h", "3d"],
                "your_hole_cards": ["Jd", "Tc"],
                "pot": 100,
                "your_stack": 970,
                "opponent_stacks": [930],
                "to_call": 40,
                "min_raise": 80,
                "max_raise": 970,
                "action_history": history,
            },
            "valid_actions": ["fold", "call", "raise"],
        },
        your_seat=1,
        dealer_seat=0,
    )


def _bot() -> chipzen_bot.PlaygroundChipzenBot:
    instance = chipzen_bot.PlaygroundChipzenBot()
    instance.on_match_start(
        {
            "game_config": {
                "variant": "nlhe",
                "starting_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
                "ante": 0,
                "num_players": 2,
            },
            "turn_timeout_ms": 5000,
        }
    )
    return instance


def test_snapshot_mapping_matches_the_worked_example() -> None:
    state = _message_4()
    table, position = chipzen_bot.snapshot_from_state(state, small_blind=5, big_blind=10)

    assert position == 1.0  # heads-up button/small blind
    assert table["selfSeatNumber"] == 1
    assert table["potChips"] == 15
    assert table["currentBet"] == 10  # SB posted 5, owes 5 more to match the BB
    allowed = table["allowedActions"]
    assert allowed["callChips"] == 5
    assert allowed["callToAmount"] == 10
    assert allowed["raiseRange"] == {"min": 20, "max": 995}
    assert allowed["availableActions"] == ["call", "fold", "raise"]
    hero = table["seats"][0]
    assert hero["currentBetChips"] == 5
    assert hero["holeCards"] == ["As", "Kh"]

    features = features_from_table(table, position=position)
    assert len(features) == 125


def test_snapshot_mapping_derives_flop_street_state() -> None:
    state = _message_17()
    table, position = chipzen_bot.snapshot_from_state(state, small_blind=5, big_blind=10)

    assert position == 0.0  # heads-up big blind
    hero = table["seats"][1]
    assert hero["currentBetChips"] == 0  # checked the flop, nothing committed yet
    assert table["currentBet"] == 40
    assert table["allowedActions"]["callToAmount"] == 40
    opponent = table["seats"][0]
    assert opponent["currentBetChips"] == 40


def test_decisions_on_worked_example_are_legal() -> None:
    instance = _bot()
    for state in (_message_4(), _message_11(), _message_17()):
        action = instance.decide(state)
        assert isinstance(action, Action)
        assert action.action in set(state.valid_actions)
        if action.action == "raise":
            assert state.min_raise <= action.amount <= state.max_raise


def test_decides_are_deterministic_per_hand() -> None:
    instance = _bot()
    first = instance.decide(_message_17())
    second = instance.decide(_message_17())
    assert first == second


def test_free_option_is_never_folded() -> None:
    instance = _bot()
    action = instance.decide(_message_11())
    assert action.action in {"check", "raise"}


def test_broken_state_falls_back_to_a_safe_action() -> None:
    instance = _bot()
    state = _message_11()
    state.board = state.hole_cards + state.board[:1]  # duplicate cards: invalid snapshot
    action = instance.decide(state)
    assert action.action == "check"  # cheapest legal action for check/raise turns


def test_raise_payloads_are_clamped_to_the_legal_range() -> None:
    state = _message_4()
    action = chipzen_bot._payload_to_action({"action": "raise", "amount": 10_000}, state)
    assert action.action == "raise"
    assert action.amount == state.max_raise
    action = chipzen_bot._payload_to_action({"action": "raise", "amount": 1}, state)
    assert action.amount == state.min_raise


def test_zero_min_raise_disables_aggression_mapping() -> None:
    state = _message_11()
    state.min_raise = 0
    state.max_raise = 0
    table, _ = chipzen_bot.snapshot_from_state(state, small_blind=5, big_blind=10)
    allowed = table["allowedActions"]
    assert allowed["canRaise"] is False
    assert "raise" not in allowed["availableActions"]
    assert allowed["raiseRange"] is None


def test_multiway_positions_follow_the_protocol_derivation() -> None:
    # Six-handed: offset 1 from the button is the small blind (position 0.0),
    # offset 2 the big blind, and the button closes at 1.0.
    assert chipzen_bot.table_position(3, 2, 6) == 0.0
    assert chipzen_bot.table_position(4, 2, 6) == pytest.approx(0.2)
    assert chipzen_bot.table_position(2, 2, 6) == 1.0
    # Heads-up: button/SB is 1.0, big blind is 0.0.
    assert chipzen_bot.table_position(0, 0, 2) == 1.0
    assert chipzen_bot.table_position(1, 0, 2) == 0.0
