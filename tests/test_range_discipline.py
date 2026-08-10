"""Regression tests for the range-aware call discipline.

The facing-a-bet fixtures reconstruct real hands from rated chipzen matches
the bot lost by calling down or raise-warring with one pair (matches
e7205883 and cb6b65fd, August 2026). The bot must now fold or brake in
those spots while keeping its value bets and small-bet defends intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

chipzen_models = pytest.importorskip("chipzen.models")
GameState = chipzen_models.GameState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "chipzen"))

import bot as chipzen_bot

from devfun_poker_playground.equity import estimate_equity

_BLINDS_50_100 = [
    {"seat": 0, "action": "post_small_blind", "amount": 50, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "post_big_blind", "amount": 100, "phase": "preflop", "is_timeout": False},
]


def _bot() -> chipzen_bot.PlaygroundChipzenBot:
    instance = chipzen_bot.PlaygroundChipzenBot()
    instance.on_match_start(
        {
            "game_config": {
                "variant": "nlhe",
                "starting_stack": 10000,
                "small_blind": 50,
                "big_blind": 100,
                "ante": 0,
                "num_players": 2,
            },
            "turn_timeout_ms": 5000,
        }
    )
    return instance


def _state(
    *,
    round_id: str,
    phase: str,
    board: list[str],
    hole: list[str],
    pot: int,
    stack: int,
    opp_stack: int,
    to_call: int,
    min_raise: int,
    max_raise: int,
    history: list[dict],
    your_seat: int,
    dealer_seat: int,
    valid: list[str],
) -> GameState:
    return GameState.from_turn_request(
        {
            "type": "turn_request",
            "seat": your_seat,
            "round_id": round_id,
            "request_id": "q",
            "state": {
                "hand_number": 1,
                "phase": phase,
                "board": board,
                "your_hole_cards": hole,
                "pot": pot,
                "your_stack": stack,
                "opponent_stacks": [opp_stack],
                "to_call": to_call,
                "min_raise": min_raise,
                "max_raise": max_raise,
                "action_history": history,
            },
            "valid_actions": valid,
        },
        your_seat=your_seat,
        dealer_seat=dealer_seat,
    )


def test_conditioned_equity_is_far_below_random_equity() -> None:
    # Second pair (K3 on Kh 7d Ad) vs the top third of holdings.
    base = estimate_equity(("Ks", "3d"), ("Kh", "7d", "Ad"), 1, trials=600, seed=11)
    conditioned = estimate_equity(
        ("Ks", "3d"), ("Kh", "7d", "Ad"), 1, trials=600, seed=11, top_fraction=0.3
    )
    assert conditioned < base - 0.12


def test_top_fraction_one_matches_the_unconditioned_estimate() -> None:
    args = (("Ah", "Kd"), ("2c", "7h", "9s"), 1)
    a = estimate_equity(*args, trials=400, seed=3)
    b = estimate_equity(*args, trials=400, seed=3, top_fraction=1.0)
    assert a == b


def test_river_third_barrel_with_weak_pair_is_folded() -> None:
    # Match e7205883 hand 8: K4o on Qd 8d 7s Qs Tc, opponent barrels
    # 264 -> 612 -> 1420. The old bot called all three streets (-2496).
    history = [
        *_BLINDS_50_100,
        {"seat": 1, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
        {"seat": 0, "action": "raise", "amount": 200, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "call", "amount": 200, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 264, "phase": "flop", "is_timeout": False},
        {"seat": 0, "action": "call", "amount": 264, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 612, "phase": "turn", "is_timeout": False},
        {"seat": 0, "action": "call", "amount": 612, "phase": "turn", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 1420, "phase": "river", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-h8-river",
        phase="river",
        board=["Qd", "8d", "7s", "Qs", "Tc"],
        hole=["Ks", "4c"],
        pot=3572,
        stack=8924,
        opp_stack=6504,
        to_call=1420,
        min_raise=2840,
        max_raise=8924,
        history=history,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


def test_bottom_pair_never_calls_three_streets() -> None:
    # Match e7205883 hand 16: 73o on Ks Td 8s facing a flop bet. The old
    # bot called down three streets with bottom pair / 7-high (-1248).
    history = [
        *_BLINDS_50_100,
        {"seat": 1, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
        {"seat": 0, "action": "check", "amount": 0, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 132, "phase": "flop", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-h16-flop",
        phase="flop",
        board=["Ks", "Td", "8s"],
        hole=["7s", "3h"],
        pot=332,
        stack=9900,
        opp_stack=9768,
        to_call=132,
        min_raise=264,
        max_raise=9900,
        history=history,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


def test_second_pair_does_not_continue_the_raise_war() -> None:
    # Match cb6b65fd hand 3: K3o second pair on Kh 7d Ad. After our flop
    # raise the opponent re-raised to 3184; the old bot four-bet to 6368
    # and stacked off 10k against top pair.
    history = [
        *_BLINDS_50_100,
        {"seat": 0, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 300, "phase": "preflop", "is_timeout": False},
        {"seat": 0, "action": "call", "amount": 300, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 528, "phase": "flop", "is_timeout": False},
        {"seat": 0, "action": "raise", "amount": 1456, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 3184, "phase": "flop", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-raise-war",
        phase="flop",
        board=["Kh", "7d", "Ad"],
        hole=["Ks", "3d"],
        pot=5968,
        stack=8194,
        opp_stack=6516,
        to_call=1728,
        min_raise=4912,
        max_raise=9650,
        history=history,
        your_seat=0,
        dealer_seat=0,
        valid=["fold", "call", "raise"],
    )
    action = _bot().decide(state)
    assert action.action in {"fold", "call"}  # never re-raise the war


def test_huge_turn_bet_with_weak_pair_is_folded_not_stacked_off() -> None:
    # Continuation of the raise-war hand: on the turn the opponent bets
    # 2808 into 7696, ~43% of our remaining stack. The old bot called off.
    history = [
        *_BLINDS_50_100,
        {"seat": 0, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 300, "phase": "preflop", "is_timeout": False},
        {"seat": 0, "action": "call", "amount": 300, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 528, "phase": "flop", "is_timeout": False},
        {"seat": 0, "action": "raise", "amount": 1456, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 3184, "phase": "flop", "is_timeout": False},
        {"seat": 0, "action": "call", "amount": 3184, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 2808, "phase": "turn", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-turn-shove",
        phase="turn",
        board=["Kh", "7d", "Ad", "Tc"],
        hole=["Ks", "3d"],
        pot=10504,
        stack=6466,
        opp_stack=3708,
        to_call=2808,
        min_raise=5616,
        max_raise=6466,
        history=history,
        your_seat=0,
        dealer_seat=0,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


def test_value_hands_still_bet_when_checked_to() -> None:
    # Match e7205883 hand 5: rivered trips (Tc4d on 5c Ts Th Jc 6c) with
    # the opponent checking. Initiative spots keep the unconditioned
    # equity, so the bot must still bet for value.
    history = [
        *_BLINDS_50_100,
        {"seat": 0, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "check", "amount": 0, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
        {"seat": 0, "action": "raise", "amount": 100, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "call", "amount": 100, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "check", "amount": 0, "phase": "turn", "is_timeout": False},
        {"seat": 0, "action": "raise", "amount": 200, "phase": "turn", "is_timeout": False},
        {"seat": 1, "action": "call", "amount": 200, "phase": "turn", "is_timeout": False},
        {"seat": 1, "action": "check", "amount": 0, "phase": "river", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-value-river",
        phase="river",
        board=["5c", "Ts", "Th", "Jc", "6c"],
        hole=["Tc", "4d"],
        pot=800,
        stack=8624,
        opp_stack=10176,
        to_call=0,
        min_raise=100,
        max_raise=8624,
        history=history,
        your_seat=0,
        dealer_seat=0,
        valid=["check", "raise"],
    )
    action = _bot().decide(state)
    assert action.action == "raise"
    assert action.amount >= 100


def test_top_pair_still_defends_against_a_small_bet() -> None:
    # Guard against over-folding: top pair, good kicker facing a one-third
    # pot bet is still a clear call.
    history = [
        *_BLINDS_50_100,
        {"seat": 1, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
        {"seat": 0, "action": "check", "amount": 0, "phase": "preflop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 66, "phase": "flop", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-small-bet-defend",
        phase="flop",
        board=["Qs", "7h", "3d"],
        hole=["Qd", "Kh"],
        pot=266,
        stack=9900,
        opp_stack=9834,
        to_call=66,
        min_raise=132,
        max_raise=9900,
        history=history,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    action = _bot().decide(state)
    assert action.action in {"call", "raise"}
