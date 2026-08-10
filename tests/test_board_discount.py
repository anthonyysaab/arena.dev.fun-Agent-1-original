"""Regression tests for the v3 board-contribution discount.

The facing-a-bet fixtures reconstruct real hands from rated chipzen matches
the bot lost by overvaluing holdings that barely improved a paired board
(matches 9f975ff1, f0422b76, abf3b14b, August 2026): trips-plus-kicker and
hollow two pair kept paying off ranges stuffed with boats. Those spots must
now fold, while genuine boats in the very same spots must keep continuing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

chipzen_models = pytest.importorskip("chipzen.models")
GameState = chipzen_models.GameState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "chipzen"))

import bot as chipzen_bot

from devfun_poker_playground.equity import board_improvement


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


@pytest.mark.parametrize(
    ("hole", "board", "tier"),
    [
        # The three real losses.
        (("Ac", "3c"), ("7s", "7c", "Ks", "7h"), "kicker"),
        (("Ac", "3c"), ("7s", "7c", "Ks", "7h", "5s"), "kicker"),
        (("4s", "Ks"), ("7h", "6s", "7s", "6d", "Kh"), "thin"),
        (("Ks", "Qc"), ("5s", "5c", "2s"), "kicker"),
        (("Ks", "Qc"), ("5s", "5c", "2s", "5d", "6h"), "kicker"),
        # No-pair hole cards on a paired flop add kickers at most.
        (("Ac", "3c"), ("7s", "7c", "Ks"), "kicker"),
        # Pocket pair below both board pairs plays the board plus a kicker.
        (("9h", "9d"), ("7h", "6s", "7s", "6d", "Kh"), "thin"),
        # Playing the board's straight is pure board strength.
        (("Ah", "Kd"), ("4s", "5d", "6h", "7c", "8d"), "kicker"),
        # Genuine upgrades stay fresh: boats, quads, trips with a hole card.
        (("Kd", "6d"), ("7s", "7c", "Ks", "7h"), "fresh"),
        (("7d", "8d"), ("7h", "6s", "7s", "6d", "Kh"), "fresh"),
        (("6c", "6d"), ("5s", "5c", "2s", "5d", "6h"), "fresh"),
        (("Ah", "2d"), ("5s", "5c", "2s", "5d", "6h"), "fresh"),
        (("Ah", "5h"), ("5s", "5c", "2s", "5d", "6h"), "fresh"),
        (("Kh", "Qd"), ("Kd", "Ks", "7s"), "fresh"),
        (("Ah", "Ad"), ("7s", "7c", "Ks"), "fresh"),
        (("Th", "9h"), ("4s", "5d", "6h", "7c", "8d"), "fresh"),
        # Unpaired boards and preflop are never discounted.
        (("Qd", "Kh"), ("Qs", "7h", "3d"), "fresh"),
        (("Ah", "Kd"), (), "fresh"),
    ],
)
def test_board_improvement_tiers(hole, board, tier) -> None:
    assert board_improvement(hole, board) == tier


_HAND_A_PREFLOP = [
    {"seat": 1, "action": "post_small_blind", "amount": 50, "phase": "preflop", "is_timeout": False},
    {"seat": 0, "action": "post_big_blind", "amount": 100, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
    {"seat": 0, "action": "raise", "amount": 200, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 200, "phase": "preflop", "is_timeout": False},
]


def test_trips_plus_kicker_folds_the_flop_raise() -> None:
    # Match 9f975ff1 hand 4: A3c on 7s 7c Ks. Our c-bet 200 got raised to
    # 590; the old bot called with board pair plus ace kicker and went on
    # to lose 5,378 by the river.
    history = [
        *_HAND_A_PREFLOP,
        {"seat": 0, "action": "raise", "amount": 200, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "raise", "amount": 590, "phase": "flop", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-777-flop",
        phase="flop",
        board=["7s", "7c", "Ks"],
        hole=["Ac", "3c"],
        pot=1190,
        stack=9170,
        opp_stack=9640,
        to_call=390,
        min_raise=980,
        max_raise=9370,
        history=history,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


_HAND_A_THROUGH_TURN_RAISE = [
    *_HAND_A_PREFLOP,
    {"seat": 0, "action": "raise", "amount": 200, "phase": "flop", "is_timeout": False},
    {"seat": 1, "action": "raise", "amount": 590, "phase": "flop", "is_timeout": False},
    {"seat": 0, "action": "call", "amount": 590, "phase": "flop", "is_timeout": False},
    {"seat": 0, "action": "raise", "amount": 790, "phase": "turn", "is_timeout": False},
    {"seat": 1, "action": "raise", "amount": 2330, "phase": "turn", "is_timeout": False},
]


def test_trips_plus_kicker_folds_the_turn_raise() -> None:
    # Match 9f975ff1 hand 4, the expensive street: board trips (7s 7c Ks 7h)
    # with only an ace kicker, our 790 bet raised to 2,330. The old bot
    # called 1,540 drawing nearly dead against K6's sevens full of kings.
    state = _state(
        round_id="regress-777-turn",
        phase="turn",
        board=["7s", "7c", "Ks", "7h"],
        hole=["Ac", "3c"],
        pot=4700,
        stack=7990,
        opp_stack=7310,
        to_call=1540,
        min_raise=3870,
        max_raise=8780,
        history=_HAND_A_THROUGH_TURN_RAISE,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


def test_genuine_boat_still_continues_the_turn_raise() -> None:
    # Same spot with the winning hand from that match (K6, sevens full of
    # kings): the discount must not touch genuine boats.
    state = _state(
        round_id="regress-777-turn-boat",
        phase="turn",
        board=["7s", "7c", "Ks", "7h"],
        hole=["Kd", "6d"],
        pot=4700,
        stack=7990,
        opp_stack=7310,
        to_call=1540,
        min_raise=3870,
        max_raise=8780,
        history=_HAND_A_THROUGH_TURN_RAISE,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action in {"call", "raise"}


_HAND_B_THROUGH_RIVER_LEAD = [
    {"seat": 0, "action": "post_small_blind", "amount": 50, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "post_big_blind", "amount": 100, "phase": "preflop", "is_timeout": False},
    {"seat": 0, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "check", "amount": 0, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
    {"seat": 0, "action": "raise", "amount": 100, "phase": "flop", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 100, "phase": "flop", "is_timeout": False},
    {"seat": 1, "action": "check", "amount": 0, "phase": "turn", "is_timeout": False},
    {"seat": 0, "action": "raise", "amount": 200, "phase": "turn", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 200, "phase": "turn", "is_timeout": False},
    {"seat": 1, "action": "raise", "amount": 800, "phase": "river", "is_timeout": False},
]


def test_hollow_two_pair_folds_the_big_river_lead() -> None:
    # Match f0422b76 hand 13 cards: K4s rivers kings-up on 7h 6s 7s 6d Kh —
    # a hand class the board already made on its own. Facing a pot-size
    # lead (the real hand value-bet into T6's sixes full and got paid),
    # kings-and-sevens with no real kicker must release.
    state = _state(
        round_id="regress-7766-river",
        phase="river",
        board=["7h", "6s", "7s", "6d", "Kh"],
        hole=["4s", "Ks"],
        pot=1600,
        stack=9600,
        opp_stack=8800,
        to_call=800,
        min_raise=1600,
        max_raise=9600,
        history=_HAND_B_THROUGH_RIVER_LEAD,
        your_seat=0,
        dealer_seat=0,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


def test_genuine_boat_still_continues_the_river_lead() -> None:
    # Same lead with sevens full: the discount must leave real boats alone.
    state = _state(
        round_id="regress-7766-river-boat",
        phase="river",
        board=["7h", "6s", "7s", "6d", "Kh"],
        hole=["7d", "8d"],
        pot=1600,
        stack=9600,
        opp_stack=8800,
        to_call=800,
        min_raise=1600,
        max_raise=9600,
        history=_HAND_B_THROUGH_RIVER_LEAD,
        your_seat=0,
        dealer_seat=0,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action in {"call", "raise"}


_HAND_C_PREFLOP = [
    {"seat": 1, "action": "post_small_blind", "amount": 50, "phase": "preflop", "is_timeout": False},
    {"seat": 0, "action": "post_big_blind", "amount": 100, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 100, "phase": "preflop", "is_timeout": False},
    {"seat": 0, "action": "raise", "amount": 200, "phase": "preflop", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 200, "phase": "preflop", "is_timeout": False},
]

_HAND_C_THROUGH_RIVER_BET = [
    *_HAND_C_PREFLOP,
    {"seat": 0, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
    {"seat": 1, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
    {"seat": 0, "action": "raise", "amount": 200, "phase": "turn", "is_timeout": False},
    {"seat": 1, "action": "call", "amount": 200, "phase": "turn", "is_timeout": False},
    {"seat": 0, "action": "check", "amount": 0, "phase": "river", "is_timeout": False},
    {"seat": 1, "action": "raise", "amount": 463, "phase": "river", "is_timeout": False},
]


def test_board_trips_plus_kicker_folds_the_river_bet() -> None:
    # Match abf3b14b hand 4: KQ on 5s 5c 2s 5d 6h. The board's trips plus
    # K-Q kickers lose to every deuce, six, five, pocket pair, and ace;
    # the old bot paid 42o's deuces full 463.
    state = _state(
        round_id="regress-555-river",
        phase="river",
        board=["5s", "5c", "2s", "5d", "6h"],
        hole=["Ks", "Qc"],
        pot=1263,
        stack=9195,
        opp_stack=9542,
        to_call=463,
        min_raise=926,
        max_raise=9195,
        history=_HAND_C_THROUGH_RIVER_BET,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action == "fold"


def test_genuine_boat_still_calls_the_river_bet() -> None:
    # Same river bet holding sixes full: still a clear continue.
    state = _state(
        round_id="regress-555-river-boat",
        phase="river",
        board=["5s", "5c", "2s", "5d", "6h"],
        hole=["6c", "6d"],
        pot=1263,
        stack=9195,
        opp_stack=9542,
        to_call=463,
        min_raise=926,
        max_raise=9195,
        history=_HAND_C_THROUGH_RIVER_BET,
        your_seat=0,
        dealer_seat=1,
        valid=["fold", "call", "raise"],
    )
    assert _bot().decide(state).action in {"call", "raise"}


def test_board_trips_plus_kicker_stops_barreling() -> None:
    # The same abf3b14b hand, one street earlier: the old bot bet the
    # board's trips (5s 5c 2s 5d with KQ) into the field, building the pot
    # it then paid off. Betting the board's own hand only levers out worse
    # board play, so v3 checks it back.
    history = [
        *_HAND_C_PREFLOP,
        {"seat": 0, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
        {"seat": 1, "action": "check", "amount": 0, "phase": "flop", "is_timeout": False},
    ]
    state = _state(
        round_id="regress-555-turn-bet",
        phase="turn",
        board=["5s", "5c", "2s", "5d"],
        hole=["Ks", "Qc"],
        pot=400,
        stack=9395,
        opp_stack=10205,
        to_call=0,
        min_raise=100,
        max_raise=9395,
        history=history,
        your_seat=0,
        dealer_seat=1,
        valid=["check", "raise"],
    )
    assert _bot().decide(state).action == "check"
