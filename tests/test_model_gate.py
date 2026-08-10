"""table_sizes-gated model routing: trained checkpoints drive short-handed play."""

from __future__ import annotations

from devfun_poker_playground.contract import FEATURE_NAMES
from devfun_poker_playground.pure_model import PurePolicy
from devfun_poker_playground.snapshots import features_from_table


def _stub_weights(*, table_sizes: list[int] | None, favored: int) -> dict:
    """Zero network whose output bias forces one family."""

    b2 = [0.0, 0.0, 0.0]
    b2[favored] = 5.0
    document = {
        "input_size": len(FEATURE_NAMES),
        "hidden_size": 1,
        "feature_names": list(FEATURE_NAMES),
        "labels": ["fold", "check_call", "aggress"],
        "w1": [[0.0] * len(FEATURE_NAMES)],
        "b1": [0.0],
        "w2": [[0.0], [0.0], [0.0]],
        "b2": b2,
    }
    if table_sizes is not None:
        document["table_sizes"] = table_sizes
    return document


def _heads_up_table() -> dict:
    return {
        "id": "gate-test",
        "tableId": "gate-test",
        "street": "flop",
        "potChips": 400,
        "currentBet": 0,
        "boardCards": ["Qs", "7h", "3d"],
        "smallBlindChips": 50,
        "bigBlindChips": 100,
        "selfSeatNumber": 1,
        "seats": [
            {
                "seatNumber": 1,
                "status": "Active",
                "stackChips": 9800,
                "currentBetChips": 0,
                "holeCards": ["Qd", "Kh"],
            },
            {
                "seatNumber": 2,
                "status": "Active",
                "stackChips": 9800,
                "currentBetChips": 0,
                "holeCards": None,
            },
        ],
        "allowedActions": {
            "canFold": False,
            "canCheck": True,
            "canCall": False,
            "canBet": False,
            "canRaise": True,
            "canAllIn": False,
            "callAmount": 0,
            "callChips": 0,
            "callToAmount": 0,
            "minBet": None,
            "minRaiseTo": 100,
            "betRange": None,
            "raiseRange": {"min": 100, "max": 9800},
            "allInToAmount": None,
            "availableActions": ["check", "raise"],
            "amountSemantics": "toAmount",
            "reasoningRequired": False,
        },
        "recentEvents": [],
    }


def _family_for(policy: PurePolicy) -> str:
    table = _heads_up_table()
    features = features_from_table(table, position=0.0)
    allowed = table["allowedActions"]
    available = set(allowed["availableActions"])
    # Top pair, strong equity: the threshold path would choose "aggress".
    return policy._short_handed_family(
        table, allowed, available, equity=0.85, features=features
    )


def test_declared_table_size_routes_through_the_network() -> None:
    policy = PurePolicy(weights=_stub_weights(table_sizes=[2], favored=1), equity_trials=0)
    assert policy.table_sizes == frozenset({2})
    assert _family_for(policy) == "check_call"  # the biased net, not the thresholds


def test_undeclared_table_size_keeps_the_equity_thresholds() -> None:
    policy = PurePolicy(weights=_stub_weights(table_sizes=None, favored=1), equity_trials=0)
    assert policy.table_sizes == frozenset()
    assert _family_for(policy) == "aggress"  # 0.85 equity clears the floor


def test_other_table_sizes_still_use_thresholds_when_declared() -> None:
    policy = PurePolicy(weights=_stub_weights(table_sizes=[6], favored=0), equity_trials=0)
    assert _family_for(policy) == "aggress"  # heads-up not covered by [6]
