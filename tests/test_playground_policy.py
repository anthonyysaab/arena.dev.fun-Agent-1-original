from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("treys")

contract_module = pytest.importorskip("devfun_poker_playground.model_contract")
arena_module = pytest.importorskip("devfun_poker_playground.playground")
FEATURE_NAMES = contract_module.FEATURE_NAMES
LABELS = contract_module.LABELS
ArenaSnapshotError = arena_module.ArenaSnapshotError
PlaygroundPolicy = arena_module.PlaygroundPolicy
features_from_table = arena_module.features_from_table


class FixedModel(torch.nn.Module):
    def __init__(self, fold: float, check_call: float, aggress: float) -> None:
        super().__init__()
        self.register_buffer("scores", torch.tensor([fold, check_call, aggress], dtype=torch.float32))

    def forward(self, features):
        return self.scores.repeat(features.shape[0], 1)


def _seat(number: int, *, hero: bool = False) -> dict[str, object]:
    return {
        "seatId": f"seat-{number}",
        "seatNumber": number,
        "agentId": "hero" if hero else f"villain-{number}",
        "agentName": "Hero" if hero else f"Villain {number}",
        "agentHandle": "hero" if hero else f"v{number}",
        "status": "Active",
        "stackChips": 990,
        "currentBetChips": 10 if hero else (30 if number == 5 else 0),
        "totalCommittedChips": 10 if hero else (30 if number == 5 else 0),
        "payoutChips": None,
        "holeCards": ["As", "Ad"] if hero else None,
    }


def _table() -> dict[str, object]:
    seats = [_seat(5), _seat(2), _seat(4, hero=True), _seat(1), _seat(6), _seat(3)]
    return {
        "id": "table-1",
        "tableId": "table-1",
        "tableNumber": 1,
        "competitionId": "playground-test",
        "status": "Active",
        "street": "Preflop",
        "potChips": 80,
        "currentBet": 30,
        "minRaiseTo": 50,
        "actionDeadlineAt": 9999999999999,
        "currentSeatNumber": 4,
        "boardCards": [],
        "smallBlindChips": 5,
        "bigBlindChips": 10,
        "buyInChips": 1000,
        "winners": [],
        "seats": seats,
        "actingSeatNumber": 4,
        "selfSeatNumber": 4,
        "allowedActions": {
            "canFold": True,
            "canCheck": False,
            "canCall": True,
            "canBet": False,
            "canRaise": True,
            "callAmount": 20,
            "callChips": 20,
            "callToAmount": 30,
            "minBet": None,
            "minRaiseTo": 50,
            "maxCommit": 1000,
            "allInToAmount": 1000,
            "betRange": None,
            "raiseRange": {"min": 50, "max": 1000},
            "canAllIn": True,
            "availableActions": ["fold", "call", "raise", "all-in"],
            "amountSemantics": "toAmount",
            "amountHint": "raise amount is total street commitment",
            "reasoningRequired": True,
            "actionHint": "fold, call 20, raise 50-1000, or all-in",
        },
        "recentEvents": [
            {
                "id": "blind-1",
                "sequence": 1,
                "type": "BlindPosted",
                "street": "Preflop",
                "occurredAt": 1,
                "summary": {"seatNumber": 2, "amount": 5, "action": None},
            },
            {
                "id": "blind-2",
                "sequence": 2,
                "type": "BlindPosted",
                "street": "Preflop",
                "occurredAt": 2,
                "summary": {"seatNumber": 3, "amount": 10, "action": None},
            },
            {
                "id": "raise-1",
                "sequence": 3,
                "type": "ActionTaken",
                "street": "Preflop",
                "occurredAt": 3,
                "summary": {"seatNumber": 5, "amount": 30, "action": "raise"},
            },
        ],
    }


def _feature(features: tuple[float, ...], name: str) -> float:
    return features[FEATURE_NAMES.index(name)]


def test_arena_snapshot_maps_to_training_features() -> None:
    features = features_from_table(_table())

    assert len(features) == len(FEATURE_NAMES)
    assert _feature(features, "hole_As") == 1
    assert _feature(features, "hole_Ad") == 1
    assert _feature(features, "player_count") == 6
    assert _feature(features, "active_player_count") == 6
    assert _feature(features, "position") == pytest.approx(0.4)
    assert _feature(features, "raises_current_street") == 1
    assert _feature(features, "legal_check_call") == 1
    assert _feature(features, "legal_aggress") == 1


def test_aggressive_proposal_uses_a_bounded_legal_raise_to_amount() -> None:
    policy = PlaygroundPolicy(
        model=FixedModel(fold=0, check_call=0, aggress=10),
        equity_trials=0,
    )

    decision = policy.decide(_table())

    assert decision["action"] == "raise"
    assert 50 <= decision["amount"] <= 356
    assert decision["amount"] != 1000
    assert decision["message"]
    assert len(decision["reasoning"]) <= 150


def test_illegal_aggression_is_masked_to_the_available_family() -> None:
    table = _table()
    allowed = table["allowedActions"]
    allowed.update(
        {
            "canRaise": False,
            "canAllIn": False,
            "raiseRange": None,
            "allInToAmount": None,
            "availableActions": ["fold", "call"],
        }
    )
    policy = PlaygroundPolicy(
        model=FixedModel(fold=0, check_call=5, aggress=10),
        equity_trials=0,
    )

    decision = policy.decide(table)

    assert decision["action"] == "call"
    assert "amount" not in decision


def test_free_option_replaces_a_fold_proposal() -> None:
    table = _table()
    table["currentBet"] = 0
    hero = next(seat for seat in table["seats"] if seat["seatNumber"] == 4)
    hero["currentBetChips"] = 0
    allowed = table["allowedActions"]
    allowed.update(
        {
            "canFold": True,
            "canCheck": True,
            "canCall": False,
            "canBet": True,
            "canRaise": False,
            "callAmount": 0,
            "callChips": 0,
            "callToAmount": None,
            "minBet": 10,
            "minRaiseTo": None,
            "betRange": {"min": 10, "max": 990},
            "raiseRange": None,
            "availableActions": ["fold", "check", "bet", "all-in"],
        }
    )
    policy = PlaygroundPolicy(
        model=FixedModel(fold=10, check_call=0, aggress=0),
        equity_trials=0,
    )

    decision = policy.decide(table)

    assert decision["action"] == "check"


def test_deadline_fallback_folds_instead_of_calling_a_large_amount() -> None:
    table = _table()
    table["allowedActions"]["callChips"] = 100
    policy = PlaygroundPolicy(
        model=FixedModel(fold=0, check_call=10, aggress=0),
        equity_trials=0,
    )

    decision = policy.decide(table, deadline_s=1.0)

    assert decision["action"] == "fold"


def test_warm_start_never_selects_an_optional_all_in() -> None:
    table = _table()
    allowed = table["allowedActions"]
    allowed.update(
        {
            "canRaise": False,
            "raiseRange": None,
            "availableActions": ["fold", "call", "all-in"],
        }
    )
    policy = PlaygroundPolicy(
        model=FixedModel(fold=0, check_call=0, aggress=10),
        equity_trials=0,
    )

    decision = policy.decide(table)

    assert decision["action"] == "call"


def test_duplicate_visible_card_is_rejected() -> None:
    table = deepcopy(_table())
    table["boardCards"] = ["As", "7d", "2c"]
    table["street"] = "Flop"

    with pytest.raises(ArenaSnapshotError, match="unique"):
        features_from_table(table)


def test_saved_checkpoint_can_make_a_legal_playground_decision() -> None:
    policy = PlaygroundPolicy(equity_trials=5)

    decision = policy.decide(_table())

    assert decision["action"] in _table()["allowedActions"]["availableActions"]
    assert 1 <= len(decision["message"]) <= 500
