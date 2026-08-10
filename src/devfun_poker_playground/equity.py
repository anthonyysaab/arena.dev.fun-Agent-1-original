"""Small Monte Carlo equity helper used only as a Playground safety fallback.

Uses the vendored, MIT-licensed pure-Python ``treys`` evaluator (see
``_vendor/treys/LICENSE``) so deployment images need no third-party wheels.
The evaluator's lookup tables are built once per process and reused.
"""

from __future__ import annotations

from random import Random

from devfun_poker_playground._vendor.treys import Card, Deck, Evaluator

_EVALUATOR: Evaluator | None = None
_FULL_DECK: tuple[int, ...] | None = None


def _shared_evaluator() -> Evaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = Evaluator()
    return _EVALUATOR


def _full_deck() -> tuple[int, ...]:
    global _FULL_DECK
    if _FULL_DECK is None:
        _FULL_DECK = tuple(Deck.GetFullDeck())
    return _FULL_DECK


def prewarm() -> None:
    """Build the card lookup tables ahead of the first timed decision."""

    _shared_evaluator()
    _full_deck()


def _treys_card(value: str) -> int:
    return Card.new(f"{value[0].upper()}{value[1].lower()}")


def _chen_score(combo: tuple[int, int]) -> float:
    """Chen-formula preflop strength; higher is stronger."""

    rank_a = Card.get_rank_int(combo[0])
    rank_b = Card.get_rank_int(combo[1])
    high, low = (rank_a, rank_b) if rank_a >= rank_b else (rank_b, rank_a)
    points = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(high, (high + 2) / 2.0)
    if rank_a == rank_b:
        return max(5.0, points * 2.0)
    if Card.get_suit_int(combo[0]) == Card.get_suit_int(combo[1]):
        points += 2.0
    gap = high - low - 1
    points -= (0.0, 1.0, 2.0, 4.0)[gap] if gap <= 3 else 5.0
    if gap <= 1 and high <= 9:  # connected low cards can still make straights
        points += 1.0
    return points


def _restricted_combos(
    deck: list[int],
    board: tuple[int, ...],
    top_fraction: float,
) -> list[tuple[int, int]]:
    """Opponent hole combos ranked by strength, strongest ``top_fraction`` kept.

    Postflop, strength is the made-hand rank on the current board — a crude
    but effective proxy for the hands an opponent bets and raises with.
    Preflop it falls back to the Chen formula. This deliberately ignores
    draws; big aggression weighted toward made hands is the point.
    """

    combos = [
        (deck[i], deck[j])
        for i in range(len(deck))
        for j in range(i + 1, len(deck))
    ]
    if board:
        evaluator = _shared_evaluator()
        board_list = list(board)
        combos.sort(key=lambda combo: evaluator.evaluate(board_list, list(combo)))
    else:
        combos.sort(key=_chen_score, reverse=True)
    keep = max(1, round(top_fraction * len(combos)))
    return combos[:keep]


def estimate_equity(
    hero_hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    opponent_count: int,
    *,
    trials: int,
    seed: int,
    top_fraction: float = 1.0,
) -> float:
    """Monte Carlo equity for the hero hand.

    ``top_fraction`` conditions ONE opponent (the aggressor) on the strongest
    fraction of possible holdings instead of a uniformly random hand; any
    remaining opponents stay random. 1.0 reproduces the unconditioned
    estimate exactly.
    """

    if len(hero_hole_cards) != 2:
        raise ValueError("hero_hole_cards must contain two cards")
    if len(board_cards) > 5:
        raise ValueError("board_cards cannot contain more than five cards")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if opponent_count < 1:
        return 1.0
    if trials < 1:
        raise ValueError("trials must be positive")

    hero = tuple(_treys_card(card) for card in hero_hole_cards)
    board = tuple(_treys_card(card) for card in board_cards)
    known = {*hero, *board}
    if len(known) != len(hero) + len(board):
        raise ValueError("known cards must be unique")

    deck = [card for card in _full_deck() if card not in known]
    missing_board = 5 - len(board)
    cards_needed = 2 * opponent_count + missing_board
    if cards_needed > len(deck):
        raise ValueError("not enough cards remain to simulate the table")

    restricted = (
        _restricted_combos(deck, board, top_fraction) if top_fraction < 1.0 else None
    )

    evaluator = _shared_evaluator()
    rng = Random(seed)
    equity = 0.0
    for _ in range(trials):
        opponents: list[tuple[int, int]] = []
        if restricted is None:
            sample = rng.sample(deck, cards_needed)
            cursor = 0
            for _ in range(opponent_count):
                opponents.append((sample[cursor], sample[cursor + 1]))
                cursor += 2
            runout = [*board, *sample[cursor:]]
        else:
            aggressor = restricted[rng.randrange(len(restricted))]
            opponents.append(aggressor)
            rest = [card for card in deck if card not in aggressor]
            sample = rng.sample(rest, cards_needed - 2)
            cursor = 0
            for _ in range(opponent_count - 1):
                opponents.append((sample[cursor], sample[cursor + 1]))
                cursor += 2
            runout = [*board, *sample[cursor:]]

        hero_score = evaluator.evaluate(runout, list(hero))
        opponent_scores = [evaluator.evaluate(runout, list(hole)) for hole in opponents]
        best_score = min(hero_score, *opponent_scores)
        if hero_score == best_score:
            equity += 1.0 / (1 + sum(score == best_score for score in opponent_scores))
    return equity / trials


__all__ = ["estimate_equity", "prewarm"]
