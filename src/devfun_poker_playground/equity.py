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


def estimate_equity(
    hero_hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    opponent_count: int,
    *,
    trials: int,
    seed: int,
) -> float:
    if len(hero_hole_cards) != 2:
        raise ValueError("hero_hole_cards must contain two cards")
    if len(board_cards) > 5:
        raise ValueError("board_cards cannot contain more than five cards")
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

    evaluator = _shared_evaluator()
    rng = Random(seed)
    equity = 0.0
    for _ in range(trials):
        sample = rng.sample(deck, cards_needed)
        cursor = 0
        opponents: list[tuple[int, int]] = []
        for _ in range(opponent_count):
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
