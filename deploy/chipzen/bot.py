"""Chipzen adapter for the dev.fun Poker Playground policy.

Bridges the Chipzen SDK's ``GameState`` to the snapshot contract the
Playground policy validates, runs the torch-free :class:`PurePolicy`
(exported tiny-policy weights + the shared deterministic safety rails), and
maps the resulting payload back to a legal SDK :class:`Action`.

Wire semantics line up cleanly: Chipzen raises are raise-TO totals, exactly
the Playground's ``toAmount`` convention. Chipzen has no separate ``bet`` or
``all-in`` wire action — ``raise`` covers both — so aggressive payloads are
clamped into ``[min_raise, max_raise]`` and submitted as ``raise``.

Environment (per DEV-MANUAL section 7.1):
    CHIPZEN_WS_URL   WebSocket URL injected by the platform at launch.
    CHIPZEN_TOKEN    Bot API token (empty is fine for localhost dev).
    CHIPZEN_TICKET   Alternative single-use ticket; forwarded when present.

Tuning:
    POKER_PURE_WEIGHTS        Exported weights JSON (baked in by the image).
    PLAYGROUND_EQUITY_TRIALS  Monte Carlo trials per decision (default 512).
    PLAYGROUND_LOG_LEVEL      Python log level name (default INFO).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from chipzen import Action, Bot, GameState
from chipzen.client import run_bot

from devfun_poker_playground.equity import prewarm
from devfun_poker_playground.pure_model import PurePolicy

logger = logging.getLogger("playground-chipzen-bot")

_DEFAULT_EQUITY_TRIALS = 512
_DEFAULT_TIMEOUT_MS = 5000

# action_history amounts for these entries are street raise-to levels, so a
# per-seat max reconstructs each seat's current-street commitment.
_STREET_LEVEL_ACTIONS = {"post_small_blind", "post_big_blind", "raise", "call"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return default
    return max(minimum, value)


def _history_entries(state: GameState) -> list[dict]:
    return [entry for entry in (state.action_history or []) if isinstance(entry, dict)]


def _street_commitments(state: GameState, phase: str) -> dict[int, int]:
    """Per-seat chips committed on the current street, as raise-to levels."""

    commitments: dict[int, int] = {}
    for entry in _history_entries(state):
        if str(entry.get("phase") or "") != phase:
            continue
        seat = entry.get("seat")
        amount = entry.get("amount")
        action = str(entry.get("action") or "")
        if not isinstance(seat, int) or isinstance(amount, bool):
            continue
        if not isinstance(amount, (int, float)) or amount < 0:
            continue
        if action in _STREET_LEVEL_ACTIONS:
            commitments[seat] = max(commitments.get(seat, 0), int(amount))
    return commitments


def _folded_seats(state: GameState) -> set[int]:
    folded: set[int] = set()
    for entry in _history_entries(state):
        if str(entry.get("action") or "") == "fold" and isinstance(entry.get("seat"), int):
            folded.add(entry["seat"])
    return folded


def table_position(your_seat: int, dealer_seat: int, num_players: int) -> float:
    """Playground position feature (0 = small blind ... 1 = button).

    Derived per POKER-GAME-STATE-PROTOCOL.md section 5.9: offset 0 is the
    button; heads-up the button posts the small blind. The Playground orders
    seats [SB, BB, ..., button] and scales the index into [0, 1], which for
    heads-up gives BB=0.0 and button/SB=1.0.
    """

    if num_players <= 1:
        return 0.0
    seats_after_button = (your_seat - dealer_seat) % num_players
    ordered_index = (seats_after_button - 1) % num_players
    return ordered_index / (num_players - 1)


def snapshot_from_state(
    state: GameState,
    *,
    small_blind: int,
    big_blind: int,
) -> tuple[dict, float]:
    """Build the Playground table snapshot plus the position feature."""

    num_players = len(state.opponent_stacks) + 1
    hero_seat = int(state.your_seat)
    phase = str(state.phase or "").casefold()
    to_call = max(0, int(state.to_call))
    commitments = _street_commitments(state, phase)
    folded = _folded_seats(state)
    hero_commitment = commitments.get(hero_seat, 0)

    valid = {str(action) for action in (state.valid_actions or [])}
    min_raise = int(state.min_raise)
    max_raise = int(state.max_raise)
    can_raise = "raise" in valid and 1 <= min_raise <= max_raise
    available = sorted(valid & {"fold", "check", "call"})
    if can_raise:
        available.append("raise")

    seats = []
    opponent_stacks = iter(state.opponent_stacks)
    for seat in range(num_players):
        is_hero = seat == hero_seat
        stack = int(state.your_stack) if is_hero else int(next(opponent_stacks))
        seats.append(
            {
                "seatNumber": seat + 1,
                "status": "Folded" if seat in folded else "Active",
                "stackChips": max(0, stack),
                "currentBetChips": commitments.get(seat, 0),
                "holeCards": [str(card) for card in state.hole_cards] if is_hero else None,
            }
        )

    allowed = {
        "canFold": "fold" in valid,
        "canCheck": "check" in valid,
        "canCall": "call" in valid,
        "canBet": False,
        "canRaise": can_raise,
        "canAllIn": False,
        "callAmount": to_call,
        "callChips": to_call,
        "callToAmount": hero_commitment + to_call,
        "minBet": None,
        "minRaiseTo": min_raise if can_raise else None,
        "betRange": None,
        "raiseRange": {"min": min_raise, "max": max_raise} if can_raise else None,
        "allInToAmount": None,
        "availableActions": available,
        "amountSemantics": "toAmount",
        "reasoningRequired": False,
    }

    table_id = state.round_id or f"hand-{state.hand_number}"
    table = {
        "id": table_id,
        "tableId": table_id,
        "street": phase,
        "potChips": max(0, int(state.pot)),
        "currentBet": hero_commitment + to_call,
        "boardCards": [str(card) for card in state.board],
        "smallBlindChips": max(1, int(small_blind)),
        "bigBlindChips": max(1, int(big_blind)),
        "seats": seats,
        "selfSeatNumber": hero_seat + 1,
        "allowedActions": allowed,
        "recentEvents": [
            {
                "type": "ActionTaken",
                "street": str(entry.get("phase") or ""),
                "summary": {
                    "action": str(entry.get("action") or ""),
                    "seatNumber": int(entry["seat"]) + 1
                    if isinstance(entry.get("seat"), int)
                    else None,
                    "amount": entry.get("amount"),
                },
            }
            for entry in _history_entries(state)
        ],
    }
    return table, table_position(hero_seat, int(state.dealer_seat), num_players)


def _safe_action(state: GameState) -> Action:
    """Cheapest legal action when the policy path cannot be trusted."""

    valid = set(state.valid_actions or [])
    if "check" in valid:
        return Action.check()
    if "fold" in valid:
        return Action.fold()
    if "call" in valid:
        return Action.call()
    if "raise" in valid:
        return Action.raise_to(max(1, int(state.min_raise)))
    return Action.fold()


def _payload_to_action(payload: dict, state: GameState) -> Action:
    valid = set(state.valid_actions or [])
    name = str(payload.get("action"))
    if name == "check" and "check" in valid:
        return Action.check()
    if name == "call" and "call" in valid:
        return Action.call()
    if name == "fold" and "fold" in valid:
        return Action.fold()
    if name in ("bet", "raise", "all-in") and "raise" in valid:
        amount = payload.get("amount", state.min_raise)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            amount = state.min_raise
        clamped = max(int(state.min_raise), min(int(amount), int(state.max_raise)))
        return Action.raise_to(clamped)
    logger.warning("payload action %r not mappable onto %s; using safe action", name, sorted(valid))
    return _safe_action(state)


class PlaygroundChipzenBot(Bot):
    """dev.fun Poker Playground policy behind the Chipzen Bot interface."""

    def __init__(self) -> None:
        super().__init__()
        trials = _env_int("PLAYGROUND_EQUITY_TRIALS", _DEFAULT_EQUITY_TRIALS)
        self._policy = PurePolicy(equity_trials=trials)
        self._small_blind: int | None = None
        self._big_blind: int | None = None
        self._turn_timeout_ms = _DEFAULT_TIMEOUT_MS
        # Build the card lookup tables now, off the decision clock.
        prewarm()

    def on_match_start(self, match_info: dict) -> None:
        config = match_info.get("game_config") or {}
        small_blind = config.get("small_blind")
        big_blind = config.get("big_blind")
        if isinstance(small_blind, (int, float)) and not isinstance(small_blind, bool):
            self._small_blind = max(1, int(small_blind))
        if isinstance(big_blind, (int, float)) and not isinstance(big_blind, bool):
            self._big_blind = max(1, int(big_blind))
        timeout = match_info.get("turn_timeout_ms")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
            self._turn_timeout_ms = int(timeout)
        logger.info(
            "match_start blinds=%s/%s players=%s timeout_ms=%s",
            self._small_blind,
            self._big_blind,
            config.get("num_players"),
            self._turn_timeout_ms,
        )

    def _blinds(self, state: GameState) -> tuple[int, int]:
        if self._big_blind is not None:
            small = self._small_blind or max(1, self._big_blind // 2)
            return small, self._big_blind
        # Fallback: read the synthetic blind posts from the hand history.
        small_blind: int | None = None
        big_blind: int | None = None
        for entry in _history_entries(state):
            action = str(entry.get("action") or "")
            amount = entry.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
                continue
            if action == "post_small_blind":
                small_blind = int(amount)
            elif action == "post_big_blind":
                big_blind = int(amount)
        if big_blind is None:
            big_blind = 2 * small_blind if small_blind else 10
        if small_blind is None:
            small_blind = max(1, big_blind // 2)
        return small_blind, big_blind

    def decide(self, state: GameState) -> Action:
        try:
            small_blind, big_blind = self._blinds(state)
            table, position = snapshot_from_state(
                state, small_blind=small_blind, big_blind=big_blind
            )
            payload = self._policy.decide(
                table,
                deadline_s=self._turn_timeout_ms / 1000.0,
                research_context={"position": position},
            )
            return _payload_to_action(payload, state)
        except Exception:
            logger.exception("decide failed; falling back to the safe action")
            return _safe_action(state)

    def on_decision_latency(self, latency_ms: float) -> None:
        if latency_ms > 1000:
            logger.warning("slow turn: %.0fms", latency_ms)


def main() -> None:
    """Entry point — invoked by the Dockerfile ENTRYPOINT."""

    log_level = os.environ.get("PLAYGROUND_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    url = os.environ.get("CHIPZEN_WS_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not url:
        print(
            "error: CHIPZEN_WS_URL not set and no URL passed on the command line",
            file=sys.stderr,
        )
        sys.exit(2)

    ticket = os.environ.get("CHIPZEN_TICKET")
    logger.info("playground bot ready; connecting")
    asyncio.run(
        run_bot(
            url,
            PlaygroundChipzenBot(),
            token=os.environ.get("CHIPZEN_TOKEN"),
            ticket=ticket if ticket else None,
            client_name="devfun-playground",
            client_version="0.1.0",
        )
    )


if __name__ == "__main__":
    main()
