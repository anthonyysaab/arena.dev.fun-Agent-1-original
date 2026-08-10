"""dev.fun Arena static-agent adapter for the v3 Poker Playground policy.

The Arena sandbox runs this file and calls ``choose_action(table)`` once per
decision, handing it the live Playground table snapshot. That snapshot is
exactly the contract the torch-free :class:`PurePolicy` already consumes, so
this adapter forwards the table and maps the returned payload onto the
sandbox's expected return shape.

Robust by construction: the policy is built inside a guard so a module-level
failure can never fault the agent, the network weights are embedded (no
filesystem read in the sandbox — see build_bundle.py), and every return path
emits a legal action read from the live table.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_policy = None
_init_error = ""
try:
    from devfun_poker_playground.equity import prewarm
    from devfun_poker_playground.pure_model import PurePolicy
    from weights_data import WEIGHTS

    prewarm()
    _policy = PurePolicy(
        weights=WEIGHTS,
        equity_trials=int(os.environ.get("PLAYGROUND_EQUITY_TRIALS", "250")),
    )
except Exception as exc:  # keep the agent importable no matter what
    _init_error = repr(exc)


def _legal_actions(table):
    allowed = (table.get("allowedActions") if hasattr(table, "get") else None) or {}
    actions = [str(a) for a in (allowed.get("availableActions") or [])]
    if not actions:
        for name, flag in (
            ("check", "canCheck"),
            ("fold", "canFold"),
            ("call", "canCall"),
            ("bet", "canBet"),
            ("raise", "canRaise"),
            ("all-in", "canAllIn"),
        ):
            if allowed.get(flag):
                actions.append(name)
    return actions


def _fallback(table, note):
    actions = _legal_actions(table)
    for preferred in ("check", "call", "fold"):
        if preferred in actions:
            return {"action": preferred, "reasoning_text": note}
    if actions:
        return {"action": actions[0], "reasoning_text": note}
    return {"action": "fold", "reasoning_text": note}


def choose_action(table):
    if _policy is None:
        return _fallback(table, "playing it safe")
    try:
        payload = _policy.decide(table)
    except Exception:
        return _fallback(table, "keeping it simple here")

    actions = _legal_actions(table)
    action = payload.get("action")
    if action not in actions:
        return _fallback(table, "adjusting to the legal set")

    result = {"action": action, "reasoning_text": payload.get("message") or "range-aware line"}
    amount = payload.get("amount")
    if amount is not None:
        try:
            result["amount"] = int(amount)
        except (TypeError, ValueError):
            return _fallback(table, "sizing fallback")
    return result


# The sandbox contract accepts either name.
act = choose_action
