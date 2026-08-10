"""Analyze one or more chipzen match replays for the user's bot.

Usage: python analyze_matches.py <match-id-or-url> [...]

Fetches each replay (cached to disk), auto-detects the hero seat via the
bot id, and prints per-match plus aggregate leak stats.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOT_ID = "77b64112-391f-4d4c-96fe-cdc85d898a6a"  # Fold-ver-2
API = "https://chipzen.ai/api/matches/{}"


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.9.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def load_match(match_id: str) -> tuple[dict, dict]:
    cache = HERE / f"match-{match_id}.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
    else:
        blob = {
            "meta": fetch(API.format(match_id)),
            "replay": fetch(API.format(match_id) + "/replay"),
        }
        cache.write_text(json.dumps(blob), encoding="utf-8")
    return blob["meta"], blob["replay"]


def hero_seat(meta: dict) -> int | None:
    for participant in meta.get("participants", []):
        if participant.get("bot_id") == BOT_ID:
            return participant["seat"]
    return None


def analyze(match_id: str, totals: Counter, all_worst: list) -> None:
    meta, replay = load_match(match_id)
    seat = hero_seat(meta)
    if seat is None:
        print(f"== {match_id}: our bot is not in this match, skipping")
        return
    opponent = next(
        p["name"] for p in meta["participants"] if p["seat"] != seat
    )
    hands = replay["hands"]
    nets = []
    for hand in hands:
        start = next(
            s["starting_stack"] for s in hand["starting_stacks"] if s["seat"] == seat
        )
        net = hand["stacks_after"][str(seat)] - start
        nets.append((hand["hand_number"], net))
        hero_hole = hand["hole_cards"].get(str(seat))

        for street in hand["streets"]:
            acts = street["actions"]
            for index, act in enumerate(acts):
                if act["seat"] == seat:
                    key = f"hero_{street['street']}_{act['action_type']}"
                    totals[key] += 1
                    continue
                if act["action_type"] != "raise":
                    continue
                nxt = acts[index + 1] if index + 1 < len(acts) else None
                if nxt is None or nxt["seat"] != seat:
                    continue
                street_name = street["street"]
                totals[f"faced_bet_{street_name}"] += 1
                totals[f"faced_bet_{street_name}_{nxt['action_type']}"] += 1
                if nxt["action_type"] == "call":
                    totals[f"chips_calling_{street_name}"] += nxt["amount"]
                    if net < 0:
                        totals[f"lost_after_{street_name}_call"] += 1
        if net <= -800:
            all_worst.append((match_id[:8], hand["hand_number"], net, hero_hole,
                              hand["community_cards"], hand.get("showdown")))

    total = sum(n for _, n in nets)
    wins = [n for _, n in nets if n > 0]
    losses = [n for _, n in nets if n < 0]
    totals["net"] += total
    totals["hands"] += len(hands)
    totals["wins_total"] += sum(wins)
    totals["losses_total"] += sum(losses)
    totals["wins_count"] += len(wins)
    totals["losses_count"] += len(losses)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    print(
        f"== {match_id[:8]} vs {opponent}: {len(hands)} hands, net {total:+}, "
        f"avg win {avg_win:+.0f} / avg loss {avg_loss:+.0f}"
    )


def main() -> int:
    ids = []
    for arg in sys.argv[1:]:
        match = re.search(r"[0-9a-f]{8}-[0-9a-f-]{27,}", arg)
        if match:
            ids.append(match.group(0))
    if not ids:
        print("usage: analyze_matches.py <match-id-or-url> [...]")
        return 1

    totals: Counter = Counter()
    worst: list = []
    for match_id in ids:
        analyze(match_id, totals, worst)

    print("\n=== aggregate ===")
    print(f"hands {totals['hands']}  net {totals['net']:+}")
    if totals["wins_count"]:
        print(
            f"wins {totals['wins_count']} avg {totals['wins_total']/totals['wins_count']:+.0f}"
            f"  losses {totals['losses_count']} avg {totals['losses_total']/totals['losses_count']:+.0f}"
        )
    for street in ("flop", "turn", "river"):
        faced = totals[f"faced_bet_{street}"]
        if not faced:
            continue
        called = totals[f"faced_bet_{street}_call"]
        folded = totals[f"faced_bet_{street}_fold"]
        lost = totals[f"lost_after_{street}_call"]
        chips = totals[f"chips_calling_{street}"]
        print(
            f"faced {street} bets: {faced}  called {called} ({100*called/faced:.0f}%)"
            f"  folded {folded}  lost-after-call {lost}  chips spent {chips}"
        )

    print("\n=== hands lost >= 800 ===")
    for match_id, number, net, hole, board, showdown in sorted(worst, key=lambda w: w[2]):
        line = f"{match_id} hand {number}: {net:+} hole={hole} board={board}"
        if showdown:
            opp = [s for s in showdown if s.get("hole_cards")]
            line += f" showdown={[(s['seat'], s['hole_cards'], s.get('hand_rank')) for s in opp]}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
