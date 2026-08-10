# dev.fun Arena — sandbox deployment

Ships the torch-free v3 Poker Playground policy to the **dev.fun Arena**
server-hosted sandbox (the "static-agent" submission flow). The sandbox runs an
uploaded bot and evaluates it; you do not play hands live.

This is the **arena.dev** deployment. It is kept separate from the chipzen.ai
deployment (which lives on the `chipzen` branch under `deploy/chipzen/`). Both
reuse the same `src/devfun_poker_playground` policy; only the adapter differs.

## How it works

`strategy.py` implements the sandbox contract — `choose_action(table)` — where
`table` is the live Playground snapshot (the same contract
`devfun_poker_playground` already consumes: `selfSeatNumber`,
`allowedActions.availableActions`, `callToAmount`, `raiseRange`, …). It calls
`PurePolicy.decide(table)` and maps the payload to
`{action, amount, reasoning_text}` (a to-amount), always returning a legal
action.

Two sandbox constraints are handled by `build_bundle.py`:

- **No weights file read** — the network weights are embedded (gzip + base64)
  into `weights_data.py`, so the policy loads with no filesystem access.
- **256 KB uncompressed `harness/` cap** — the torch-only modules
  (`playground.py`, `model_contract.py`) are dropped and `bundle_init.py`
  supplies a torch-free package init.

## Build and submit

```bash
# 0. Ensure the exported weights exist (a build artifact, gitignored — same as
#    the chipzen pipeline). Regenerate from the torch checkpoint if missing:
python tools/export_pure_weights.py          # writes artifacts/tiny-policy-pure.json

# 1. Build + local smoke test (folds the 3 real-loss spots, raises the boat)
python deploy/devfun-arena/build_bundle.py

# 2. Point at your dev.fun credentials (written by the arena client at
#    registration; keep it untracked)
export ARENA_CREDENTIALS=/path/to/.arena-credentials      # PowerShell: $env:ARENA_CREDENTIALS=...

# 3. Submit the bundle to the Eval competition, then poll for the score
python deploy/devfun-arena/tools/submit.py deploy/devfun-arena/build/../bundle.zip seed_poker_eval_s1
python deploy/devfun-arena/tools/poll.py <submissionId>
```

`poll.py` reports `rawBbPer100` / `adjustedBbPer100` when the run succeeds. A
`TimedOut` with `sandbox_eval_incomplete` is a transient platform issue
("usually temporary, resubmit") and does not count against the daily submission
limit. Only one run executes at a time; PVE runs cannot be cancelled.

## Registration

The agent (`Fold-ver-3`) is registered via the arena client; the sandbox
submission needs only the resulting API key. Submissions to the Eval S1
competition are enabled without the claim/X-verification gate.
