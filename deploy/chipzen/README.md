# Chipzen deployment

Packages the dev.fun Poker Playground policy as a [Chipzen](https://chipzen.ai)
hosted/upload bot: a sandboxed Docker image the platform runs for you.

## What's in the image

- `bot.py` — Chipzen SDK adapter: maps each `GameState` onto the Playground
  snapshot contract, runs the shared decision rails, and returns a legal
  `Action`. Any internal error degrades to the cheapest legal action
  (check → fold → call), never an illegal submission.
- `devfun_poker_playground/` — the policy package, torch-free at runtime:
  snapshot validation + features (`snapshots.py`), deterministic safety rails
  (`rules.py`), pure-Python inference over exported weights (`pure_model.py`),
  Monte Carlo equity fallback with the vendored MIT-licensed `treys`
  (`equity.py`, `_vendor/treys/`).
- `artifacts/tiny-policy-pure.json` — the trained tiny-policy checkpoint
  exported to JSON by `tools/export_pure_weights.py` (torch never enters the
  image; `pickle` is blocked in the sandbox anyway).

Heads-up decisions are driven by the seeded equity rails (the 6-max
behavior-cloned network engages at full tables), one decision costs ~15-40 ms
against the 2000 ms ranked budget, and everything in the image is pure
Python, so the seccomp sandbox has no native extensions to kill.

## Prerequisites

- Docker installed and running (`docker version` shows Client + Server).
- The weights export at `artifacts/tiny-policy-pure.json` (regenerate with a
  torch-capable Python: `python tools/export_pure_weights.py`).
- For pre-upload validation: `pip install chipzen-bot` somewhere on PATH.

## Build and export

```bash
./deploy/chipzen/build.sh            # Git Bash / WSL / Linux
```

```powershell
.\deploy\chipzen\build.ps1           # PowerShell (gzips without the pipe)
```

Either script builds `playground-bot:v1` for linux/amd64, smoke-tests the bot
inside the image, enforces the platform caps (image ≤ 200 MB, archive
≤ 250 MB compressed), and writes `deploy/chipzen/playground-bot.tar.gz`.

Manual equivalent, from the repo root:

```bash
docker build --platform linux/amd64 -f deploy/chipzen/Dockerfile -t playground-bot:v1 .
docker save playground-bot:v1 | gzip > deploy/chipzen/playground-bot.tar.gz
```

(Do not run that pipe in PowerShell — it corrupts the archive; use Git Bash,
WSL, or `build.ps1`.)

## Validate before uploading

Runs the same checks as the platform's upload pipeline, plus a mock-server
protocol conformance pass:

```bash
chipzen-sdk validate deploy/chipzen --check-connectivity
```

(The validator imports `bot.py` in-process, so run it with the repo's `src/`
on `PYTHONPATH`.)

## Upload

Upload `playground-bot.tar.gz` through the Chipzen developer UI and watch the
bot move `pending_review → reviewing → active`. Review is automatic.

## Runtime contract and knobs

The platform injects `CHIPZEN_WS_URL` and `CHIPZEN_TOKEN` (or
`CHIPZEN_TICKET`) at launch; the entrypoint refuses to start without a URL.
No credentials are baked into the image — never add any; the tarball is
uploaded to a third party.

| Env var | Default | Meaning |
|---|---|---|
| `PLAYGROUND_EQUITY_TRIALS` | `512` | Monte Carlo trials per decision |
| `PLAYGROUND_LOG_LEVEL` | `INFO` | Python log level |
| `POKER_PURE_WEIGHTS` | baked path | Weights JSON location |

## Notes

- The image ships readable `.py` strategy source. If that matters, the SDK's
  IP-protected starter compiles the entry file with Cython — see
  `packages/python/IP-PROTECTION.md` in the chipzen-sdk repo — at the cost of
  adding a native extension to the sandbox equation.
- The base image is pinned to the same `python:3.11-alpine` digest the SDK's
  reference bot uses.
