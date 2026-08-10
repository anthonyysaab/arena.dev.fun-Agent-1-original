# dev.fun Poker Playground policy

An independent, deliberately small inference project for dev.fun Poker Playground. It does not
import or modify the legacy `poker-agent` repository, and it performs no network calls, registration,
joining, action submission, leaving, or rebuying.

The only boundary with the separate training project is a validated checkpoint contract:

- 125 ordered float features;
- three proposal classes: `fold`, `check_call`, and `aggress`;
- a one-hidden-layer PyTorch state dictionary plus contract metadata.

At load time, the adapter rejects checkpoints whose feature names, input size, or labels differ.
It then maps a proposal to a legal Arena action and a bounded raise-to amount. Equity simulation is
only a safety fallback for table shapes outside the six-player warm-start data.

## Setup and test

```powershell
cd C:\Users\user\devfun-poker-playground
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

By default the policy looks first for `artifacts/tiny-policy.pt`, then for the sibling training
artifact at `..\poker-nn-training\artifacts\tiny-policy.pt`. Override that explicitly with
`POKER_POLICY_CHECKPOINT`.

```python
from devfun_poker_playground import PlaygroundPolicy

policy = PlaygroundPolicy()
payload = policy.decide(table_snapshot, deadline_s=12.0)
```

`table_snapshot` is one fresh item from Arena's pending-actions response. A network client is a
separate future layer; this package currently stops at producing a validated action payload.

## Torch-free core and deployment builds

The snapshot validation (`snapshots.py`), deterministic safety rails
(`rules.py`), equity fallback (`equity.py`, with a vendored MIT-licensed
`treys` under `_vendor/`), and the feature/label contract (`contract.py`) are
torch-free. `PlaygroundPolicy` layers the trained PyTorch checkpoint on top;
`PurePolicy` runs the same rails over JSON weights exported by
`tools/export_pure_weights.py` (equivalence is covered by
`tests/test_pure_model_equivalence.py`). Importing the package without torch
installed keeps everything except `PlaygroundPolicy` usable.

## Chipzen arena container

`deploy/chipzen/` packages the policy as a [chipzen.ai](https://chipzen.ai)
upload bot (sandboxed Docker image, pure-Python, ~30 MB): see
[deploy/chipzen/README.md](deploy/chipzen/README.md) for the
export → validate → build → upload pipeline.
