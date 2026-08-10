"""Assemble the dev.fun Arena sandbox bundle (``bundle.zip``) and smoke-test it.

The dev.fun sandbox runs an uploaded ``strategy.py`` (or a ``bundle.zip``) in a
restricted container. Two constraints shape this build:

* the sandbox blocks the weights file read, so the network weights are embedded
  (gzip + base64) into ``weights_data.py`` instead of shipped as a JSON asset;
* ``harness/`` must stay under 256 KB uncompressed, so the torch-only modules
  (``playground.py``, ``model_contract.py``) are dropped and a torch-free
  package init (``bundle_init.py``) is used.

Layout inside the zip::

    harness/strategy.py
    harness/weights_data.py                 (gzip+base64 of tiny-policy-pure.json)
    harness/devfun_poker_playground/...      (trimmed torch-free policy package)

Run from anywhere::

    python deploy/devfun-arena/build_bundle.py

then upload ``bundle.zip`` via ``tools/submit.py`` (see the README).
"""

from __future__ import annotations

import base64
import gzip
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC_PKG = REPO / "src" / "devfun_poker_playground"
WEIGHTS_JSON = REPO / "artifacts" / "tiny-policy-pure.json"

BUILD = HERE / "build"
HARNESS = BUILD / "harness"
BUNDLE = HERE / "bundle.zip"
SIZE_CAP = 262144  # 256 KB uncompressed harness/ cap enforced by the sandbox


def _ignore(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def assemble() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    HARNESS.mkdir(parents=True)

    pkg = HARNESS / "devfun_poker_playground"
    shutil.copytree(SRC_PKG, pkg, ignore=_ignore)
    for torch_only in ("playground.py", "model_contract.py"):
        target = pkg / torch_only
        if target.exists():
            target.unlink()
    shutil.copy2(HERE / "bundle_init.py", pkg / "__init__.py")

    blob = base64.b64encode(gzip.compress(WEIGHTS_JSON.read_bytes(), 9)).decode("ascii")
    (HARNESS / "weights_data.py").write_text(
        "import base64, gzip, json\n"
        f'_B64 = "{blob}"\n'
        "WEIGHTS = json.loads(gzip.decompress(base64.b64decode(_B64)))\n",
        encoding="utf-8",
    )
    shutil.copy2(HERE / "strategy.py", HARNESS / "strategy.py")

    harness_bytes = sum(p.stat().st_size for p in HARNESS.rglob("*") if p.is_file())
    print(f"harness/ uncompressed: {harness_bytes} bytes (cap {SIZE_CAP})")
    if harness_bytes >= SIZE_CAP:
        raise SystemExit(f"harness/ too big: {harness_bytes} >= {SIZE_CAP}")

    if BUNDLE.exists():
        BUNDLE.unlink()
    files = 0
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(HARNESS.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD).as_posix())
                files += 1
    print(f"bundle.zip: {files} files, {BUNDLE.stat().st_size // 1024} KB -> {BUNDLE}")


def _board(text):
    return [text[i : i + 2] for i in range(0, len(text), 2)]


def _table(street, hole, board, pot, stack, to_call, raises, available):
    return {
        "id": "probe", "tableId": "probe", "street": street, "potChips": pot,
        "currentBet": to_call, "boardCards": _board(board),
        "smallBlindChips": 50, "bigBlindChips": 100, "selfSeatNumber": 2,
        "seats": [
            {"seatNumber": 1, "status": "Active", "stackChips": stack, "currentBetChips": to_call, "holeCards": None},
            {"seatNumber": 2, "status": "Active", "stackChips": stack, "currentBetChips": 0, "holeCards": _board(hole)},
        ],
        "recentEvents": [
            {"type": "ActionTaken", "street": street,
             "summary": {"action": "raise", "seatNumber": 1, "amount": to_call}}
            for _ in range(raises)
        ],
        "allowedActions": {
            "canFold": "fold" in available, "canCheck": "check" in available,
            "canCall": "call" in available, "canBet": "bet" in available,
            "canRaise": "raise" in available, "canAllIn": "all-in" in available,
            "callChips": to_call, "callAmount": to_call, "callToAmount": to_call,
            "minBet": None, "minRaiseTo": (to_call * 2 if to_call else 100),
            "betRange": None, "raiseRange": {"min": to_call * 2 if to_call else 100, "max": stack},
            "allInToAmount": None, "availableActions": available,
            "amountSemantics": "toAmount", "reasoningRequired": False,
        },
    }


def smoke_test() -> int:
    sys.path.insert(0, str(HARNESS))
    import strategy  # noqa: E402  (imported from the assembled bundle)

    facing = ["fold", "call", "raise"]
    cases = [
        ("A3c 777K turn raise -> fold", _table("turn", "Ac3c", "7s7cKs7h", 4700, 7990, 1540, 2, facing), "fold"),
        ("K4s 7766K river lead -> fold", _table("river", "4sKs", "7h6s7s6dKh", 1600, 9600, 800, 1, facing), "fold"),
        ("KQ 555-2-6 river bet -> fold", _table("river", "KsQc", "5s5c2s5d6h", 1263, 9195, 463, 1, facing), "fold"),
        ("K6 boat 777K turn raise -> continue", _table("turn", "Kd6d", "7s7cKs7h", 4700, 7990, 1540, 2, facing), "not-fold"),
    ]
    ok = True
    for name, table, want in cases:
        action = strategy.choose_action(table).get("action")
        passed = (action == "fold") if want == "fold" else (action != "fold")
        ok = ok and passed
        print(f"  {'ok ' if passed else 'XX '}{name}: got {action}")
    return 0 if ok else 1


if __name__ == "__main__":
    assemble()
    print("=== local smoke test ===")
    rc = smoke_test()
    print("SMOKE", "PASS" if rc == 0 else "FAIL")
    raise SystemExit(rc)
