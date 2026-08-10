# syntax=docker/dockerfile:1.7
#
# Chipzen upload image for the dev.fun Poker Playground policy.
#
# Torch-free by design: inference runs over exported JSON weights
# (artifacts/tiny-policy-pure.json, produced by tools/export_pure_weights.py)
# and the vendored pure-Python treys evaluator. Every dependency is pure
# Python, so the platform's seccomp sandbox has no native extensions to
# object to, and the image stays far under the 200 MB image /
# 250 MB compressed caps.
#
# Build from the REPO ROOT (the build context needs src/ and artifacts/):
#   docker build -f deploy/chipzen/Dockerfile -t playground-bot:v1 .
#
# Export for upload:
#   docker save playground-bot:v1 | gzip > playground-bot.tar.gz
#   (Run the pipe in Git Bash or WSL - PowerShell corrupts binary pipes -
#    or use deploy/chipzen/build.ps1 which gzips without a pipe.)
#
# Base pinned by digest, same as the SDK reference bot. Tag: python:3.11-alpine
FROM python:3.11-alpine@sha256:8b5bfdb1fd2d78aa94e21c4d61be52487693f54be7f1021647751ff365795703

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /bot

# Chipzen SDK (pulls only the pure-Python `websockets`).
COPY deploy/chipzen/requirements.txt /bot/requirements.txt
RUN pip install -r /bot/requirements.txt \
    && rm -rf /root/.cache \
    && find /usr/local/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11 -type d -name tests -exec rm -rf {} + 2>/dev/null || true

# The policy package (validation, rules, pure inference, vendored treys),
# the exported weights, and the adapter entry point.
COPY src/devfun_poker_playground /bot/devfun_poker_playground
COPY artifacts/tiny-policy-pure.json /bot/artifacts/tiny-policy-pure.json
COPY deploy/chipzen/bot.py /bot/bot.py

ENV POKER_PURE_WEIGHTS=/bot/artifacts/tiny-policy-pure.json

# Non-root user (defense in depth; the platform adds seccomp, cap-drop, and
# a read-only root FS on top - /tmp is the only writable path at runtime).
RUN adduser -D -u 10001 -s /bin/sh bot \
    && chown -R bot:bot /bot
USER bot

# The platform injects CHIPZEN_WS_URL and CHIPZEN_TOKEN / CHIPZEN_TICKET.
ENTRYPOINT ["python", "-u", "/bot/bot.py"]
