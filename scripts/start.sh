#!/bin/sh
# Production entrypoint (Render, Docker): validate -> migrate -> serve.
set -eu

# 1. Refuse to start with a broken configuration. Prints which settings are
#    wrong, never their values.
python scripts/check_env.py

# 2. Bring the database schema (and the LangGraph checkpoint tables) up to date.
#    Idempotent, and retries while a suspended Neon database wakes up.
python scripts/migrate.py

# 3. Serve.
#    * ONE worker: the free tier has 512 MB, and the in-process rate limiter and
#      WebSocket registry are per process (the database state is shared anyway).
#    * --no-access-log: access lines contain the query string, which carries the
#      single-use voice token. The app logs request paths without queries itself.
#    * --no-proxy-headers: uvicorn would trust the left-most X-Forwarded-For
#      entry, which the client controls. The app's rate limiter reads the header
#      itself, counting from the right (TRUSTED_PROXY_HOPS).
#    * keep-alive above the proxy's idle timeout avoids 502s on reused connections.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --no-access-log \
    --no-proxy-headers \
    --timeout-keep-alive 75 \
    --ws-ping-interval 20 \
    --ws-ping-timeout 20
