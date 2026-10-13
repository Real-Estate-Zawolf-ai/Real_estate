#!/bin/sh
# Kept for existing docker-compose setups; the real entrypoint is start.sh.
exec sh "$(dirname "$0")/start.sh"
