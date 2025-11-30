#!/usr/bin/env bash
set -euo pipefail

# Default values can be overridden by exporting SERVER_HOST / SERVER_PORT before running.
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
FORWARD_HOST="${NGROK_FORWARD_HOST:-127.0.0.1}"
NGROK_ONLY="${NGROK_ONLY:-0}"

if [[ "${NGROK_ONLY}" != "1" ]]; then
  if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Error: no Python interpreter found. Export PYTHON=/path/to/python or add python3 to PATH." >&2
    exit 1
  fi
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "Error: ngrok executable not found in PATH. Install it from https://ngrok.com/download and run 'ngrok config add-authtoken <token>'." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && ps -p "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
start_server() {
  echo "Starting DNI pipeline server on http://${SERVER_HOST}:${SERVER_PORT} ..."
  "${PYTHON_BIN}" -m dni_pipeline.server --host "${SERVER_HOST}" --port "${SERVER_PORT}" "$@" &
  SERVER_PID=$!
  sleep 2
  if ! ps -p "${SERVER_PID}" >/dev/null 2>&1; then
    echo "The server process exited unexpectedly. Check the logs above." >&2
    wait "${SERVER_PID}"
    exit 1
  fi
}

if [[ "${NGROK_ONLY}" != "1" ]]; then
  trap cleanup EXIT INT TERM
  start_server "$@"
  echo "Launching ngrok tunnel (Ctrl+C to stop both the tunnel and the server)..."
else
  echo "Skipping server launch (NGROK_ONLY=1). Ensure something is already listening on http://${FORWARD_HOST}:${SERVER_PORT}"
  echo "Launching ngrok tunnel (Ctrl+C to stop the tunnel)..."
fi
ngrok http "http://${FORWARD_HOST}:${SERVER_PORT}"
