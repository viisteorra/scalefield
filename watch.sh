#!/usr/bin/env bash
# One-command ScaleField watcher: rebuilds INPUT | OUTPUT images and shows them.
#
#   ./watch.sh
#   ./watch.sh --ckpt runs/live/best.npz
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "missing venv python at $PY" >&2
  exit 1
fi

CKPT="${WATCH_CKPT:-$ROOT/runs/live/best.npz}"
DATA="${WATCH_DATA:-$ROOT/data}"
OUT="${WATCH_OUT:-$ROOT/runs/live/watch}"
INTERVAL="${WATCH_INTERVAL:-4}"
OPEN_VIEWER=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --no-viewer) OPEN_VIEWER=0; shift ;;
    --once) EXTRA+=(--once); shift ;;
    -h|--help)
      sed -n '2,8p' "$0"
      echo "Flags: --ckpt PATH  --data DIR  --out DIR  --interval SEC  --no-viewer  --once"
      exit 0
      ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

mkdir -p "$OUT"
BOARD="$OUT/board.png"

echo "rendering first INPUT | OUTPUT board…"
"$PY" "$ROOT/watch.py" --ckpt "$CKPT" --data "$DATA" --out "$OUT" --once "${EXTRA[@]+"${EXTRA[@]}"}"

if [[ " ${EXTRA[*]-} " == *" --once "* ]]; then
  echo "board  $BOARD"
  echo "html   $OUT/index.html"
  exit 0
fi

"$PY" "$ROOT/watch.py" --ckpt "$CKPT" --data "$DATA" --out "$OUT" --interval "$INTERVAL" "${EXTRA[@]+"${EXTRA[@]}"}" &
WATCH_PID=$!

IMV_PID=""
cleanup() {
  if [[ -n "$IMV_PID" ]] && kill -0 "$IMV_PID" 2>/dev/null; then
    kill "$IMV_PID" 2>/dev/null || true
  fi
  if kill -0 "$WATCH_PID" 2>/dev/null; then
    kill "$WATCH_PID" 2>/dev/null || true
    wait "$WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "board  $BOARD"
echo "html   $OUT/index.html"
echo "pairs  $OUT/pairs/   (each *_input.png / *_output.png)"

if [[ "$OPEN_VIEWER" -eq 1 ]] && command -v imv >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  # imv reloads the file when it changes on disk.
  imv -s shrink -u nearest_neighbour -w "ScaleField  INPUT | OUTPUT" "$BOARD" &
  IMV_PID=$!
  echo "viewer imv (closes watcher on window close)"
  wait "$IMV_PID"
else
  echo "no imv/display; watching in the terminal. open:"
  echo "  $BOARD"
  echo "  $OUT/index.html"
  wait "$WATCH_PID"
fi
