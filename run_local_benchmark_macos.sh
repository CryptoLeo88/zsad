#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  ./run_local_benchmark_macos.sh <image_path> <checkpoint_path> [framework] [output_json]

Example:
  ./run_local_benchmark_macos.sh \
    ./assets/test1.jpg \
    /Users/leo/Downloads/weights/epoch_15.pth \
    open_vocab \
    ./benchmark_results/macos_open_vocab.json

Notes:
  - framework defaults to open_vocab
  - device defaults to auto (prefers cuda, then mps, then cpu)
  - weights are not bundled; download them separately
EOF
  exit 1
fi

IMAGE_PATH="$1"
CHECKPOINT_PATH="$2"
FRAMEWORK="${3:-open_vocab}"
OUTPUT_JSON="${4:-./benchmark_results/macos_${FRAMEWORK}.json}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

python3 benchmark_single_image_inference.py \
  --image_path "$IMAGE_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --framework "$FRAMEWORK" \
  --device auto \
  --warmup 5 \
  --repeat 20 \
  --output_json "$OUTPUT_JSON"
