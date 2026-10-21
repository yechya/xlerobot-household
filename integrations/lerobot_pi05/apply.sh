#!/usr/bin/env bash
# Clone upstream LeRobot at the reviewed base commit and apply the pi0.5 patch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "usage: $0 <target-directory>" >&2
  exit 2
fi
if [ -e "$TARGET" ]; then
  echo "refusing to overwrite existing path: $TARGET" >&2
  exit 1
fi

UPSTREAM="${UPSTREAM_REPO:-https://github.com/huggingface/lerobot.git}"
BASE_COMMIT="473f1bd0e"
PATCH="$HERE/patches/0001-pi05-jetson-integration.patch"

git clone --filter=blob:none --no-checkout "$UPSTREAM" "$TARGET"
git -C "$TARGET" checkout "$BASE_COMMIT"
git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"

echo
echo "applied $PATCH"
echo "target: $TARGET"
echo "next: cd $TARGET && cp /path/to/xlerobot-repo/.env.example .env && source envs.sh"
