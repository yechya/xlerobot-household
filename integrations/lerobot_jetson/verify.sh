#!/usr/bin/env bash
# Verify the patch applies to a pristine upstream checkout without modifying it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="${UPSTREAM_REPO:-https://github.com/huggingface/lerobot.git}"
BASE_COMMIT="1033680a5"
PATCH="$HERE/patches/0001-xlerobot-jetson-integration.patch"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "patch sha256: $(sha256sum "$PATCH" | awk '{print $1}')"
git clone --quiet --filter=blob:none --no-checkout "$UPSTREAM" "$TMP/lerobot"
git -C "$TMP/lerobot" checkout --quiet "$BASE_COMMIT"
git -C "$TMP/lerobot" apply --check "$PATCH"
echo "OK: patch applies cleanly to $BASE_COMMIT"
