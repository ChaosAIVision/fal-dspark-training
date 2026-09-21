#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/upstream.lock"

clone_at() {
  local repo="$1" commit="$2" destination="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$repo" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

mkdir -p "$ROOT/upstream"
clone_at "$TORCHSPEC_REPO" "$TORCHSPEC_COMMIT" "$ROOT/upstream/TorchSpec"
clone_at "$SGLANG_REPO" "$SGLANG_COMMIT" "$ROOT/upstream/sglang"

if [[ -s "$ROOT/patches/torchspec.patch" ]]; then
  git -C "$ROOT/upstream/TorchSpec" apply --check "$ROOT/patches/torchspec.patch" 2>/dev/null \
    && git -C "$ROOT/upstream/TorchSpec" apply "$ROOT/patches/torchspec.patch" \
    || git -C "$ROOT/upstream/TorchSpec" apply --reverse --check "$ROOT/patches/torchspec.patch"
fi

python3.12 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.lock"
"$ROOT/.venv/bin/pip" install -e "$ROOT/upstream/TorchSpec"

if [[ "${INSTALL_SGLANG:-1}" == "1" ]]; then
  python3.12 -m venv "$ROOT/.venv-sglang"
  "$ROOT/.venv-sglang/bin/pip" install --upgrade pip
  "$ROOT/.venv-sglang/bin/pip" install -e "$ROOT/upstream/sglang/python"
fi

echo "Bootstrap complete. Run ./scripts/preflight.sh next."
