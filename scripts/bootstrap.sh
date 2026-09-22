#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/upstream.lock"

[[ -f "$ROOT/upstream/TorchSpec/torchspec/train_entry.py" ]] || {
  echo "Vendored TorchSpec source is incomplete" >&2; exit 1;
}
[[ -f "$ROOT/upstream/sglang/python/pyproject.toml" ]] || {
  echo "Vendored SGLang source is incomplete" >&2; exit 1;
}

python3.12 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.lock"
"$ROOT/.venv/bin/pip" install -e "$ROOT/upstream/TorchSpec"

if [[ "${INSTALL_SGLANG:-1}" == "1" ]]; then
  python3.12 -m venv "$ROOT/.venv-sglang"
  "$ROOT/.venv-sglang/bin/pip" install --upgrade pip
  "$ROOT/.venv-sglang/bin/pip" install -e "$ROOT/upstream/sglang/python"
fi

echo "Bootstrap complete from vendored, pinned source. Run ./scripts/preflight.sh next."
