#!/usr/bin/env bash
set -euo pipefail
p="$(cd -- "$(dirname -- "$0")/.." && pwd)"
[ "$(uname -m)" = aarch64 ] || { echo 'Build this profile on ARM64 Linux'; exit 2; }
python3 "$p/scripts/verify-profile.py"
python3 "$p/scripts/fetch-build-sources.py"
docker pull --platform linux/arm64 vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
docker build --pull=false --network=none --provenance=false -t local/dsv41-ring-ssd:20260910 "$p/build/base"
docker build --pull=false --network=none --provenance=false -t local/dsv41-ring-ssd:fi07-20260911 "$p/build/flashinfer"
echo 'Images built. Verify collectives and runtime behavior before starting the full checkpoint.'
