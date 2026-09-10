#!/usr/bin/env bash
# Source patch: FujitsuPolycom/sparkring, Apache-2.0. See repository NOTICE.md.
set -euo pipefail
p="$(cd -- "$(dirname -- "$0")/.." && pwd)"
source_dir="$p/build/nccl/source"
output_dir="${1:?Usage: build-nccl.sh /absolute/empty/output/directory}"
[ "$(uname -m)" = aarch64 ] || { echo 'Build on ARM64 Linux with CUDA 13'; exit 2; }
command -v nvcc >/dev/null
[ ! -e "$source_dir" ] || { echo 'Source directory exists; inspect it before rebuilding'; exit 3; }
[ ! -e "$output_dir" ] || { echo 'Output directory exists; choose a new directory'; exit 3; }
git clone https://github.com/NVIDIA/nccl.git "$source_dir"
git -C "$source_dir" checkout --detach 73cf112295c33aee2b895f329f592f2a9b4b0f97
printf '%s  %s\n' 097656d07a5774919f0d51558b51ec05de8168c0097ed6cb7764c33230ba6eb2 "$p/build/nccl/nccl-2.30.7-skip-tree-pat.patch" | sha256sum -c -
git -C "$source_dir" apply --check "$p/build/nccl/nccl-2.30.7-skip-tree-pat.patch"
git -C "$source_dir" apply "$p/build/nccl/nccl-2.30.7-skip-tree-pat.patch"
make -C "$source_dir" -j"${BUILD_JOBS:-4}" src.build NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
mkdir -p "$output_dir"
cp -a "$source_dir/build/lib/libnccl.so"* "$output_dir/"
cp "$source_dir/LICENSE.txt" "$output_dir/LICENSE.NCCL.txt"
cp "$p/../../LICENSES/Apache-2.0.txt" "$output_dir/LICENSE.patch.txt"
cp "$p/build/nccl/nccl-2.30.7-skip-tree-pat.patch" "$output_dir/"
sha256sum "$output_dir/libnccl.so.2.30.7"
echo 'Source build complete; binary identity depends on toolchain. Run four-node collectives before serving.'
