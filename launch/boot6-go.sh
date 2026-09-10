#!/bin/bash
# Boot 5 = boot 4 + DSpark (recipe config, k=5, probabilistic draft, block rejection, adaptive verification). ONE change vs boot 4.
export EXP_NAME=boot6 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=131072 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
echo "boot5 go $(date -u +%FT%TZ) gmu=$GMU"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
