#!/bin/bash
# Boot 8 = boot 7 at 300000 max context: the DSpark serving config, benched and left running.
export EXP_NAME=boot8 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=300000 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN spec=$SPEC"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
