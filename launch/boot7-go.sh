#!/bin/bash
# Boot 6 = boot 5 + vision (drop --language-model-only). ONE change vs boot 5. This is the target config left serving.
# No --mm-processor-kwargs (V4.1 processor rejects any kwarg). Processor cache capped at 1 GB (host RAM = GPU pool on GB10).
export EXP_NAME=boot7 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=131072 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
export VLLM_EXTRA='--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "boot6 go $(date -u +%FT%TZ) gmu=$GMU"
bash $HOME/boot_dsv41.sh
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
