#!/bin/bash
# Boot 4 = boot 3 + overlay5 (FlashInfer mxfp8_gemm + sparse_mla prebuilt under the runtime env; its runtime compile wedged all 4 nodes in boot 3)
#          + MAX_JOBS=2 cap so any residual runtime JIT cannot exhaust unified memory. Text-only, eager, no DSpark, thinking off.
export EXP_NAME=boot8 IMAGE=vllm-dsv41:overlay5 PATCH_DIR=$HOME/patches/dsv41-boot3 GMU=${GMU:-0.80} MAXLEN=131072 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=0 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0"
echo "boot4 go $(date -u +%FT%TZ) gmu=$GMU"
echo "boot_dsv41 exit=$? $(date -u +%FT%TZ)"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
bash $HOME/boot_dsv41.sh
