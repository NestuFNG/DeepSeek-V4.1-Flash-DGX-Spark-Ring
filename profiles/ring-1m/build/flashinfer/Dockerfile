FROM local/dsv41-ring-ssd:20260910
ENV MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1 FLASHINFER_CUDA_ARCH_LIST=12.1a TORCH_CUDA_ARCH_LIST=12.1a
COPY flashinfer.tar.gz cutlass.tar.gz cccl.tar.gz spdlog.tar.gz sources.json /opt/dsv41-upgrade/
RUN python3 -c "import json,hashlib,pathlib; p=pathlib.Path('/opt/dsv41-upgrade'); [(_ for _ in ()).throw(ValueError(x['name'])) if hashlib.sha256((p/(x['name']+'.tar.gz')).read_bytes()).hexdigest()!=x['sha256'] else None for x in json.loads((p/'sources.json').read_text())]"
RUN mkdir -p /opt/fi-src && tar xzf /opt/dsv41-upgrade/flashinfer.tar.gz -C /opt/fi-src --strip-components=1 \
 && mkdir -p /opt/fi-src/3rdparty/cutlass /opt/fi-src/3rdparty/cccl /opt/fi-src/3rdparty/spdlog \
 && tar xzf /opt/dsv41-upgrade/cutlass.tar.gz -C /opt/fi-src/3rdparty/cutlass --strip-components=1 \
 && tar xzf /opt/dsv41-upgrade/cccl.tar.gz -C /opt/fi-src/3rdparty/cccl --strip-components=1 \
 && tar xzf /opt/dsv41-upgrade/spdlog.tar.gz -C /opt/fi-src/3rdparty/spdlog --strip-components=1
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin flashinfer-python \
 && cd /opt/fi-src && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation .
COPY patches/ /opt/dsv41-upgrade/patches/
COPY install_patches.py /opt/dsv41-upgrade/install_patches.py
RUN python3 /opt/dsv41-upgrade/install_patches.py
RUN python3 -c "import flashinfer; from flashinfer.mla import supported_sparse_mla_sm120_configs as f; c=f()['dsv4']; assert c.supports_decode(num_heads=16,topk=1152); print('FlashInfer version',flashinfer.__version__,'1152 decode supported')"
RUN python3 -c "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8; gen_gemm_sm120_module_cutlass_mxfp8().build(); print('MXFP8 build complete')"
RUN python3 -c "from flashinfer.jit.mla import gen_sparse_mla_sm120_module; gen_sparse_mla_sm120_module().build(); print('Sparse MLA build complete')"
