FROM vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
RUN echo '41c5bdf25cf8337088247be768e3d43fa41e6358397800f8b602ebf1ede69480  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py' | sha256sum -c - \
 && echo '494b07cd6b4e4fb40314dec1d0e0c3b433fe62a135a3a9d6d9e4058059054e4a  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py' | sha256sum -c -
COPY engram.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py
COPY weight_utils.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
COPY manifest.json /opt/dsv41-ring-patch-manifest.json
RUN python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
LABEL org.opencontainers.image.description="DGX Spark V4.1 Flash SSD Engram patch; pinned official 0909 ARM64 base"
