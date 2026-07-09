FROM vllm/vllm-openai:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install additional runpod serverless dependencies (vLLM and torch are already pre-installed and matched!)
RUN pip install --no-cache-dir runpod>=1.6.2 huggingface_hub>=0.24.0

# Model will be downloaded to RunPod Network Volume on first boot
# (no longer baked into the image — too large for Qwen3.5-35B-A3B at ~70GB)
ENV MODEL_DIR=/runpod-volume/models/qwen3.5-35b-a3b
ENV MODEL_REPO=Qwen/Qwen3.5-35B-A3B

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

# Fix CUDA driver/toolkit version mismatch (host has CUDA 12.8, vLLM builds for newer)
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1
# Fix: vLLM v0.24.0 forks EngineCore subprocess — must use 'spawn' to avoid CUDA re-init crash
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
# Delay CUDA initialization to prevent conflicts with forked processes
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /app
COPY handler.py /app/handler.py

# Override the entrypoint of the base image so we can run our handler.py instead of the vLLM server directly
ENTRYPOINT []

CMD ["python3", "handler.py"]