FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies — latest vLLM for Qwen3.5 MoE support
RUN pip install --no-cache-dir vllm runpod>=1.6.2 huggingface_hub>=0.24.0

# Model will be downloaded to RunPod Network Volume on first boot
# (no longer baked into the image — too large for Qwen3.5-35B-A3B at ~70GB)
ENV MODEL_DIR=/runpod-volume/models/qwen3.5-35b-a3b
ENV MODEL_REPO=Qwen/Qwen3.5-35B-A3B

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

# Fix: vLLM v0.24.0 forks EngineCore subprocess — must use 'spawn' to avoid CUDA re-init crash
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
# Delay CUDA initialization to prevent conflicts with forked processes
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]