FROM vllm/vllm-openai:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install additional runpod serverless dependencies and the latest transformers to support glm4_moe_lite
RUN pip install --no-cache-dir runpod>=1.6.2 huggingface_hub>=0.24.0 && \
    pip install --no-cache-dir https://github.com/huggingface/transformers/archive/refs/heads/main.zip

# Model will be downloaded to RunPod Network Volume on first boot
ENV MODEL_DIR=/runpod-volume/models/glm-4.7-flash
ENV MODEL_REPO=zai-org/GLM-4.7-Flash

ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_CACHE=/runpod-volume/hf-cache

# Prioritize host NVIDIA driver libraries to prevent Error 803 (driver/compat mismatch)
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# Remove container-bundled cuda-compat shims so they don't conflict with host NVIDIA drivers
RUN rm -rf /usr/local/cuda/compat

# Fix: vLLM v0.24.0 forks EngineCore subprocess — must use 'spawn' to avoid CUDA re-init crash
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
# Delay CUDA initialization to prevent conflicts with forked processes
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /app
COPY handler.py /app/handler.py

# Override the entrypoint of the base image so we can run our handler.py instead of the vLLM server directly
ENTRYPOINT []

CMD ["python3", "handler.py"]