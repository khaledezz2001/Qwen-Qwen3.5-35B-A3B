import time
import os
import sys
import traceback

# Must be set BEFORE importing vllm — vLLM v0.24.0 reads these env vars
# to decide how to spawn its EngineCore subprocess
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Enable CUDA forward-compatibility layer (host driver CUDA 12.8, vLLM built for newer)
os.environ.setdefault("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Configuration
# =====================================================
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models/qwen3.5-35b-a3b")
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3.5-35B-A3B")

# =====================================================
# Download model to Network Volume (first boot only)
# =====================================================
def ensure_model_on_volume():
    """
    Check if the model already exists on the RunPod Network Volume.
    If not, download it from Hugging Face. Subsequent workers will
    find the model already present and skip the download.
    """
    if not os.path.exists("/runpod-volume"):
        log("ERROR: /runpod-volume does not exist!")
        log("Make sure you attached a Network Volume to this endpoint in RunPod console.")
        log("Go to: Serverless → Your Endpoint → Edit → Advanced → Network Volumes")
        sys.exit(1)

    config_path = os.path.join(MODEL_DIR, "config.json")

    if os.path.exists(config_path):
        log(f"Model already present at {MODEL_DIR} — skipping download")
        return MODEL_DIR

    log(f"Model not found at {MODEL_DIR} — downloading {MODEL_REPO} from Hugging Face...")
    log("This will take 10-20 minutes on first boot (model is ~70 GB)")

    os.makedirs(MODEL_DIR, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
    )

    log(f"Download complete — model saved to {MODEL_DIR}")
    return MODEL_DIR

# =====================================================
# RunePod handler (defined at module level, uses global `llm`)
# =====================================================
def handler(event):
    log("Handler started")

    input_data = event["input"]
    
    system_prompt = input_data.get("system_prompt")
    user_prompt = input_data.get("user_prompt")
    temperature = input_data.get("temperature", 0.7)
    top_p = input_data.get("top_p", 0.9)
    max_new_tokens = input_data.get("max_new_tokens", 1024)
    
    if not user_prompt:
        log("Error: user_prompt is required")
        return {"error": "user_prompt is required"}

    log(f"Incoming Request - System Prompt: {system_prompt}")
    log(f"Incoming Request - User Prompt: {user_prompt}")

    log("Starting text generation...")
    start_time = time.time()
    
    is_list = isinstance(user_prompt, list)
    prompts_to_process = user_prompt if is_list else [user_prompt]
    
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    
    for q in prompts_to_process:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})
        
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted)
    
    sampling_params = SamplingParams(
        temperature=temperature if temperature and temperature > 0.0 else 0.0,
        top_p=top_p if top_p else 1.0,
        max_tokens=max_new_tokens,
    )
    
    try:
        outputs = llm.generate(formatted_prompts, sampling_params)
        
        response_data = []
        for idx, output in enumerate(outputs):
            text = output.outputs[0].text
            response_data.append(text)
            
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            finish_reason = output.outputs[0].finish_reason
            log(f"vLLM State (Req {idx}) - Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Finish Reason: {finish_reason}")
        
    except Exception as e:
        err_msg = f"Generation failed: {str(e)}"
        log(err_msg)
        return {"error": err_msg}

    log(f"Text generation completed in {time.time() - start_time:.4f}s")
    
    if not is_list:
        response_data = response_data[0]
        log(f"Generated Response: {response_data}")
        return {"response": response_data}
    else:
        log(f"Generated Responses: {response_data}")
        return {"response": response_data}

# =====================================================
# Main entry point — guarded for multiprocessing spawn safety
# =====================================================
# When vLLM uses 'spawn', it re-imports this module in the child process.
# Without this guard, all initialization code would run again in the child,
# causing the "bootstrapping phase" crash.
if __name__ == '__main__':
    log("="*60)
    log("Worker starting up...")
    log(f"Python version: {sys.version}")
    log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
    log(f"Network volume exists: {os.path.exists('/runpod-volume')}")
    log(f"Network volume contents: {os.listdir('/runpod-volume') if os.path.exists('/runpod-volume') else 'N/A'}")
    log("="*60)

    # Diagnostic import of torch to see what PyTorch version/CUDA version is loaded
    try:
        import torch
        log(f"PyTorch version: {torch.__version__}")
        log(f"PyTorch compiled with CUDA: {torch.version.cuda}")
        log(f"PyTorch believes CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"PyTorch detected GPU device name: {torch.cuda.get_device_name(0)}")
            log(f"PyTorch detected device count: {torch.cuda.device_count()}")
            log(f"PyTorch detected driver version (internal): {torch.cuda.get_device_capability(0)}")
    except Exception as e:
        log(f"Failed to run diagnostic torch import: {e}")


    # Import heavy deps
    try:
        log("Importing runpod...")
        import runpod
        log("runpod imported OK")
    except Exception as e:
        log(f"FATAL: Failed to import runpod: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        log("Importing vllm...")
        from vllm import LLM, SamplingParams
        log("vllm imported OK")
    except Exception as e:
        log(f"FATAL: Failed to import vllm: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Ensure model is on volume
    try:
        model_path = ensure_model_on_volume()
    except Exception as e:
        log(f"FATAL: Failed to ensure model on volume: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Initialize vLLM engine
    log("Initializing vLLM engine...")
    start = time.time()

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=8,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
        )
        log(f"vLLM engine ready in {time.time() - start:.1f}s")
    except Exception as e:
        log(f"FATAL: Failed to initialize vLLM engine: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Start RunPod serverless
    runpod.serverless.start({"handler": handler})