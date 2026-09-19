import os
from huggingface_hub import snapshot_download

token = None
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if line.startswith('HF_TOKEN='):
                token = line.strip().split('=', 1)[1].strip('\"\'')
                break

print("Starting snapshot download for Qwen/Qwen3-4B...")
path = snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    token=token,
    resume_download=True
)
print(f"Downloaded Qwen/Qwen3-4B to: {path}")
