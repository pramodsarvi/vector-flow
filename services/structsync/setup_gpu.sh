#!/usr/bin/env bash
# Linux + NVIDIA GPU setup. Usage:  bash setup_gpu.sh [--download-models] [--with-harmonizer]
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv -p 3.12 .venv
# PyPI torch wheels on Linux ship with CUDA; override with --index-url for a specific CUDA build if needed
uv pip install -p .venv/bin/python -r requirements.lock   # exact verified versions

.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

if [[ " $* " == *" --download-models "* ]]; then
  .venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download as s
for r in ["IDEA-Research/grounding-dino-base", "facebook/sam2.1-hiera-small",
          "google/siglip2-base-patch16-224", "Ruicheng/moge-2-vitl-normal"]:
    print(r, s(r))
EOF
fi
if [[ " $* " == *" --with-harmonizer "* ]]; then
  .venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download as s
s("madebyollin/sdxl-vae-fp16-fix")
s("diffusers/controlnet-depth-sdxl-1.0", allow_patterns=["*.json", "*fp16.safetensors"])
s("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])
EOF
fi

echo
echo "Ready. Next:"
echo "  .venv/bin/python smoke_test.py --harmonize          # one-photo end-to-end test"
echo "  .venv/bin/python eval.py samples/ --harmonize       # batch report → outputs/eval/results.csv"
echo "  .venv/bin/python app.py --host 0.0.0.0 --preload    # UI on :7860 (or --share for a public link)"
