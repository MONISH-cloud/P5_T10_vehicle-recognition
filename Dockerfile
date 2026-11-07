FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
COPY app.py .
COPY examples examples
COPY results results
COPY models/meta.json models/meta.json
RUN pip install --no-cache-dir huggingface_hub hf_xet && \
    python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
files = ["classifier_int8_dynamic.onnx", "detector_finetuned.onnx", "classical.npz"]
for f in files:
    hf_hub_download(repo_id="Monish1212121212/p5-vehicle-id-models", filename=f, local_dir="models")
EOF
ENV PORT=7860 GRADIO_ANALYTICS_ENABLED=False MALLOC_ARENA_MAX=2 LEAN_MODE=1
EXPOSE 7860
CMD ["python", "app.py"]
