# Parallax-AI

## Installation

```bash
git clone https://github.com/panuthept/Parallax-AI.git
cd Parallax-AI
pip install -e .
```

## Benchmarking Safeguards
Deploy a vLLM Model Serving API
```bash
ray start --head
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8000 \
    --model <your_model_name> \
    --dtype=auto \
    --tensor-parallel-size=4
```
Run Benchmarking Script
```bash
python example_scripts/safeguard_benchmarking.py --model_name <your_model_name> --model_address <your_model_address>
```