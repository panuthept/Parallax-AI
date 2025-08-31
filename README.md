# Parallax-AI

Lightweight parallel AI API calls in Python.

## Installation

You can install Parallax using pip:

```bash
pip install parallax-ai
```

## Testing
```bash
vllm serve google/gemma-3-27b-it
```

```bash
python -m parallax_ai.test
>>>
ParallaxOpenAIClient.chat_completions:
First Output Elapsed Time: 7.32s
Total Elapsed Time (500 requires): 7.32s
ParallaxOpenAIClient.ichat_completions:
First Output Elapsed Time: 2.85s
Total Elapsed Time (500 requires): 7.09s
Vanilla OpenAI Client:
First Output Elapsed Time: 1.78s
Total Elapsed Time (500 requires): 672.02s
```

## Usage (Compatible with any OpenAI-API–compatible server, e.g., vLLM)

### Initialize Client
```python
from parallax_ai import ParallaxOpenAIClient

# Initialize Client
parallax_client = ParallaxOpenAIClient(
    api_key=YOUR_API_KEY,
    base_url=YOUR_API_BASE_URL,
)

list_of_messages = [
    [{"role": "user", "content": "..."}],
    [{"role": "user", "content": "..."}],
    [{"role": "user", "content": "..."}],
    .
    .
    .
    [{"role": "user", "content": "..."}],
]
```

### `chat_completions`: Returns a list of outputs in order (waits until all are finished)

```python
outputs = parallax_client.chat_completions(list_of_messages, model="gpt-3.5-turbo")
for output in outputs:
    # PROCESS OUTPUT
    pass
```

### `ichat_completions`: Returns outputs one by one in order (yields as soon as each finishes)

```python
for output in parallax_client.ichat_completions(list_of_messages, model="gpt-3.5-turbo"):
    # PROCESS OUTPUT
    pass
```

### `ichat_completions_unordered`: Returns outputs as they finish (order not guaranteed)
```python
for output, index in parallax_client.ichat_completions_unordered(list_of_messages, model="gpt-3.5-turbo"):
    # PROCESS OUTPUT
    pass
```

