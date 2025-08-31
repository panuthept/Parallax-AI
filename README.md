# Parallax-AI

Lightweight parallel AI API calls in Python.

## Installation

You can install Parallax using pip:

```bash
pip install parallax-ai
```

## Usage

```python
from parallax_ai import ParallaxOpenAIClient

# Initialize Client
parallax_client = ParallaxOpenAIClient(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE_URL,
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

for output in parallax_client.ichat_completions(list_of_messages, model="gpt-3.5-turbo"):
    print(output)
```

