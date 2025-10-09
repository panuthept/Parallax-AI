# Parallax-AI

## Installation

You can install Parallax using pip:

```bash
pip install parallax-ai
```

### Client
```python
from parallax_ai import Client

# Initialize Client with multiple models and load balancing
client = Client(
    # Add custom remote model addresses here
    model_remote_address={
        "openai/gpt-oss-20b": [
            {"api_key": "EMPTY", "base_url": f"http://<YOUR_MODEL_ADDRESS>:8000/v1"},
        ],
        # Load balance across multiple instances of the same model
        "google/gemma-3-27b-it": [
            {"api_key": "EMPTY", "base_url": f"http://<YOUR_MODEL_ADDRESS>:8000/v1"},
            {"api_key": "EMPTY", "base_url": f"http://<YOUR_MODEL_ADDRESS>:8000/v1"},
            {"api_key": "EMPTY", "base_url": f"http://<YOUR_MODEL_ADDRESS>:8000/v1"},
            {"api_key": "EMPTY", "base_url": f"http://<YOUR_MODEL_ADDRESS>:8000/v1"},
        ],
    },
)
```

### Agent
```python
from parallax_ai import Agent
# Create an Agent with structured input and output
agent = Agent(
    model_name="google/gemma-3-27b-it",
    system_prompt="You are a helpful assistant that translates English to French.",
    input_structure={"text": str},
    output_structure={"translation": str},
)
# Run the agent with batch of structured inputs
agent.run([
    {"text": "Hello, how are you?"},
    {"text": "What is your name?"},
    {"text": "Where do you live?"},
])
```

### MultiAgent
```python
from parallax_ai import MultiAgent, AgentIO, Dependency

multi_agent = MultiAgent(
    agents={
        "translator": Agent(
            model_name="google/gemma-3-27b-it",
            system_prompt="You are a helpful assistant that translates English to French.",
            input_structure={"text": str},
            output_structure={"translation": str},
        ),
        "summarizer": Agent(
            model_name="openai/gpt-oss-20b",
            system_prompt="You are a helpful assistant that summarizes text.",
            input_structure={"translation": str},
            output_structure={"summary": str},
        ),
    },
    agent_ios={
        "translator": AgentIO(
            dependency=Dependency(external_data=["text"]),
        ),
        "summarizer": AgentIO(
            dependency=Dependency(agent_outputs=["translator"]),
        ),
    },
)
multi_agent.run([
    {"text": "Hello, how are you?"},
    {"text": "What is your name?"},
    {"text": "Where do you live?"},
])
```