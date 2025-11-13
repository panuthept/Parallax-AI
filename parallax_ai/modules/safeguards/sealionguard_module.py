from ..agent_module import ModelSpec
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from .base_module import BaseGuardModule


@dataclass
class SealionGuardModule(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="aisingapore/Llama-Guard-Delta-500k"))
    max_retries: int = 10
    representative_token_index: int = 0
    representative_tokens: dict = field(default_factory=lambda:
        {
            "safe": "Safe",
            "s": "Sensitive",
            "unsafe": "Harmful",
        }
    )

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)

    def get_safeguard_input(self, module_input: dict) -> dict:
        if module_input.get("response") is None:
            task = "prompt_classification"
            prompt = module_input["prompt"]
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the human user's request is safe or unsafe.\nHuman user:{prompt}"}
                ],
                tokenize=False,
                add_generation_prompt=False,
                thinking_mode="off",
            )
        else:
            task = "response_classification"
            prompt = module_input["prompt"]
            response = module_input["response"]
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant's response is safe or unsafe.\nHuman user:{prompt}\nAI assistant:{response}"}
                ],
                tokenize=False,
                add_generation_prompt=False,
                thinking_mode="off",
            )
        prompt = prompt + "<|start_header_id|>model<|end_header_id|>\n\n"
        prompt = prompt[len(self.tokenizer.bos_token):] if prompt.startswith(self.tokenizer.bos_token) else prompt
        return {"prompt": prompt, "task": task}
    
    def get_executor_input(self, module_input: dict) -> dict:
        assert self.worker_nodes is not None, "worker_nodes must be provided for AgentModule."
        assert self.spec.model_name in self.worker_nodes, f"Model addresses for model '{self.spec.model_name}' not found in worker_nodes."

        safeguard_input = self.get_safeguard_input(module_input)
        executor_input = {
            **safeguard_input,
            "representative_tokens": self.representative_tokens,
            "representative_token_index": self.representative_token_index,
            "model": self.spec.model_name,
            "model_addresses": self.worker_nodes[self.spec.model_name],
            "max_retries": self.max_retries,
            "kwargs": {
                "max_tokens": 10,
                "logprobs": 20,
            }
        }
        return executor_input
    
@dataclass
class GemmaSealionGuardModule(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="aisingapore/Gemma-Guard-SEALION-27B-Delta"))
    max_retries: int = 10
    representative_token_index: int = 0
    representative_tokens: dict = field(default_factory=lambda:
        {
            "safe": "Safe",
            "s": "Sensitive",
            "unsafe": "Harmful",
        }
    )

    def get_safeguard_input(self, module_input: dict) -> dict:
        if module_input.get("response") is None:
            task = "prompt_classification"
            prompt = module_input["prompt"]
            messages = [
                {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the human user's request is safe or unsafe.\nHuman user:{prompt}"}
            ]
        else:
            task = "response_classification"
            prompt = module_input["prompt"]
            response = module_input["response"]
            messages = [
                {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant's response is safe or unsafe.\nHuman user:{prompt}\nAI assistant:{response}"}
            ]
        return {"messages": messages, "task": task}