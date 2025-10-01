import ray
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from functools import partial
from typing import Optional, Union, List, Dict


def openai_completions(
    inputs,
    api_key,
    base_url,
    model,
    **kwargs,
):
    assert isinstance(inputs, tuple) and len(inputs) == 2, "inputs should be a tuple of (index, input)."
    index, input = inputs

    if input is None:
        return index, None

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        if isinstance(input, str):
            response = client.completions.create(
                model=model,
                prompt=input,
                **kwargs
            )
        elif isinstance(input, list) and isinstance(input[0], dict):
            response = client.chat.completions.create(
                model=model,
                messages=input,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown input format:\n{input}")
        return index, response
    except Exception as e:
        print(e)
        return index, None


def batched_openai_completions(
    batch_inputs,
    api_key,
    base_url,
    model,
    **kwargs,
):
    """ Preventing too fine-grained parallelization """
    return [openai_completions(inputs, api_key, base_url, model, **kwargs) for inputs in batch_inputs]


class ParallaxClient:
    def __init__(
        self, 
        api_key: str = "EMPTY",
        base_url: Optional[Union[List[str],str]] = None,
        model_remote_address: Optional[Dict[str, List[dict]]] = None,
        ray_remote_address: Optional[str] = None,
        ray_local_workers: Optional[int] = None,
        proportions: Optional[List[float]] = None,
        chunk_size: Optional[int] = 1,
        **kwargs,
    ):
        if base_url is None:
            base_url = "http://localhost:8000/v1"
        if not isinstance(base_url, list):
            assert isinstance(base_url, str), f"base_url should be a list of strings or a string, but got {type(base_url): {base_url}}"
            base_url = [base_url]

        if model_remote_address is not None:
            assert isinstance(model_remote_address, dict), f"model_remote_address should be a dict, but got {type(model_remote_address): {model_remote_address}}"
            for k, vs in model_remote_address.items():
                if not isinstance(vs, list):
                    model_remote_address[k] = [vs]
                for v in model_remote_address[k]:
                    assert isinstance(v, dict) and "api_key" in v and "base_url" in v, f"Each value in model_remote_address should be a dict with 'api_key' and 'base_url', but got {v}"
        else:
            model_remote_address = {"any": [{"api_key": api_key, "base_url": url} for url in base_url]}

        self.model_remote_address = model_remote_address
        self.proportions = proportions
        self.chunk_size = chunk_size

        if not ray.is_initialized():
            if ray_remote_address is not None:
                ray.init(address=ray_remote_address, **kwargs)
            else:
                ray.init(num_cpus=ray_local_workers, **kwargs) if ray_local_workers is not None else ray.init()
            if 'CPU' in ray.available_resources():
                print(f"Ray detected CPUs: {ray.available_resources()['CPU']}")
            else:
                print("Ray detected no CPUs")

    def _preprocess_inputs(self, inputs):
        # inputs: can be 'str', 'list[dict]', 'list[str]', or 'list[list[dict]]'
        if inputs is None or isinstance(inputs, str):
            # Convert 'str' to 'list[str]'
            inputs = [inputs]
        elif isinstance(inputs, list):
            if isinstance(inputs[0], dict):
                # Convert 'list[dict]' to 'list[list[dict]]'
                inputs = [inputs]
            elif isinstance(inputs[0], str):
                inputs = inputs
            elif isinstance(inputs[0], list) and isinstance(inputs[0][0], dict):
                inputs = inputs
            elif inputs[0] is None:
                inputs = inputs
            else:
                raise ValueError(f"Unknown inputs format:\n{inputs}")
        else:
            raise ValueError(f"Unknown inputs format:\n{inputs}")
        return inputs

    def _run(
        self,
        inputs,
        model: str,
        **kwargs,
    ):
        inputs = self._preprocess_inputs(inputs)
        partial_func = partial(batched_openai_completions, model=model, **kwargs)

        @ray.remote
        def remote_openai_completions(batch_inputs, api_key, base_url):
            return partial_func(batch_inputs=batch_inputs, api_key=api_key, base_url=base_url)

        inputs = [(i, input) for i, input in enumerate(inputs)]
        batch_inputs = [inputs[i:i + self.chunk_size] for i in range(0, len(inputs), self.chunk_size)]

        model_addresses = self.model_remote_address.get("any", []) + self.model_remote_address.get(model, [])
        address_indices = np.random.choice(len(model_addresses), len(batch_inputs), p=self.proportions)
        running_tasks = [
            remote_openai_completions.remote(
                batch_inputs[i], 
                api_key=model_addresses[address_indices[i]]["api_key"],
                base_url=model_addresses[address_indices[i]]["base_url"],
            ) 
            for i in range(len(batch_inputs))
        ]
        
        while running_tasks:
            done_tasks, running_tasks = ray.wait(running_tasks, num_returns=1)
            for task in done_tasks:
                for index, output in ray.get(task):
                    yield (index, output)
    
    def run(
        self,
        inputs,
        model: str,
        verbose: bool = False,
        desc: Optional[str] = None,
        **kwargs,
    ):
        outputs = []
        for i, output in tqdm(self._run(inputs=inputs, model=model, **kwargs), total=len(inputs), disable=not verbose, desc=desc):
            outputs.append((i, output))
        outputs = sorted(outputs, key=lambda x: x[0])
        outputs = [output for _, output in outputs]
        return outputs


if __name__ == "__main__":
    client = ParallaxClient(
        base_url="http://localhost:8888/v1",
    )
    inputs = ["Sing me a song."] * 1000
    outputs = client.run(
        inputs=inputs,
        model="gemma3:1b-it-qat",
        verbose=True
    )