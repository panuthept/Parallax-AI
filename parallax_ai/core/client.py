import ray
import json
import requests
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from functools import partial
from multiprocessing import Pool
from typing import Optional, Union, List, Dict


def completions_wrapper(
    func, 
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
    
    try:
        response = func(
            input=input,
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs,
        )
        return index, response
    except Exception as e:
        print(e)
        return index, None


def openai_completions(
    input,
    api_key,
    base_url,
    model,
    **kwargs,
):
    client = OpenAI(api_key=api_key, base_url=base_url)

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
    return response


# def batched_openai_completions(
#     batch_inputs,
#     api_key,
#     base_url,
#     model,
#     **kwargs,
# ):
#     """ Preventing too fine-grained parallelization """
#     return [openai_completions(inputs, api_key, base_url, model, **kwargs) for inputs in batch_inputs]


# def wrapped_openai_completions(
#     inputs,
#     **kwargs,
# ):
#     index, input, api_key, base_url, model = inputs
#     return openai_completions((index, input), api_key, base_url, model, **kwargs)


class Client:
    def __init__(
        self, 
        api_key: str = "EMPTY",
        base_url: Optional[Union[List[str],str]] = None,
        completions_func: Optional[callable] = None,
        model_remote_address: Optional[Dict[str, List[dict]]] = None,
        ray_remote_address: Optional[str] = None,
        ray_local_workers: Optional[int] = None,
        local_workers: Optional[int] = None,
        proportions: Optional[List[float]] = None,
        chunk_size: Optional[int] = 6000,   # Maximum requests to send in each batch
        max_tokens: int = 2048,
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

        self.completions_func = openai_completions if completions_func is None else completions_func
        self.model_remote_address = model_remote_address
        self.proportions = proportions
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens

        self.pool = None
        if ray_remote_address is not None or ray_local_workers is not None:
            if ray.is_initialized():
                ray.shutdown()
            try:
                if ray_remote_address is not None:
                    server_info = ray.init(address=f"ray://{ray_remote_address}:10001", **kwargs)
                elif ray_local_workers is not None:
                    server_info = ray.init(num_cpus=ray_local_workers, **kwargs) 
                print(f"Ray initialized:\n{server_info}")
            except:
                print("Fail to initialize Ray, no parallelization method is used.")
        else:
            self.pool = Pool(processes=local_workers)
            print("Multiprocessing Pool initialized.")

        # Report which function initialize Client
        import inspect
        stack = inspect.stack()
        if len(stack) > 1:
            print(f"Client initialized from {stack[1].filename}:{stack[1].lineno} in function {stack[1].function}().")

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
        models: List[str],
        **kwargs,
    ):
        assert len(models) == len(inputs), f"Length of models ({len(models)}) should be equal to length of inputs ({len(inputs)})."
        kwargs["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
        kwargs["func"] = self.completions_func
        inputs = self._preprocess_inputs(inputs)
        
        indices = list(range(len(inputs)))

        model_addresses = []
        for model in models:
            cand_addresses = self.model_remote_address.get("any", []) + self.model_remote_address.get(model, [])
            model_addresses.append(np.random.choice(cand_addresses, size=1, p=self.proportions)[0])

        for start_index in range(0, len(inputs), self.chunk_size):
            batched_indices = indices[start_index: start_index + self.chunk_size]
            batched_inputs = inputs[start_index: start_index + self.chunk_size]
            batched_model_addresses = model_addresses[start_index: start_index + self.chunk_size]
            batched_models = models[start_index: start_index + self.chunk_size]

            if ray.is_initialized():
                partial_func = partial(completions_wrapper, **kwargs)

                @ray.remote
                def remote_openai_completions(inputs, api_key, base_url, model):
                    return partial_func(inputs=inputs, api_key=api_key, base_url=base_url, model=model)

                running_tasks = [
                    remote_openai_completions.remote(
                        inputs=(batched_indices[i], batched_inputs[i]),
                        api_key=batched_model_addresses[i]["api_key"],
                        base_url=batched_model_addresses[i]["base_url"],
                        model=batched_models[i],
                    )
                    for i in range(len(batched_inputs))
                ]
                
                while running_tasks:
                    done_tasks, running_tasks = ray.wait(running_tasks, num_returns=1)
                    for task in done_tasks:
                        index, output = ray.get(task)
                        yield (index, output)
            elif self.pool is not None:
                # Prepare partial function for multiprocessing
                partial_func = partial(completions_wrapper, **kwargs)
                # Prepare inputs for multiprocessing
                batched_inputs = [
                    ((batched_indices[i], batched_inputs[i]), batched_model_addresses[i]["api_key"], batched_model_addresses[i]["base_url"], batched_models[i]) 
                    for i in range(len(batched_inputs))
                ]
                for index, output in self.pool.imap_unordered(partial_func, batched_inputs):
                    yield (index, output)
            else:
                for i in range(len(batched_inputs)):
                    yield completions_wrapper(
                        inputs=(batched_indices[i], batched_inputs[i]), 
                        api_key=batched_model_addresses[i]["api_key"], 
                        base_url=batched_model_addresses[i]["base_url"], 
                        model=batched_models[i], 
                        **kwargs
                    )

    def run(
        self,
        inputs,
        model: Union[str, List[str]],
        progress_name: Optional[Union[str, List[Optional[str]]]] = None,
        **kwargs,
    ):
        # Duplicate model if single string is provided
        models = [model] * len(inputs) if isinstance(model, str) else model
        # Progess bar setup
        pbars = {}
        if progress_name is not None:
            # Duplicate progress_name if single string is provided
            progress_names = [progress_name] * len(inputs) if isinstance(progress_name, str) else progress_name
            assert len(progress_names) == len(inputs), f"Length of progress_name ({len(progress_names)}) should be equal to length of inputs ({len(inputs)})."
            for name in set(progress_names):
                if name is not None:
                    pbars[name] = tqdm(total=len([n for n in progress_names if n == name]), desc=name, position=len(pbars))

        # Run
        outputs = []
        for i, output in self._run(inputs=inputs, models=models, **kwargs):
            if progress_name is not None and progress_names[i] is not None:
                pbars[progress_names[i]].update(1)
            outputs.append((i, output))
        
        # Sort outputs to the original order
        outputs = sorted(outputs, key=lambda x: x[0])
        outputs = [output for _, output in outputs]
        return outputs


if __name__ == "__main__":
    def custom_completions(
        input,
        api_key,
        base_url,
        model,
        **kwargs,
    ):
        url = f"{base_url}/chat/completions"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "messages": input,
            "model": model,
            **kwargs,
            "cache": {
                "no-cache": True
            }
        }
        response = requests.post(url, headers=headers, data=json.dumps(data)).json()
        return response

    client = Client(
        ray_remote_address=8,
        completions_func=custom_completions,
        api_key="xxx",
        base_url="xxx",
    )
    response = client.run(
        inputs=[[{"role": "user", "content": "Hello!"}]] * 100,
        model="xxx",
        logprobs=True,
        top_logprobs=5,
    )
    print(response)
    # inputs = ["Sing me a song."] * 1000
    # outputs = client.run(
    #     inputs=inputs,
    #     model="gemma3:1b-it-qat",
    # )

    # response = completions(
    #     inputs=(0, [{"role": "user", "content": "Hello!"}]),
    #     api_key="sk-NgDAhFUD0Y2PfBCKiOSsYA",
    #     base_url="https://dev.api.sea-lion-inference.com/v1",
    #     model="aisingapore/SEA-Guard-V2",
    #     logprobs=True,
    #     top_logprobs=5,
    # )
    # print(response)