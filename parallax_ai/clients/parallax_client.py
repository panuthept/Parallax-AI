import numpy as np
from tqdm import tqdm
from openai import OpenAI
from functools import partial
from multiprocessing import Pool
from typing import Optional, List


def openai_completions(
    inputs,
    **kwargs,
):
    assert isinstance(inputs, tuple) and len(inputs) == 5, "inputs should be a tuple of (index, input, model, api_key, base_url)."
    index, input, model, api_key, base_url = inputs

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


class ParallaxClient:
    def __init__(
        self, 
        api_key: str = "EMPTY",
        base_url: Optional[List[str]|str] = None,
        proportions: Optional[List[float]] = None,
        max_parallel_processes: Optional[int] = None,
    ):
        if base_url is None:
            base_url = "http://localhost:8000/v1"
        if not isinstance(base_url, list):
            assert isinstance(base_url, str), f"base_url should be a list of strings or a string, but got {type(base_url): {base_url}}"
            base_url = [base_url]

        self.api_key = api_key
        self.base_urls = base_url
        self.proportions = proportions
        self.max_parallel_processes = max_parallel_processes

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
        partial_func = partial(openai_completions, **kwargs)
        url_indices = np.random.choice(len(self.base_urls), len(inputs), p=self.proportions)

        inputs = [(i, input, model, self.api_key, self.base_urls[url_index]) for i, (input, url_index) in enumerate(zip(inputs, url_indices))]
        with Pool(processes=self.max_parallel_processes) as pool:
            for index, output in pool.imap_unordered(partial_func, inputs):
                yield (index, output)
    
    def run(
        self,
        inputs,
        model: str,
        verbose: bool = False,
        **kwargs,
    ):
        outputs = []
        for i, output in tqdm(self._run(inputs=inputs, model=model, **kwargs), total=len(inputs), disable=not verbose):
            outputs.append((i, output))
        outputs = sorted(outputs, key=lambda x: x[0])
        outputs = [output for _, output in outputs]
        return outputs
        


if __name__ == "__main__":
    client = ParallaxClient(
        base_url="http://localhost:8000/v1",
    )
    inputs = ["Sing me a song."] * 10000
    client.run(
        inputs=inputs,
        model="google/gemma-3-27b-it",
        verbose=True
    )