from openai import OpenAI
from functools import partial
from multiprocessing import Pool
from typing import List, Optional


def run(
    inputs,
    model: str,
    api_key: str = "EMPTY",
    base_url: Optional[str] = None,
    **kwargs,
):
    if isinstance(inputs, tuple):
        assert len(inputs) == 2, "inputs should be a tuple of (input, index)."
        index, input = inputs
    else:
        input = inputs
        index = None

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


class ParallaxOpenAIClient:
    def __init__(
        self, 
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        max_parallel_processes: Optional[int] = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_parallel_processes = max_parallel_processes

    def _prepare_run(
        self,
        inputs,
        model: str,
        **kwargs,
    ):
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
            else:
                raise ValueError(f"Unknown inputs format:\n{inputs}")
        else:
            raise ValueError(f"Unknown inputs format:\n{inputs}")

        partial_func = partial(
            run,
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )
        return inputs, partial_func

    def run(
        self,
        inputs,
        model: str,
        **kwargs,
    ):
        inputs, partial_run_func = self._prepare_run(inputs, model, **kwargs)
        with Pool(processes=self.max_parallel_processes) as pool:
            outputs = pool.map(partial_run_func, inputs)
            outputs = [response for _, response in outputs]
        return outputs

    def irun(
        self,
        inputs,
        model: str,
        **kwargs,
    ):
        inputs, partial_run_func = self._prepare_run(inputs, model, **kwargs)
        with Pool(processes=self.max_parallel_processes) as pool:
            for _, output in pool.imap(partial_run_func, inputs):
                yield output

    def irun_unordered(
        self,
        inputs,
        model: str,
        **kwargs,
    ):
        inputs, partial_run_func = self._prepare_run(inputs, model, **kwargs)
        inputs = [(i, input) for i, input in enumerate(inputs)]
        with Pool(processes=self.max_parallel_processes) as pool:
            for index, output in pool.imap_unordered(partial_run_func, inputs):
                yield (index, output)


if __name__ == "__main__":
    from time import time

    model = "google/gemma-3-27b-it"
    inferallel_client = ParallaxOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    messages = [
        {"role": "user", "content": "Sing me a song."},
    ]
    messagess = [messages for _ in range(100)]
    
    start_time = time()
    for i, output in enumerate(inferallel_client.irun(messagess, model=model)):
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}, Output lenght: {len(output.choices[0].message.content)}")