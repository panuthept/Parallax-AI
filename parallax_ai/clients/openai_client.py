from openai import OpenAI
from functools import partial
from multiprocessing import Pool
from typing import List, Optional


def completions(
    inputs,
    model: str,
    api_key: str = "EMPTY",
    base_url: Optional[str] = None,
    **kwargs,
):
    if isinstance(inputs, tuple):
        assert len(inputs) == 2, "inputs should be a tuple of (prompt, index)."
        index, prompt = inputs
    else:
        prompt = inputs
        index = None

    if prompt is None:
        return index, None

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            **kwargs
        )
        return index, response
    except Exception as e:
        print(e)
        return index, None


def chat_completions(
    inputs,
    model: str,
    api_key: str = "EMPTY",
    base_url: Optional[str] = None,
    **kwargs,
):
    if isinstance(inputs, tuple):
        assert len(inputs) == 2, "inputs should be a tuple of (messages, index)."
        index, messages = inputs
    else:
        messages = inputs
        index = None

    if messages is None:
        return index, None

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return index, response
    except Exception as e:
        print(e)
        return index, None


class VanillaOpenAIClient:
    # This one is for baseline comparison
    def __init__(
        self, 
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url

    def chat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        **kwargs,
    ):
        if isinstance(messages, list) and isinstance(messages[0], list):
            messages = messages
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        else:
            raise ValueError("messages must be a list of list of dict or list of dict")

        outputs = []
        for message in messages:
            _, output = chat_completions(
                message,
                model=model,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )
            outputs.append(output)
        return outputs

    def ichat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        **kwargs,
    ):
        if isinstance(messages, list) and isinstance(messages[0], list):
            messages = messages
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        else:
            raise ValueError("messages must be a list of list of dict or list of dict")

        for message in messages:
            _, output = chat_completions(
                message,
                model=model,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )
            yield output


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

    def _prepare_completions(
        self,
        prompts: List[str]|str,
        model: str,
        **kwargs,
    ):
        if isinstance(prompts, list):
            prompts = prompts
        elif isinstance(prompts, str):
            prompts = [prompts]
        else:
            raise ValueError("prompts must be a list of str or str")

        partial_completions = partial(
            completions,
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )
        return prompts, partial_completions

    def completions(
        self,
        prompts: List[str]|str,
        model: str,
        **kwargs,
    ):
        """
        Parallely process inputs, wait for all to finished, output in order.
        """
        prompts, partial_completions = self._prepare_completions(
            prompts=prompts,
            model=model,
            **kwargs,
        )

        with Pool(processes=self.max_parallel_processes) as pool:
            outputs = pool.map(
                partial_completions,
                prompts,
            )
            outputs = [response for _, response in outputs]
        return outputs

    def icompletions(
        self,
        prompts: List[str]|str,
        model: str,
        **kwargs,
    ):
        """
        Parallely process inputs, output as soon as one finished, in order.
        """
        prompts, partial_completions = self._prepare_completions(
            prompts=prompts,
            model=model,
            **kwargs,
        )

        with Pool(processes=self.max_parallel_processes) as pool:
            for _, output in pool.imap(partial_completions, prompts):
                yield output

    def icompletions_unordered(
        self,
        prompts: List[str]|str,
        model: str,
        **kwargs,
    ):
        """
        Parallely process inputs, output as soon as one finished without order.
        """
        prompts, partial_completions = self._prepare_completions(
            prompts=prompts,
            model=model,
            **kwargs,
        )

        inputs = [(i, prompt) for i, prompt in enumerate(prompts)]
        with Pool(processes=self.max_parallel_processes) as pool:
            for index, output in pool.imap_unordered(partial_completions, inputs):
                yield (index, output)

    def _prepare_chat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        **kwargs,
    ):
        if isinstance(messages, list) and isinstance(messages[0], list):
            messages = messages
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        else:
            raise ValueError("messages must be a list of list of dict or list of dict")

        partial_chat_completions = partial(
            chat_completions,
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )
        return messages, partial_chat_completions

    def chat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        **kwargs,
    ):
        """
        Parallely process inputs, wait for all to finished, output in order.
        """
        messages, partial_chat_completions = self._prepare_chat_completions(
            messages=messages,
            model=model,
            **kwargs,
        )

        with Pool(processes=self.max_parallel_processes) as pool:
            outputs = pool.map(
                partial_chat_completions,
                messages,
            )
            outputs = [response for _, response in outputs]
        return outputs

    def ichat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        **kwargs,
    ):
        """
        Parallely process inputs, output as soon as one finished, in order.
        """
        messages, partial_chat_completions = self._prepare_chat_completions(
            messages=messages,
            model=model,
            **kwargs,
        )

        with Pool(processes=self.max_parallel_processes) as pool:
            for _, output in pool.imap(partial_chat_completions, messages):
                yield output

    def ichat_completions_unordered(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        **kwargs,
    ):
        """
        Parallely process inputs, output as soon as one finished without order.
        """
        messages, partial_chat_completions = self._prepare_chat_completions(
            messages=messages,
            model=model,
            **kwargs,
        )

        inputs = [(i, message) for i, message in enumerate(messages)]
        with Pool(processes=self.max_parallel_processes) as pool:
            for index, output in pool.imap_unordered(partial_chat_completions, inputs):
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
    for i, output in enumerate(inferallel_client.ichat_completions(messagess, model=model)):
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}, Output lenght: {len(output.choices[0].message.content)}")