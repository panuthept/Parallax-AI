from openai import OpenAI
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Optional


def chat_completions(
    inputs,
    model: str,
    api_key: str = "EMPTY",
    base_url: Optional[str] = None,
    chat_completions_kwargs: Optional[dict] = None,
):
    if isinstance(inputs, tuple):
        assert len(inputs) == 2, "inputs should be a tuple of (messages, index)."
        messages, index = inputs
    else:
        messages = inputs
        index = None

    if chat_completions_kwargs is None or not isinstance(chat_completions_kwargs, dict):
        chat_completions_kwargs = {}

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **chat_completions_kwargs
        )
        return response, index
    except Exception as e:
        return None, index


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
        chat_completions_kwargs: Optional[dict] = None,
    ):
        if isinstance(messages, list) and isinstance(messages[0], list):
            messages = messages
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        else:
            raise ValueError("messages must be a list of list of dict or list of dict")

        outputs = []
        for message in messages:
            output, _ = chat_completions(
                message,
                model=model,
                api_key=self.api_key,
                base_url=self.base_url,
                chat_completions_kwargs=chat_completions_kwargs,
            )
            outputs.append(output)
        return outputs

    def ichat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        chat_completions_kwargs: Optional[dict] = None,
    ):
        if isinstance(messages, list) and isinstance(messages[0], list):
            messages = messages
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        else:
            raise ValueError("messages must be a list of list of dict or list of dict")

        for message in messages:
            output, _ = chat_completions(
                message,
                model=model,
                api_key=self.api_key,
                base_url=self.base_url,
                chat_completions_kwargs=chat_completions_kwargs,
            )
            yield output


class ParallaxOpenAIClient:
    def __init__(
        self, 
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        max_parallel_processes: int = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_parallel_processes = max_parallel_processes

    def chat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        chat_completions_kwargs: Optional[dict] = None,
    ):
        """
        Parallely process inputs, wait for all to finished, output in order.
        """
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
            chat_completions_kwargs=chat_completions_kwargs,
        )

        with Pool(processes=self.max_parallel_processes) as pool:
            outputs = pool.map(
                partial_chat_completions,
                messages,
            )
            outputs = [response for response, index in outputs]
        return outputs

    def ichat_completions(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        chat_completions_kwargs: Optional[dict] = None,
    ):
        """
        Parallely process inputs, output as soon as one finished, in order.
        """
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
            chat_completions_kwargs=chat_completions_kwargs,
        )

        with Pool(processes=self.max_parallel_processes) as pool:
            for output, _ in pool.imap(partial_chat_completions, messages):
                yield output

    def ichat_completions_unordered(
        self,
        messages: List[List[dict]]|List[dict],
        model: str,
        chat_completions_kwargs: Optional[dict] = None,
    ):
        """
        Parallely process inputs, output as soon as one finished without order.
        """
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
            chat_completions_kwargs=chat_completions_kwargs,
        )

        inputs = [(message, i) for i, message in enumerate(messages)]
        with Pool(processes=self.max_parallel_processes) as pool:
            for output, index in pool.imap_unordered(partial_chat_completions, inputs):
                yield (output, index)


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