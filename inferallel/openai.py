from openai import OpenAI
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Any


class InferallelOpenAI:
    def __init__(
        self, 
        api_key: str = None,
        api_base: str = None,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def _chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        chat_completions_kwargs: Dict[str, Any],
    ):
        outputs = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **chat_completions_kwargs,
        )
        return outputs

    def chat_completions(
        self,
        model: str,
        messages: List[List[Dict[str, str]]]|List[Dict[str, str]],
        chat_completions_kwargs: Dict[str, Any],
        stream: bool = False,
    ):
        if isinstance(messages, list) and isinstance(messages[0], list):
            batch_messages = messages
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            batch_messages = [messages]
        else:
            raise ValueError("messages must be a list of list of dict or list of dict")

        partial_chat_completions = partial(
            self._chat_completions,
            model=model,
            chat_completions_kwargs=chat_completions_kwargs,
        )

        with Pool() as pool:
            if stream:
                for output in pool.imap(
                    partial_chat_completions,
                    batch_messages,
                    chunksize=1,
                ):
                    yield output
            else:
                outputs = pool.map(
                    partial_chat_completions,
                    batch_messages,
                    chunksize=1,
                )
                return outputs


if __name__ == "__main__":
    openai = InferallelOpenAI()
    