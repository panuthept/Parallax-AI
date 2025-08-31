from parallax import ParallaxOpenAIClient, VanillaOpenAIClient


def main():
    from time import time

    messages = [
        {"role": "user", "content": "Sing me a song."},
    ]
    messagess = [messages for _ in range(1000)]

    model = "google/gemma-3-27b-it"

    # Parallax Client
    parallax_client = ParallaxOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    
    start_time = time()
    for i, output in enumerate(parallax_client.ichat_completions(messagess, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
    total_elapsed_time = time() - start_time
    print("ParallaxOpenAIClient:")
    print(f"first_output_elapsed_time: {first_output_elapsed_time:.2f}")
    print(f"total_elapsed_time: {total_elapsed_time:.2f}")

    # Vanilla Client
    vanilla_client = VanillaOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    
    start_time = time()
    for i, output in enumerate(vanilla_client.ichat_completions(messagess, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
    total_elapsed_time = time() - start_time
    print("VanillaOpenAIClient:")
    print(f"first_output_elapsed_time: {first_output_elapsed_time:.2f}")
    print(f"total_elapsed_time: {total_elapsed_time:.2f}")