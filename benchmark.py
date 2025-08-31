from parallax import ParallaxOpenAIClient, VanillaOpenAIClient


def main():
    from time import time
    from tqdm import tqdm

    messages = [
        {"role": "user", "content": "Sing me a song."},
    ]
    messagess = [messages for _ in range(1000)]

    model = "google/gemma-3-27b-it"

    # Parallax Client
    parallax_client = ParallaxOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        max_parallel_processes=None,
    )
    
    print("ParallaxOpenAIClient: ichat_completions")
    start_time = time()
    for i, output in enumerate(tqdm(parallax_client.ichat_completions(messagess, model=model))):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"first_output_elapsed_time: {first_output_elapsed_time:.2f}")
    total_elapsed_time = time() - start_time
    print(f"total_elapsed_time: {total_elapsed_time:.2f}")

    print("ParallaxOpenAIClient: ichat_completions_unordered")
    start_time = time()
    for i, (output, index) in enumerate(tqdm(parallax_client.ichat_completions_unordered(messagess, model=model))):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"first_output_elapsed_time: {first_output_elapsed_time:.2f}")
    total_elapsed_time = time() - start_time
    print(f"total_elapsed_time: {total_elapsed_time:.2f}")
    

    # Vanilla Client
    vanilla_client = VanillaOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    
    print("VanillaOpenAIClient:")
    start_time = time()
    for i, output in enumerate(tqdm(vanilla_client.ichat_completions(messagess, model=model))):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"first_output_elapsed_time: {first_output_elapsed_time:.2f}")
    total_elapsed_time = time() - start_time
    print(f"total_elapsed_time: {total_elapsed_time:.2f}")


if __name__ == "__main__":
    main()