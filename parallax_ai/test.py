from parallax_ai import ParallaxOpenAIClient, VanillaOpenAIClient


def main():
    from time import time

    messages = [
        {"role": "user", "content": "Sing me a song."},
    ]
    messagess = [messages for _ in range(500)]

    model = "google/gemma-3-27b-it"

    # Parallax Client
    parallax_client = ParallaxOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        max_parallel_processes=None,
    )
    
    print("ParallaxOpenAIClient.chat_completions:")
    start_time = time()
    for i, output in enumerate(parallax_client.chat_completions(messagess, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"First Output Elapsed Time: {first_output_elapsed_time:.2f}")
    total_elapsed_time = time() - start_time
    print(f"Total Elapsed Time (500 examples): {total_elapsed_time:.2f}")
    
    print("ParallaxOpenAIClient.ichat_completions:")
    start_time = time()
    for i, output in enumerate(parallax_client.ichat_completions(messagess, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"First Output Elapsed Time: {first_output_elapsed_time:.2f}")
    total_elapsed_time = time() - start_time
    print(f"Total Elapsed Time (500 examples): {total_elapsed_time:.2f}")
    

    # Vanilla Client
    vanilla_client = VanillaOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    
    print("Vanilla OpenAI Client:")
    start_time = time()
    for i, output in enumerate(vanilla_client.chat_completions(messagess, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"First Output Elapsed Time: {first_output_elapsed_time:.2f}")
    total_elapsed_time = time() - start_time
    print(f"Total Elapsed Time (500 examples): {total_elapsed_time:.2f}")


if __name__ == "__main__":
    main()