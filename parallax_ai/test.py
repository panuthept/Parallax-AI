from parallax_ai import ParallaxOpenAIClient
from parallax_ai.clients.openai_client import run


def main():
    from time import time

    prompts = ["Sing me a song." for _ in range(500)]
    model = "google/gemma-3-27b-it"

    # Parallax Client
    parallax_client = ParallaxOpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        max_parallel_processes=None,
    )
    
    print("ParallaxOpenAIClient.run:")
    start_time = time()
    for i, output in enumerate(parallax_client.run(prompts, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"First Output Elapsed Time: {first_output_elapsed_time:.2f}s")
    total_elapsed_time = time() - start_time
    print(f"Total Elapsed Time (500 requires): {total_elapsed_time:.2f}s")
    print()
    
    print("ParallaxOpenAIClient.irun:")
    start_time = time()
    for i, output in enumerate(parallax_client.irun(prompts, model=model)):
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"First Output Elapsed Time: {first_output_elapsed_time:.2f}s")
    total_elapsed_time = time() - start_time
    print(f"Total Elapsed Time (500 requires): {total_elapsed_time:.2f}s")
    print()
    
    # Vanilla Client
    print("Vanilla OpenAI Client:")
    start_time = time()
    for i, prompt in enumerate(prompts):
        output = run(prompt, model=model, api_key="EMPTY", base_url="http://localhost:8000/v1")
        if i == 0:
            first_output_elapsed_time = time() - start_time
            print(f"First Output Elapsed Time: {first_output_elapsed_time:.2f}s")
    total_elapsed_time = time() - start_time
    print(f"Total Elapsed Time (500 requires): {total_elapsed_time:.2f}s")


if __name__ == "__main__":
    main()