from dataclasses import dataclass
from parallax_ai.core import ParallaxClient
from typing import Optional, Callable, Literal, Tuple, List, Dict, Any, get_origin


@dataclass
class Job:
    input: Any
    model: str
    output: Any = None
    output_processor: Optional[Callable] = None  # Function to process the output
    session_id: Optional[str] = None  # Session ID for conversational models
    progress_callback: Optional[Callable[[int, int], None]] = None  # Function to report progress


class ParallaxEngine:
    """
    This class manages the execution of jobs from one or multiple agents with auto-retry function.
    """
    def __init__(
        self, 
        client: ParallaxClient,
        max_tries: int = 3,
    ):
        self.client = client
        self.max_tries = max_tries

    def __call__(
        self, 
        jobs: List[Job],
        **kwargs,
    ):
        remaining_job_indices = []
        for i in range(len(jobs)):
            if jobs[i].input is None:
                jobs[i].output = None
            else:
                remaining_job_indices.append(i)

        for _ in range(self.max_tries):
            # Run client
            outputs = self.client.run(
                inputs=[jobs[true_index].input for true_index in remaining_job_indices], 
                model=[jobs[true_index].model for true_index in remaining_job_indices], 
                **kwargs,
            )
            new_remaining_job_indices = []
            for i, output in enumerate(outputs):
                true_index = remaining_job_indices[i]
                # Check output validity and convert output to desired format
                processed_output = jobs[true_index].output_processor(output)
                if processed_output is None:
                    # Not pass
                    new_remaining_job_indices.append(true_index)
                else:
                    # Pass
                    jobs[true_index].output = processed_output
            remaining_job_indices = new_remaining_job_indices
            if len(remaining_job_indices) == 0:
                break
        return jobs