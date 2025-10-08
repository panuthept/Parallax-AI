from dataclasses import dataclass
from typing import Optional, Callable, List, Any
from parallax_ai.core.client import Client


@dataclass
class Job:
    inp: Any
    model: str
    output: Any = None
    session_id: Optional[str] = None  # Session ID for conversational models
    output_processor: Optional[Callable] = None  # Function to process the output
    progress_name: Optional[str] = None  # Name for progress bar


class ParallaxEngine:
    """
    This class manages the execution of jobs from one or multiple agents with auto-retry function.
    """
    def __init__(
        self, 
        client: Client,
        max_tries: int = 3,
        dismiss_none_output: bool = False,
    ):
        self.client = client
        self.max_tries = max_tries
        self.dismiss_none_output = dismiss_none_output

    def __call__(
        self, 
        jobs: List[Job],
        **kwargs,
    ):
        remaining_job_indices = []
        for i in range(len(jobs)):
            if jobs[i].inp is None:
                jobs[i].output = None
            else:
                remaining_job_indices.append(i)

        for _ in range(self.max_tries):
            # Run client
            outputs = self.client.run(
                inputs=[jobs[true_index].inp for true_index in remaining_job_indices], 
                model=[jobs[true_index].model for true_index in remaining_job_indices], 
                progress_name=[jobs[true_index].progress_name for true_index in remaining_job_indices],
                **kwargs,
            )
            new_remaining_job_indices = []
            for i, output in enumerate(outputs):
                true_index = remaining_job_indices[i]
                # Check output validity and convert output to desired format
                processed_output = jobs[true_index].output_processor(output) if jobs[true_index].output_processor is not None else output
                if processed_output is None:
                    # Not pass
                    new_remaining_job_indices.append(true_index)
                else:
                    # Pass
                    jobs[true_index].output = processed_output
            remaining_job_indices = new_remaining_job_indices
            if len(remaining_job_indices) == 0:
                break
        if self.dismiss_none_output:
            jobs = [job for job in jobs if job.output is not None]
        return jobs