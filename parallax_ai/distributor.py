import ray
from tqdm import tqdm
from .dataclasses import Job
from concurrent.futures import as_completed
from typing import Any, Tuple, List, Callable, Optional
from concurrent.futures import ProcessPoolExecutor as Pool


def func_wrapper(
    inputs: Tuple[int, Any, Callable],
) -> Tuple[bool, Any]:
    index, executor_input, executor_func = inputs
    try:
        executor_output = executor_func(executor_input)
        return index, executor_output, True
    except Exception as e:
        return True, None, False

class Distributor:
    def __init__(
        self,
        ray_remote_address: Optional[str] = None,
        ray_local_workers: Optional[int] = None,
        local_workers: Optional[int] = None,
        chunk_size: Optional[int] = 6000,   # Maximum requests to send in each batch
        debug_mode: bool = False,   # If True, disable parallelism for easier debugging
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.debug_mode = debug_mode

        self.pool = None
        if ray_remote_address is not None or ray_local_workers is not None:
            if ray.is_initialized():
                ray.shutdown()
            try:
                if ray_remote_address is not None:
                    server_info = ray.init(address=f"ray://{ray_remote_address}:10001", **kwargs)
                elif ray_local_workers is not None:
                    server_info = ray.init(num_cpus=ray_local_workers, **kwargs) 
                print(f"Ray initialized:\n{server_info}")
            except:
                self.pool = Pool(max_workers=local_workers)
                print("Fail to initialize Ray, switch to ProcessPoolExecutor.")
        else:
            self.pool = Pool(max_workers=local_workers)
            print("ProcessPoolExecutor initialized.")

    def execute(self, jobs: List[Job], verbose: bool = False):
        for start_index in range(0, len(jobs), self.chunk_size):
            batched_inputs = [
                (index, jobs[index].executor_input, jobs[index].executor_func) 
                for index in range(start_index, min(start_index + self.chunk_size, len(jobs)))
            ]

            if self.debug_mode:
                for batched_input in tqdm(batched_inputs, desc="Executing jobs", position=0, disable=not verbose):
                    index, output, success = func_wrapper(batched_input)
                    jobs[index].update_execution_result(output, success)
                    yield (index, jobs[index])
            elif ray.is_initialized():
                ray_func_wrapper = ray.remote(func_wrapper)
                running_tasks = [ray_func_wrapper.remote(inp) for inp in batched_inputs] 
                with tqdm(total=len(running_tasks), desc="Executing jobs", position=0, disable=not verbose) as pbar:
                    while running_tasks:
                        done, running_tasks = ray.wait(running_tasks)
                        for finished in done:
                            index, output, success = ray.get(finished)
                            jobs[index].update_execution_result(output, success)
                            yield (index, jobs[index])
                            pbar.update(1)
            else:
                running_tasks = self.pool.submit(func_wrapper, batched_inputs)
                with tqdm(total=len(running_tasks), desc="Executing jobs", position=0, disable=not verbose) as pbar:
                    for future in as_completed(running_tasks):
                        index, output, success = future.result()
                        jobs[index].update_execution_result(output, success)
                        yield (index, jobs[index])
                        pbar.update(1)

    def __call__(self, jobs: List[Job], verbose: bool = False):
        pbars = {}
        progress_names = set(job.progress_name for job in jobs if job.progress_name is not None)
        for progress_name in set(progress_names):
            pbars[progress_name] = tqdm(total=len([job for job in jobs if job.progress_name == progress_name]), desc=progress_name, position=len(pbars) + 1, disable=not verbose)

        completed_jobs = []
        for i, job in self.execute(jobs):
            completed_jobs.append((i, job))
            if job.progress_name in pbars:
                pbars[job.progress_name].update(1)
        completed_jobs = [job for i, job in sorted(completed_jobs, key=lambda x: x[0])]
        return completed_jobs
