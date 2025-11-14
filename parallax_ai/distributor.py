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
    for _ in range(executor_input.get("max_retries", 10)):
        try:
            executor_output = executor_func(executor_input)
            return index, executor_output, True
        except Exception as e:
            error = e
            pass
    print(error)
    return index, None, False

class Distributor:
    def __init__(
        self,
        ray_remote_address: Optional[str] = None,
        ray_local_workers: Optional[int] = None,
        local_workers: Optional[int] = None,
        chunk_size: Optional[int] = 6000,   # Maximum requests to send in each batch
        **kwargs
    ):
        self.ray_remote_address = ray_remote_address
        self.ray_local_workers = ray_local_workers
        self.local_workers = local_workers
        self.chunk_size = chunk_size
        self.pool = None

    def start_engine(self):
        if self.ray_remote_address is not None or self.ray_local_workers is not None:
            if ray.is_initialized():
                ray.shutdown()
            try:
                if self.ray_remote_address is not None:
                    server_info = ray.init(address=f"ray://{self.ray_remote_address}:10001")
                elif self.ray_local_workers is not None:
                    server_info = ray.init(num_cpus=self.ray_local_workers) 
                print(f"Ray initialized:\n{server_info}")
            except:
                self.pool = Pool(max_workers=self.local_workers)
                print("Fail to initialize Ray, switch to ProcessPoolExecutor.")
        else:
            self.pool = Pool(max_workers=self.local_workers)
            # print("ProcessPoolExecutor initialized.")

    def stop_engine(self):
        if ray.is_initialized():
            ray.shutdown()
        if self.pool is not None:
            self.pool.shutdown()

    def execute(self, jobs: List[Job], verbose: bool = False, debug_mode: bool = False):
        for start_index in range(0, len(jobs), self.chunk_size):
            batched_inputs = [
                (index, jobs[index].executor_input, jobs[index].executor_func) 
                for index in range(start_index, min(start_index + self.chunk_size, len(jobs)))
            ]

            if debug_mode:
                for batched_input in batched_inputs:
                    index, output, success = func_wrapper(batched_input)
                    jobs[index].update_execution_result(output, success)
                    yield (index, jobs[index])
            elif ray.is_initialized():
                ray_func_wrapper = ray.remote(func_wrapper)
                running_tasks = [ray_func_wrapper.remote(inp) for inp in batched_inputs] 
                while running_tasks:
                    done, running_tasks = ray.wait(running_tasks)
                    for finished in done:
                        index, output, success = ray.get(finished)
                        jobs[index].update_execution_result(output, success)
                        yield (index, jobs[index])
            else:
                running_tasks = [self.pool.submit(func_wrapper, inp) for inp in batched_inputs]
                for future in as_completed(running_tasks):
                    index, output, success = future.result()
                    jobs[index].update_execution_result(output, success)
                    yield (index, jobs[index])

    def __call__(
        self, 
        jobs: List[Job], 
        verbose: bool = False,
        debug_mode: bool = False,   # If True, disable parallelism for easier debugging
    ):
        self.start_engine()
        
        pbars = {}
        progress_names = set(job.progress_name for job in jobs if job.progress_name is not None)
        for progress_name in set(progress_names):
            pbars[progress_name] = tqdm(total=len([job for job in jobs if job.progress_name == progress_name]), desc=progress_name, position=len(pbars), disable=not verbose)

        completed_jobs = []
        for i, job in self.execute(jobs, debug_mode=debug_mode):
            completed_jobs.append((i, job))
            if job.progress_name in pbars:
                pbars[job.progress_name].update(1)
        completed_jobs = [job for i, job in sorted(completed_jobs, key=lambda x: x[0])]
        
        self.stop_engine()

        return completed_jobs
