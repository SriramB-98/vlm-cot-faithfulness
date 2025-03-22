import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

async def gather_async_results_async(output_dict, batch_size):
    keys = []
    coroutines = []
    for desc, results_list in output_dict.items():
        for i, results in enumerate(results_list):
            for j, out in enumerate(results['pred']):
                keys.append((desc, i, j))
                coroutines.append(out)

    # Use a semaphore to limit concurrency to batch_size
    semaphore = asyncio.Semaphore(batch_size)
    
    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro
    
    # Create tasks from coroutines and track indices
    task_to_idx = {}  # Dictionary to map tasks to their original indices
    tasks = []
    for idx, coro in enumerate(coroutines):
        # First wrap the coroutine with the semaphore
        semaphore_coro = run_with_semaphore(coro)
        # Then create a task from it
        task = asyncio.create_task(semaphore_coro)
        task_to_idx[task] = idx
        tasks.append(task)
    
    # Create a progress bar that shows completion
    pending = set(tasks)
    results = [None] * len(tasks)  # Pre-allocate results list
    
    with tqdm(total=len(tasks), desc="Processing requests") as pbar:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                # Get the original index from our mapping
                idx = task_to_idx[task]
                results[idx] = task.result()
                pbar.update(1)
    
    # Map results back to the original keys
    for (desc, i, j), result in zip(keys, results):
        output_dict[desc][i]['pred'][j] = result
    
    return output_dict

def gather_async_results(output_dict, batch_size):
    # This function should only be called from a non-async context
    return asyncio.run(gather_async_results_async(output_dict, batch_size))
