"""
Bounded executor for blocking work (PDF parsing, regex-heavy scoring, DB, file I/O).

asyncio.to_thread uses the loop's shared default executor (tiny: cpu+4 threads).
Anything slow that lands there can starve every other to_thread caller, including
DB access. Heavy/untrusted-input work goes through this dedicated pool instead,
with a hard wall-clock limit so one pathological input cannot pin a worker.
"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(
    max_workers=int(os.getenv("BLOCKING_POOL_SIZE", "6")),
    thread_name_prefix="blocking",
)


async def run_blocking(fn, *args, timeout: float | None = 30.0, **kwargs):
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_POOL, lambda: fn(*args, **kwargs))
    if timeout is None:
        return await fut
    return await asyncio.wait_for(fut, timeout=timeout)
