import asyncio
from collections import defaultdict
from typing import AsyncIterator, Dict

class ProgressBus:
    def __init__(self) -> None:
        self._channels: Dict[str, asyncio.Queue[str]] = defaultdict(asyncio.Queue)

    async def publish(self, job_id: str, message: str) -> None:
        await self._channels[job_id].put(message.rstrip("\n"))

    async def stream(self, job_id: str) -> AsyncIterator[str]:
        q = self._channels[job_id]
        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            # allow GC if you want to clean up channels after completion
            pass

progress_bus = ProgressBus()
