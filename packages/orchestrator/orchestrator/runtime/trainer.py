"""Miles train-buffer client (docs/orchestrator-design.md: the train contract).

Wraps ``POST /v1/train`` with the three wait modes. ``train()`` awaits the
ack inline; ``submit()`` schedules it and returns an ``asyncio.Task`` so the
orchestrator can block at arbitrary points. Wait semantics:

    wait="none"     returns once the group is enqueued (ack via nothing — fate
                    unknown until trained; v1 has no polling endpoint)
    wait="train"    returns when the batch containing this group starts training
    wait="weights"  returns after that batch trained and weights synced

Blocking requests are held open across a full train step, so the HTTP client
uses no read timeout.

Optional checkpoint policy: pass ``save_checkpoint`` (a no-arg callable) plus
``checkpoint_every_submitted_groups=N`` and the client invokes it right
before every Nth submission — intent-logging at a submit boundary. Saving
orchestrator state remains the orchestrator's responsibility; both v1
example orchestrators run without it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

WAIT_MODES = ("none", "train", "weights")


class TrainSubmitError(RuntimeError):
    pass


class MilesTrainer:
    def __init__(
        self,
        buffer_url: str,
        save_checkpoint: Callable[[], None] | None = None,
        checkpoint_every_submitted_groups: int | None = None,
    ):
        self.buffer_url = buffer_url.rstrip("/")
        self._save_checkpoint = save_checkpoint
        self._checkpoint_every = checkpoint_every_submitted_groups
        self._submitted = 0
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    # ------------------------- manifest helpers -------------------------

    @staticmethod
    def episode(reward: float, artifact_paths: list[str], metadata: dict | None = None) -> dict:
        return {
            "reward": float(reward),
            "artifacts": [
                {"path": path, **({"metadata": dict(metadata)} if metadata else {})} for path in artifact_paths
            ],
        }

    @staticmethod
    def group(episodes: list[dict], metadata: dict | None = None) -> dict:
        return {"episodes": episodes, **({"metadata": metadata} if metadata else {})}

    # ------------------------- submission -------------------------

    async def train(self, group: dict, wait: str = "weights") -> dict[str, Any]:
        if wait not in WAIT_MODES:
            raise ValueError(f"wait must be one of {WAIT_MODES}, got {wait!r}")
        self._submitted += 1
        if (
            self._save_checkpoint is not None
            and self._checkpoint_every
            and self._submitted % self._checkpoint_every == 0
        ):
            self._save_checkpoint()
        response = await self._client.post(f"{self.buffer_url}/v1/train", json={"wait": wait, "group": group})
        if response.status_code != 200:
            raise TrainSubmitError(f"/v1/train rejected the group ({response.status_code}): {response.text}")
        return response.json()

    def submit(self, group: dict, wait: str = "none") -> asyncio.Task:
        return asyncio.create_task(self.train(group, wait=wait))

    async def health(self) -> dict[str, Any]:
        response = await self._client.get(f"{self.buffer_url}/health")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
