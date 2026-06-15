"""NanoRollout client for orchestrators (docs/orchestrator-design.md).

Runs one agent episode against NanoRollout, then — when configured with a
TITO proxy (``inference_url``) — calls ``POST /v1/finalize`` per task id and
returns the durable artifact path. Orchestrators without a proxy (e.g. no
training) simply never configure ``inference_url`` and the finalize call is
skipped. NanoRollout itself stays capture-agnostic.

The run payload mirrors what miles' deleted per-slot integration used to
build, but is driven entirely by orchestrator config + the datapoint — no
miles imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    nanorollout_url: str
    inference_url: str | None = None  # TITO proxy base_url (.../v1); None = no capture/finalize
    model_name: str = "model"
    run_name: str = "orchestrator"
    runner: str = "oh-core"
    task: str = "swe"
    env_type: str = "docker"
    api_timeout_s: float | None = None  # HTTP client timeout; None = no timeout
    sampling_params: dict[str, Any] = field(default_factory=dict)
    task_timeout_s: int = 1800
    step_timeout_s: int = 600
    eval_timeout_s: int = 600
    env_timeout_s: int = 120
    create_timeout_s: int = 600
    max_iterations: int = 100
    use_fn_calling: bool = True


@dataclass
class EpisodeRun:
    task_id: str
    response: dict
    artifact_path: str | None


class RolloutClient:
    def __init__(self, config: RolloutConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(config.api_timeout_s))

    def build_request(
        self,
        datapoint: dict,
        task_id: str,
        rollout_tag: str = "0",
        initial_messages: list[dict] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> dict:
        """Build the stateless NanoRollout /run payload; per-datapoint
        ``metadata`` keys override the config defaults."""
        cfg = self.config
        metadata = dict(datapoint.get("metadata") or {})
        instance_id = str(metadata.get("instance_id") or datapoint.get("instance_id") or task_id)
        prompt = datapoint.get("prompt", "")

        payload = {
            "instance_id": instance_id,
            "task_timeout_s": metadata.get("task_timeout", cfg.task_timeout_s),
            "model_name": cfg.model_name,
            "run_name": f"{cfg.run_name}-rollout-{rollout_tag}",
            "base_url": cfg.inference_url or "",
            "api_key": task_id,
            "env_type": metadata.get("env_type", cfg.env_type),
            "sampling_params": dict(cfg.sampling_params),
            "runtime_env": {"env_vars": {}},
            "runner": metadata.get("runner", cfg.runner),
            "task": metadata.get("task") or metadata.get("task_type") or cfg.task,
            "agent": metadata.get("agent", metadata.get("runner", cfg.runner)),
            "extra_args": {
                "instance_id": instance_id,
                "dataset": metadata.get("dataset", "local"),
                "split": metadata.get("split", "train"),
                "step_timeout": metadata.get("step_timeout", cfg.step_timeout_s),
                "eval_timeout": metadata.get("eval_timeout", cfg.eval_timeout_s),
                "env_timeout": metadata.get("env_timeout", cfg.env_timeout_s),
                "create_timeout": metadata.get("create_timeout", cfg.create_timeout_s),
                "max_iterations": metadata.get("max_iterations", cfg.max_iterations),
                "use_fn_calling": metadata.get("use_fn_calling", cfg.use_fn_calling),
                "prompt": prompt,
                "initial_messages": initial_messages,
                **(datapoint.get("extra_args") or {}),
                **(extra_args or {}),
            },
        }
        return payload

    async def run(
        self,
        datapoint: dict,
        task_id: str,
        rollout_tag: str = "0",
        initial_messages: list[dict] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> dict:
        """One stateless NanoRollout /run; returns the response dict. On HTTP
        failure returns an error-shaped response — the orchestrator decides
        whether to drop or retry."""
        payload = self.build_request(datapoint, task_id, rollout_tag, initial_messages, extra_args)
        try:
            response = await self._client.post(f"{self.config.nanorollout_url.rstrip('/')}/run", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.warning("NanoRollout /run failed for %s: %s", task_id, exc)
            return {"messages": initial_messages or [], "reward": 0.0, "exit_status": "error"}

    async def finalize(self, task_id: str, metadata: dict | None = None) -> str | None:
        """Consolidate the trajectory in the TITO proxy and return the artifact
        path. None when no proxy is configured, when the run produced no
        capture, or on error — the episode then carries no trainable sample."""
        if self.config.inference_url is None:
            return None
        root = self.config.inference_url.rstrip("/").removesuffix("/v1")
        try:
            response = await self._client.post(
                f"{root}/v1/finalize", json={"task_id": task_id, "metadata": metadata or {}}
            )
        except httpx.HTTPError as exc:
            logger.warning("TITO finalize failed for %s: %s", task_id, exc)
            return None
        if response.status_code == 404:
            logger.warning("No TITO capture for %s", task_id)
            return None
        if response.status_code != 200:
            logger.warning("TITO finalize error for %s: %s %s", task_id, response.status_code, response.text)
            return None
        return response.json()["path"]

    async def run_and_finalize(self, datapoint: dict, task_id: str, **kwargs) -> EpisodeRun:
        """One episode end to end; errored runs are not finalized, so their
        ``artifact_path`` is None."""
        response = await self.run(datapoint, task_id, **kwargs)
        artifact_path = None
        if response.get("exit_status") != "error":
            artifact_path = await self.finalize(task_id, metadata={"exit_status": response.get("exit_status")})
        return EpisodeRun(task_id=task_id, response=response, artifact_path=artifact_path)

    async def aclose(self) -> None:
        await self._client.aclose()
