"""TITO training-proxy rollout function: the Miles side of the
orchestrator-driven rollout buffer (docs/orchestrator-design.md).

Loaded through the existing rollout-function path:

    --rollout-function-path miles.rollout.train_service.train_buffer.generate_rollout
    --custom-reward-post-process-path miles.rollout.train_service.advantages.post_process_rewards
    --train-buffer-port 11200
    (no prompt dataset, eval_interval=None, num_rollout set high)

It starts a small HTTP server inside the Miles rollout process:

    POST /v1/train   {wait, group:{metadata?, episodes:[{reward, artifacts:[{path, metadata?}]}]}}
    GET  /health

Groups are buffered FIFO; ``generate(rollout_id)`` blocks until
``rollout_batch_size`` whole groups are available, loads the referenced
artifacts from disk, and returns normal Samples. Groups are atomic — never
split across batches. ``wait="train"``/``wait="weights"`` requests are held
open until ``training_started(rollout_id)`` / ``weights_synced(rollout_id)``
fire for the rollout that consumed the group (forwarded by RolloutManager
from train.py).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# module-level so string annotations (from __future__ import annotations)
# resolve when FastAPI inspects the request handlers
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.utils.metric_utils import compute_statistics, dict_add_prefix

from .artifacts import load_artifact, validate_artifact

if TYPE_CHECKING:
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)

WAIT_MODES = ("none", "train", "weights")

_server: TrainBufferServer | None = None
_server_lock = threading.Lock()


@dataclass
class _SubmittedGroup:
    manifest: dict
    wait: str
    group_index: int
    future: Any = field(default=None, repr=False)  # asyncio future in the server loop
    rollout_id: int | None = None


class _ManifestError(ValueError):
    pass


def _validate_manifest(group: dict) -> None:
    episodes = group.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise _ManifestError("group.episodes must be a non-empty list")
    for ep_idx, episode in enumerate(episodes):
        try:
            float(episode["reward"])
        except (KeyError, TypeError, ValueError):
            raise _ManifestError(f"episodes[{ep_idx}].reward must be a number") from None
        artifacts = episode.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise _ManifestError(f"episodes[{ep_idx}].artifacts must be a non-empty list")
        for art in artifacts:
            path = art.get("path") if isinstance(art, dict) else None
            if not path:
                raise _ManifestError(f"episodes[{ep_idx}] has an artifact without a path")
            try:
                validate_artifact(path)
            except FileNotFoundError as exc:
                raise _ManifestError(str(exc)) from exc


class TrainBufferServer:
    """FIFO group buffer bridging the HTTP submit side (event loop) and the
    Miles trainer side (rollout thread); wait-mode futures resolve on the
    lifecycle acks."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._cond = threading.Condition()
        self._queue: deque[_SubmittedGroup] = deque()
        self._by_rollout: dict[int, list[_SubmittedGroup]] = {}
        self._group_counter = 0
        self._sample_counter = 0
        self._rollout_id: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start()

    # ------------------------- HTTP side -------------------------

    def _start(self) -> None:
        import uvicorn

        outer = self

        @asynccontextmanager
        async def lifespan(app):
            outer._loop = asyncio.get_running_loop()
            yield

        app = FastAPI(lifespan=lifespan)

        @app.post("/v1/train")
        async def train(request: Request) -> JSONResponse:
            body = await request.json()
            wait = body.get("wait", "none")
            if wait not in WAIT_MODES:
                return JSONResponse(content={"error": f"wait must be one of {WAIT_MODES}"}, status_code=400)
            group = body.get("group") or {}
            try:
                _validate_manifest(group)
            except _ManifestError as exc:
                return JSONResponse(content={"error": str(exc)}, status_code=400)

            record = _SubmittedGroup(manifest=group, wait=wait, group_index=-1)
            if wait != "none":
                record.future = asyncio.get_running_loop().create_future()
            with outer._cond:
                record.group_index = outer._group_counter
                outer._group_counter += 1
                outer._queue.append(record)
                outer._cond.notify_all()
            if wait == "none":
                return JSONResponse(content={"ok": True, "group_index": record.group_index})
            payload = await record.future
            return JSONResponse(content=payload)

        @app.get("/health")
        async def health() -> JSONResponse:
            with outer._cond:
                buffered = len(outer._queue)
                rollout_id = outer._rollout_id
            return JSONResponse(content={"ok": True, "buffer_groups": buffered, "rollout_id": rollout_id})

        class _Server(uvicorn.Server):
            def install_signal_handlers(self):
                pass

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._uvicorn = _Server(config)
        thread = threading.Thread(target=self._uvicorn.run, daemon=True, name="tito-train-buffer")
        thread.start()
        deadline = time.time() + 15
        while not self._uvicorn.started:
            if time.time() > deadline:
                raise RuntimeError(f"train buffer server failed to start on {self.host}:{self.port}")
            time.sleep(0.02)
        logger.info("TITO train buffer serving at http://%s:%d", self.host, self.port)

    # ------------------------- trainer side -------------------------

    def pop_groups(self, n: int, rollout_id: int) -> list[_SubmittedGroup]:
        """Block until n whole groups are buffered, pop them FIFO, and
        register them as belonging to ``rollout_id`` for lifecycle acks."""
        with self._cond:
            while len(self._queue) < n:
                self._cond.wait(timeout=1.0)
            popped = [self._queue.popleft() for _ in range(n)]
            for record in popped:
                record.rollout_id = rollout_id
            self._by_rollout[rollout_id] = popped
            self._rollout_id = rollout_id
        return popped

    def training_started(self, rollout_id: int) -> None:
        with self._cond:
            records = list(self._by_rollout.get(rollout_id, []))
        for record in records:
            if record.wait == "train":
                self._resolve(record, {"ok": True, "status": "training", "rollout_id": rollout_id})

    def weights_synced(self, rollout_id: int) -> None:
        with self._cond:
            records = self._by_rollout.pop(rollout_id, [])
        for record in records:
            if record.wait == "weights":
                self._resolve(record, {"ok": True, "status": "trained", "rollout_id": rollout_id})

    def next_sample_index(self) -> int:
        with self._cond:
            index = self._sample_counter
            self._sample_counter += 1
        return index

    def _resolve(self, record: _SubmittedGroup, payload: dict) -> None:
        if record.future is None or record.future.done() or self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            lambda: record.future.set_result(payload) if not record.future.done() else None
        )


def _get_server(args) -> TrainBufferServer:
    global _server
    with _server_lock:
        if _server is None:
            port = getattr(args, "train_buffer_port", None)
            if port is None:
                raise RuntimeError("Set --train-buffer-port for the TITO train buffer.")
            _server = TrainBufferServer(host=getattr(args, "train_buffer_host", "0.0.0.0"), port=port)
    return _server


# ------------------------- sample assembly -------------------------


def _samples_from_group(args, server: TrainBufferServer, record: _SubmittedGroup) -> list[Sample]:
    from miles.utils.types import Sample

    group_metadata = record.manifest.get("metadata") or {}
    samples: list[Sample] = []
    for ep_idx, episode in enumerate(record.manifest["episodes"]):
        reward = float(episode["reward"])
        for art in episode["artifacts"]:
            data = load_artifact(art["path"])
            sidecar_metadata = data.get("sidecar_metadata") or {}  # finalize-time metadata travels with the artifact
            sample = Sample(
                group_index=record.group_index,
                index=server.next_sample_index(),
                tokens=data["tokens"],
                loss_mask=data["loss_mask"],
                rollout_log_probs=data["rollout_log_probs"],
                rollout_routed_experts=data["rollout_routed_experts"],
                response=data["response"],
                response_length=data["response_length"],
                reward=reward,
                status=Sample.Status.COMPLETED,
                metadata={
                    **sidecar_metadata,
                    "episode_index": ep_idx,
                    "artifact_path": art["path"],
                    **(art.get("metadata") or {}),
                    **({"group_metadata": group_metadata} if group_metadata else {}),
                },
            )
            if sample.response_length == 0:
                sample.remove_sample = True
            samples.append(sample)
    return samples


def _make_padding_samples(server: TrainBufferServer, reference: Sample, count: int) -> list[Sample]:
    """Synthetic zero-loss padding for data-parallel alignment. Excluded from
    advantages, loss-masked out, marked metadata.padding."""
    import numpy as np

    from miles.utils.types import Sample

    padding = []
    for _ in range(count):
        sample = Sample(
            group_index=-1,
            index=server.next_sample_index(),
            tokens=[0, 0],
            loss_mask=[0],
            rollout_log_probs=[0.0] if reference.rollout_log_probs is not None else None,
            response_length=1,
            reward=0.0,
            remove_sample=True,
            status=Sample.Status.COMPLETED,
            metadata={"padding": True},
        )
        if reference.rollout_routed_experts is not None:
            _, num_layers, topk = reference.rollout_routed_experts.shape
            sample.rollout_routed_experts = np.full((1, num_layers, topk), fill_value=-1, dtype=np.int32)
        padding.append(sample)
    return padding


# ------------------------- rollout entry points -------------------------


def generate_rollout(args, rollout_id: int, data_source: Any, evaluation: bool = False):
    from miles.rollout.base_types import RolloutFnTrainOutput

    del data_source  # data selection is the orchestrator's job
    if evaluation:
        raise RuntimeError(
            "The TITO train buffer has no eval path; run with eval_interval=None (eval is the orchestrator's job)."
        )

    server = _get_server(args)
    records = server.pop_groups(args.rollout_batch_size, rollout_id)
    groups = [_samples_from_group(args, server, record) for record in records]

    flat_count = sum(len(group) for group in groups)
    pad_multiple = getattr(args, "train_buffer_pad_multiple", None) or 1
    pad_count = (-flat_count) % pad_multiple
    if pad_count:
        reference = groups[0][0]
        groups.append(_make_padding_samples(server, reference, pad_count))

    logger.info(
        "TITO train buffer rollout %d: groups=%d samples=%d padding=%d",
        rollout_id,
        len(records),
        flat_count,
        pad_count,
    )
    metrics = {
        "tito_buffer/groups": float(len(records)),
        "tito_buffer/samples": float(flat_count),
        "tito_buffer/padding": float(pad_count),
    }
    # batch reward distribution — a bad buffer slice (degenerate rewards)
    # shows up here before it shows up as a loss spike
    rewards = [s.reward for group in groups for s in group if not (s.metadata or {}).get("padding")]
    stats = compute_statistics(rewards)
    stats["std"] = float(np.std(rewards))
    metrics |= dict_add_prefix(stats, "tito_buffer/reward/")
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


def training_started(rollout_id: int) -> None:
    if _server is not None:
        _server.training_started(rollout_id)


def weights_synced(rollout_id: int) -> None:
    if _server is not None:
        _server.weights_synced(rollout_id)


def add_arguments(parser):
    existing = {option for action in parser._actions for option in action.option_strings}

    def add_once(*names, **kwargs):
        if names[0] not in existing:
            parser.add_argument(*names, **kwargs)

    add_once("--train-buffer-port", type=int, default=None)
    add_once("--train-buffer-host", type=str, default="0.0.0.0")
    add_once(
        "--train-buffer-pad-multiple",
        type=int,
        default=1,
        help="Pad batch sample count to this multiple (set to the training dp_size) so dynamic batching never trims real samples.",
    )
    add_once(
        "--tito-advantage-mode",
        choices=("grpo", "leave_one_out"),
        default="grpo",
    )
    return parser


generate_rollout.add_arguments = add_arguments
generate_rollout.training_started = training_started
generate_rollout.weights_synced = weights_synced
