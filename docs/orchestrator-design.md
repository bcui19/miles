# Orchestrator-Driven Training: Design

## The orchestrator is the boss

One orchestrator per run, and it is the **top-level program**. It owns the
data (dataset, curriculum, or self-generated), its internal state, its
concurrency, and termination. Miles and NanoRollout are passive services it
calls. Classic iid GRPO is not special: it's just an orchestrator that loops
over a dataset (`orchestrator/dataset_grpo.py`).

Miles never picks datapoints, never calls into orchestrator code, and has no
opinion about when the run ends (set `num_rollout` high; the run ends when
the orchestrator stops and you stop Miles).

## Services start before the orchestrator, outside it

Miles (trainer + train buffer), the TITO proxy, and NanoRollout are launched
externally (scripts / job manager), each on a fixed port. The orchestrator
just connects to URLs. Reason: the services are heavy (GPUs, ray, containers)
and the orchestrator is the thing you restart twenty times a day — crashing
or editing it must never tear down the trainer.

## Disk is the data plane, HTTP is control plane only

Token-level training data **never travels through an HTTP API**. The TITO
proxy records the exact tokens of every trajectory (agents talk to it as
their OpenAI endpoint); when the orchestrator calls `POST /v1/finalize
{task_id}`, the proxy consolidates the trajectory and writes one binary
artifact (`.npz` + small json sidecar) to the shared filesystem, returning
only the path. Train requests reference these paths. There is no API that
returns token payloads — by construction, not convention. Consequence: Miles
and the proxy must share a filesystem (e.g. Lustre).

## The train contract

`POST /v1/train` accepts **one GRPO group per request** — groups are the only
unit, because advantage normalization needs the whole group. Groups are
atomic: never split across training batches. The manifest is minimal —
identity is structural, no required ids:

```jsonc
{ "wait": "weights",
  "group": { "episodes": [
      { "reward": 1.0, "artifacts": [{ "path": ".../seg0.npz" }, { "path": ".../seg1.npz" }] },
      { "reward": 0.0, "artifacts": [{ "path": ".../ep1.npz" }] } ] } }
```

One `episodes[]` entry = one trajectory; it may hold many artifacts. In the
compaction case, each artifact is one segment: a NanoRollout run up to the
compaction threshold, finalized before the next segment starts. All segments in
an episode share the episode reward. The `wait` flag is the whole sync/async
story:

| `wait` | returns when | training style |
|---|---|---|
| `none` | group enqueued | fully async |
| `train` | its batch starts training | one-off policy |
| `weights` | its batch trained + weights synced | fully sync / on-policy |

The ack *is* the HTTP response returning. Miles only acks groups it actually
trained on; an unacked group's fate is unknown (crash) — resubmit or drop,
orchestrator's choice.

## Deliberately not handled

- **No backpressure.** The buffer is unbounded; blocking wait modes are
  self-limiting, fully-async orchestrators self-pace. Miles stays dumb.
- **No dedup.** Submitting the same manifest twice trains it twice.
- **No eval in Miles.** Eval is the orchestrator's job (`eval_interval=None`).
- **Deadlock rule:** with blocking wait modes, keep at least
  `rollout_batch_size` submissions in flight, or nothing ever trains.

## Advantages are computed in Miles, from rewards

The wire carries **rewards only**, one per episode. Miles centers (GRPO)
across the episodes within each group and broadcasts each episode's advantage
to all of its samples — so a 3-segment compaction trajectory gets one
advantage. Override via the existing reward post-process hook.

## What stayed simple on purpose

- Miles' classic `train.py` loop is untouched except two notifications
  (`training_started`, `weights_synced`) that wake blocking submitters. The
  buffer is just a rollout function — no second trainer.
- NanoRollout is completely capture-agnostic: zero changes; it never learns
  whether its `base_url` is a TITO proxy. Orchestrators that don't train
  (e.g. OpenEvolve) simply never call finalize.
- Colocation: `wait="weights"` works with today's colocated + offload setup;
  `train`/`none` generate during training and need resident rollout engines.

## Pointers

Code: Miles side `miles/rollout/train_service/`, proxy
`miles/rollout/nanorollout/proxy/`, orchestrator side
`orchestrator/runtime/`, examples
`orchestrator/{dataset_grpo,compaction_grpo}.py`.
