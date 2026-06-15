# Miles + NanoRollout Minimal Integration

This example is the smallest local integration of NanoRollout/OpenHands with Miles.

Miles still owns the trainer, local SGLang rollout engines, weight updates, token/logprob capture, and train-data conversion. NanoRollout owns the external agent/environment loop through `POST /run`.

For local smoke tests, `examples.nanorollout.mock_server` implements the same `/run` shape and simulates an OpenHands worker. It calls `miles.rollout.nanorollout.proxy.TITOProxy` through an OpenAI-compatible chat endpoint, and the proxy forwards to the local Miles SGLang router. No OpenAI cloud model is used.

Run the one-GPU smoke from the host:

```bash
bash packages/miles/scripts/run_miles_nanorollout_smoke.sh
```

For a real NanoRollout service, start `nro serve` and pass:

```bash
NANOROLLOUT_URL=http://host:11000 bash packages/miles/scripts/run_miles_nanorollout_smoke.sh
```
