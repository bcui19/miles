# Contributing

This is a monorepo. Packages live under `packages/`. `packages/miles` is a **git subtree**
of the upstream `miles` repo. You can edit it directly here like any other code — some of
those changes get pushed back upstream, others stay local to this repo. We also pull updates
from upstream and resolve merge conflicts as they come up.

## Setup

```bash
git clone https://github.com/bcui19/Test-Time-Trainer-Is-All-you-Need
cd Test-Time-Trainer-Is-All-you-Need

# add the upstream remote (one-time)
git remote add miles https://github.com/bcui19/miles.git
```

## Pulling miles from upstream

```bash
git subtree pull --prefix=packages/miles miles main
```

Run from the repo root on a clean working tree. Resolve any merge conflicts as usual, then
commit. (We avoid `--squash` so local miles history stays intact for pushing back upstream.)

## Pushing miles changes upstream

For changes made under `packages/miles` that should go back to upstream miles:

```bash
git subtree push --prefix=packages/miles miles <your-branch>
```

Then open a PR against miles from that branch. Changes that are TITO-specific can just stay
in this repo — no push needed.

## Pre-commit

Run from the repo root (ruff, black, isort, autoflake, plus local bans):

```bash
pip install pre-commit
pre-commit install            # run hooks automatically on every commit
pre-commit run --all-files    # run against the whole tree manually
```

The root `.pre-commit-config.yaml` is a symlink to `packages/miles/.pre-commit-config.yaml`,
so it stays in sync automatically when miles is pulled from upstream.

This is the canonical way to run the hooks — it uses the pinned tool versions from the config
and should match what CI/CD runs (hopefully). Prefer `pre-commit run --all-files` over invoking
ruff/black/isort directly, since ad-hoc runs may pull different tool versions and produce a
diff that the pinned hooks then disagree with.
