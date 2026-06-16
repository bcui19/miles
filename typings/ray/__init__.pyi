"""Local stub: treat `ray` as dynamically typed.

The ray package available to the type checker is incomplete/mis-resolved, so
pyright flags core APIs (ray.get, ray.remote, ray.kill, ray.init, ray.util, ...)
as unknown attributes. ray is exercised at runtime, not statically verifiable
here, so expose everything as Any rather than litter call sites with ignores.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
