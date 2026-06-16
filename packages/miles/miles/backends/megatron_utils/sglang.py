from typing import Any

# the file to manage all sglang deps in the megatron actor.
#
# NOTE: every public symbol below is re-exported via an unconditional module-level
# assignment from a private `_name` alias. A bare `from sglang... import X` re-export of
# a symbol originating in an unresolved third-party module (sglang is not installed in the
# type-checking env) is NOT seen as a member by importers; binding it to a definite local
# name here makes `from ...sglang import X` resolve.
try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0 as _quant_weight_ue8m0
    from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0 as _transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0 as _should_deepgemm_weight_requant
except ImportError:
    _quant_weight_ue8m0 = None
    _transform_scale_ue8m0 = None
    _should_deepgemm_weight_requant = None

try:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8 as _per_block_cast_to_fp8
except ImportError:
    _per_block_cast_to_fp8 = None

# mxfp8
try:
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize as _mxfp8_group_quantize
except ImportError:
    _mxfp8_group_quantize = None

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions as _monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions as _monkey_patch_torch_reductions

from sglang.srt.utils import MultiprocessingSerializer as _MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket as _FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket as _FlattenedTensorBucket

# Typed Any: these are optional third-party callables/classes (None when sglang is absent);
# callers gate them behind feature flags, so expose them dynamically rather than as `X | None`.
quant_weight_ue8m0: Any = _quant_weight_ue8m0
transform_scale_ue8m0: Any = _transform_scale_ue8m0
should_deepgemm_weight_requant_ue8m0: Any = _should_deepgemm_weight_requant
per_block_cast_to_fp8: Any = _per_block_cast_to_fp8
mxfp8_group_quantize: Any = _mxfp8_group_quantize
monkey_patch_torch_reductions: Any = _monkey_patch_torch_reductions
MultiprocessingSerializer: Any = _MultiprocessingSerializer
FlattenedTensorBucket: Any = _FlattenedTensorBucket

__all__ = [
    "mxfp8_group_quantize",
    "per_block_cast_to_fp8",
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
