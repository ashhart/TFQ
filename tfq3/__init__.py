"""
TFQ3 / TFQ4 Python package — standalone codecs, PyTorch/MPS/CUDA linear layers, MoE loader, and vLLM plugin.
"""
from __future__ import annotations

import os

__version__ = "3.0.0"

from .reference import tfq3_reference, tfq4_lattice


def default_pw_v3_tables_path() -> str:
    """Return the path to the bundled canonical PWV3 tables binary."""
    return os.path.join(os.path.dirname(tfq3_reference.__file__), "tables", "pw_v3_tables.bin")


def default_pw_v4_tables_path() -> str:
    """Return the path to the bundled canonical PWV4 E8P tables binary."""
    return os.path.join(os.path.dirname(tfq4_lattice.__file__), "tables", "pw_v4_e8p_tables.bin")


__all__ = [
    "__version__",
    "tfq3_reference",
    "tfq4_lattice",
    "default_pw_v3_tables_path",
    "default_pw_v4_tables_path",
]

try:
    from .pytorch.tfq_torch import Tfq3Linear, Tfq4Linear, load_tfq_into_hf
    from .pytorch.tfq_mps import mps_kernels_available, tfq3_linear_mps, tfq4_linear_mps

    __all__ += [
        "Tfq3Linear",
        "Tfq4Linear",
        "load_tfq_into_hf",
        "mps_kernels_available",
        "tfq3_linear_mps",
        "tfq4_linear_mps",
    ]
except Exception:
    pass

try:
    from .pytorch.tfq_moe import TfpwCheckpoint, TfpwLayer

    __all__ += ["TfpwCheckpoint", "TfpwLayer"]
except Exception:
    pass

try:
    from .vllm.tfq3_quant import Tfq3Config, Tfq3LinearMethod, load_tfq3_checkpoint_weights

    __all__ += ["Tfq3Config", "Tfq3LinearMethod", "load_tfq3_checkpoint_weights"]
except Exception:
    pass
