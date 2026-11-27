"""
TFQ3 / TFQ4 Python package — standalone codecs, PyTorch/MPS/CUDA linear layers,
hybrid MoE loader, vLLM plugin, and checkpoint evaluator (`tfq3-eval`).
"""
from __future__ import annotations

import os
import sys

__version__ = "3.1.0"

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)


def _import_submodule(pkg_rel: str, checkout_rel_dir: str, mod_name: str):
    """Import from installed `tfq3.*` namespace, falling back to repo checkout directories."""
    import importlib

    try:
        return importlib.import_module(f".{pkg_rel}.{mod_name}", __name__)
    except ImportError:
        cand_dir = os.path.join(_REPO_ROOT, checkout_rel_dir)
        if os.path.isdir(cand_dir) and cand_dir not in sys.path:
            sys.path.insert(0, cand_dir)
        return importlib.import_module(mod_name)


tfq3_reference = _import_submodule("reference", "reference", "tfq3_reference")
tfq4_lattice = _import_submodule("reference", "reference", "tfq4_lattice")


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
    tfq3_eval = _import_submodule("tools", "tools", "tfq3_eval")
    __all__ += ["tfq3_eval", "eval_main"]
except Exception:
    tfq3_eval = None  # type: ignore


def eval_main(argv=None) -> int:
    """Console script entry point for `tfq3-eval` (works for both wheel and editable installs)."""
    if tfq3_eval is None:
        raise RuntimeError("tfq3_eval module failed to import")
    return tfq3_eval.main(argv)

try:
    tfq_mps = _import_submodule("pytorch", os.path.join("integrations", "pytorch"), "tfq_mps")
    tfq_torch = _import_submodule("pytorch", os.path.join("integrations", "pytorch"), "tfq_torch")
    from .pytorch.tfq_torch import Tfq3Linear, Tfq4Linear, load_tfq_into_hf  # type: ignore
    from .pytorch.tfq_mps import mps_kernels_available, tfq3_linear_mps, tfq4_linear_mps  # type: ignore
except ImportError:
    try:
        Tfq3Linear = tfq_torch.Tfq3Linear
        Tfq4Linear = tfq_torch.Tfq4Linear
        load_tfq_into_hf = tfq_torch.load_tfq_into_hf
        mps_kernels_available = tfq_mps.mps_kernels_available
        tfq3_linear_mps = tfq_mps.tfq3_linear_mps
        tfq4_linear_mps = tfq_mps.tfq4_linear_mps
    except Exception:
        pass
except Exception:
    pass

if "tfq_torch" in globals():
    __all__ += [
        "tfq_torch",
        "tfq_mps",
        "Tfq3Linear",
        "Tfq4Linear",
        "load_tfq_into_hf",
        "mps_kernels_available",
        "tfq3_linear_mps",
        "tfq4_linear_mps",
    ]

try:
    tfq_moe = _import_submodule("pytorch", os.path.join("integrations", "pytorch"), "tfq_moe")
    TfpwCheckpoint = tfq_moe.TfpwCheckpoint
    TfpwLayer = tfq_moe.TfpwLayer
    __all__ += ["tfq_moe", "TfpwCheckpoint", "TfpwLayer"]
except Exception:
    pass

try:
    tfq3_quant = _import_submodule("vllm", os.path.join("integrations", "vllm"), "tfq3_quant")
    Tfq3Config = tfq3_quant.Tfq3Config
    Tfq3LinearMethod = tfq3_quant.Tfq3LinearMethod
    Tfq4CheckpointReader = tfq3_quant.Tfq4CheckpointReader
    Tfq4LinearMethod = tfq3_quant.Tfq4LinearMethod
    Tfq3MoeMethod = tfq3_quant.Tfq3MoeMethod
    load_tfq3_checkpoint_weights = tfq3_quant.load_tfq3_checkpoint_weights
    load_tfq3_moe_weights = tfq3_quant.load_tfq3_moe_weights
    __all__ += [
        "tfq3_quant",
        "Tfq3Config",
        "Tfq3LinearMethod",
        "Tfq4CheckpointReader",
        "Tfq4LinearMethod",
        "Tfq3MoeMethod",
        "load_tfq3_checkpoint_weights",
        "load_tfq3_moe_weights",
    ]
except Exception:
    pass
