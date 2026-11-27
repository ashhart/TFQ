import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name="tfq3_cuda_ext",
    version="3.1.0",
    description="PyTorch CUDA extension for TFQ3 (3/4/5-bit dense & routed MoE) and TFQ4-E8P (2.00/2.25 bpw)",
    ext_modules=[
        CUDAExtension(
            name="tfq3_cuda_ext",
            sources=[os.path.join(here, "tfq3_cuda_ext.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
