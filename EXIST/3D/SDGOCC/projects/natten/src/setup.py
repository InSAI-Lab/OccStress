"""
Neighborhood Attention Torch Extension (CUDA only) Setup

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import shlex

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _flags_from_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return shlex.split(value)


_EXTRA_COMPILE_ARGS = {
    'cxx': _flags_from_env('NATTEN_CXX_FLAGS'),
    'nvcc': _flags_from_env('NATTEN_NVCC_FLAGS'),
}

setup(
    name='natcuda',
    version='0.11',
    author='Ali Hassani',
    author_email='alih@uoregon.edu',
    description='Neighborhood Attention CUDA Kernel',
    ext_modules=[
        CUDAExtension('nattenav_cuda', [
            'nattenav_cuda.cpp',
            'nattenav_cuda_kernel.cu',
        ], extra_compile_args=_EXTRA_COMPILE_ARGS),
        CUDAExtension('nattenqkrpb_cuda', [
            'nattenqkrpb_cuda.cpp',
            'nattenqkrpb_cuda_kernel.cu',
        ], extra_compile_args=_EXTRA_COMPILE_ARGS),
        CUDAExtension('natten1dav_cuda', [
            'natten1dav_cuda.cpp',
            'natten1dav_cuda_kernel.cu',
        ], extra_compile_args=_EXTRA_COMPILE_ARGS),
        CUDAExtension('natten1dqkrpb_cuda', [
            'natten1dqkrpb_cuda.cpp',
            'natten1dqkrpb_cuda_kernel.cu',
        ], extra_compile_args=_EXTRA_COMPILE_ARGS),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
