from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='VOXEL_OP',
    ext_modules=[
        CUDAExtension(
            'VOXEL_OP',
            [
            'src/ops_api.cpp',
            'src/ops.cpp',
            'src/cuda.cu'
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
            # include_dirs=['']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)