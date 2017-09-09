
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os
import shutil

import platform

import numpy as np

if platform.python_compiler()[:2] == 'MS':
    extra_compile_args = ['/openmp', '/Qvec-report:1', '/O2', '/fp:fast']
else:
    extra_compile_args = ['-std=c++11', '-fopenmp', '-O3', '-ftree-vectorize', '-fopt-info-vec-optimized']
                      
extensions = [
    
    Extension("mowgly.containers.vectors", ["mowgly/containers/vectors.pyx"],
        include_dirs = [np.get_include()],
        extra_compile_args = extra_compile_args,
    ),   
    Extension("mowgly.sampling.random_choice", ["mowgly/sampling/random_choice.pyx"],
        include_dirs = [np.get_include()],
        extra_compile_args = extra_compile_args,
    ),   
]

  
setup(
    name = "Mowgly",
    packages=[
              'mowgly',
              ],
    ext_modules = cythonize(extensions),
)

#clean up
for ext in extensions:
    for fpath in ext.sources:
        if '.pyx' in fpath:
            os.remove(fpath.replace('.pyx', '.cpp'))
shutil.rmtree('build')
