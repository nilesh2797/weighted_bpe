from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("weighted_bpe_wrapper.weighted_bpe_wrapper", 
              ["weighted_bpe_wrapper/weighted_bpe_wrapper.pyx", "weighted_bpe.cpp"],
              extra_compile_args=['-std=c++11'],
              language='c++')
]

setup(
    name='weighted_bpe',
    version='0.1.1',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules)
)
