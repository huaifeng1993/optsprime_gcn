import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension)

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name='optsprime',
    version='0.0.1',
    author='shen huixiang',
    author_email='shhuixi@qq.com',
    description='Code for wang lexuan',
    long_description=long_description,
    #long_description_content_type='text/markdown',
    url=None,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    ext_modules=[
    ],
    cmdclass={'build_ext': BuildExtension},
)