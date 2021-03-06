from setuptools import setup
from setuptools import find_packages

setup(name='pcgcn',
      version='0.1',
      description='Partition Centric Graph Convolutional Networks in PyTorch',
      author='Nicolas Meseguer',
      author_email='n.mesegueriborra@um.es',
      download_url='https://github.com/NicolasMeseguer/pcgcn',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      package_data={'pcgcn': ['README.md']},
      packages=find_packages())
