Partition Centric Graph Convolutional Networks in PyTorch, PCGCN
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1] and Partition Centric Processing (PCGCN) for Accelerating Graph Convolutional Network [3].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

This implementation makes use of the Cora dataset from [2], more datasets are expected to be available.

## Requirements

  * C compiler that supports the C99 standard (e.g. gcc 7.5.0).
  * CMake 2.8
  * Python 3.6.x (rec. 15)
    * NumPy 1.19.5
    * PyTorch 1.2.0
    * SciPy 1.5.4

## Installation

  1. Install METIS
    i. ```$ cd metis-5.1.0/``` :point_right: Move to the METIS dir.
    ii. ```$ make config cc=gcc prefix=../../../metis``` :wrench: Configure METIS passing options.
    iii. ```$ make install``` :arrow_down: Install METIS.

  2. Compile PCGCN
    i. ```$ cd /pcgcn``` :point_right: Move to PCGCN root dir.
    ii. ```$ python requirements.py``` :wrench: Quickly check requirements.
    iii. ```$ make``` :arrow_down: Compiles and installs the PCGCN module.

## Usage

  * ```$ make run``` :running: Executes the model with default parameters (check the Makefile).

## Uninstall

  1. Uninstall METIS
    i. ```$ cd metis-5.1.0/``` :point_right: Move to the METIS dir.
    ii. ```$ make uninstall``` :warning: Removes all files installed by 'make install'.
    iii. ```$ make distclean``` :recycle: Performs clean and completely removes the build directory.

  2. Uninstall PCGCN
    i. ```$ cd /pcgcn``` :point_right: Move to PCGCN root dir.
      - ```$ make clean``` :recycle: Deletes the PCGCN module but keeps the dirs.
      - ```$ make fullclean``` :recycle: Completely removes PCGCN.

## Notes

  1. Make sure to recompile PCGCN, ```$ make```, whenever you change any module (i.e. layers, model, train, etc).

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

[3] [Tian et al., Partition-Centric Processing for Accelerating Graph Convolutional Network, 2020](https://ieeexplore.ieee.org/document/9139807)

## Cite

Please cite this paper if you use any of this (code) in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
@article{tian2020pcgcn,
  title={PCGCN: Partition-Centric Processing for Accelerating Graph Convolutional Network}, 
  author={Tian, Chao and Ma, Lingxiao and Yang, Zhi and Dai, Yafei},
  pages={936-945},
  doi={10.1109/IPDPS47924.2020.00100},
  year={2020}
}
```
