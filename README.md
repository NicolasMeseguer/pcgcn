Partition Centric Graph Convolutional Networks in PyTorch PC(py)GCN
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1] and Partition Centric Processing (PCGCN) for Accelerating Graph Convolutional Network [3].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

This implementation makes use of the Cora dataset from [2] more datasets are expected to be available.

## Installation

```python setup.py install``` :warning: Check requirements !

## Requirements

  * Python 3.6
    * NumPy 1.19.5
    * PyTorch 1.2.0
    * SciPy 1.5.4

## Usage

* ```make``` :wrench: To compile necessaries libraries.
* ```make run``` :running: Executes the model (Makefile --> to modify it).
* ```make clean``` :recycle: Cleans the project folder.

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
  doi={10.1109/IPDPS47924.2020.00100}
  year={2020},
  }
```
