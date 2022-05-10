Partition Centric Graph Convolutional Networks in PyTorch, PCGCN
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1] and Partition Centric Processing (PCGCN) for Accelerating Graph Convolutional Network [3] using METIS [4] as the partitioning algorithm.

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

This implementation makes use of the Cora dataset from [2], more datasets are expected to be available.

## Requirements

* C compiler that supports the C99 standard (e.g. gcc 7.5.0).
* CMake 2.8 or above.
* Python 3.6.x (rec. 15)
   * NumPy 1.19.5
   * PyTorch 1.2.0
   * SciPy 1.5.4

The dependencies of Python can be easily installed running the following command:
```
python -m pip install -r requirements.txt
```

## Installation

0. Clone the repo
   1. ```$ git clone https://github.com/NicolasMeseguer/pcgcn``` :open_file_folder: Clone the repo to local.
   2. ```$ cd pcgcn``` :point_right: Move to the PCGCN root dir.

1. Install METIS
   1. ```$ cd metis-5.1.0/``` :point_right: Move to the METIS dir.
   2. ```$ make config cc=gcc prefix=../../../metis``` :wrench: Configure METIS passing options to the Makefile.
   3. ```$ make install``` :arrow_down: Installs METIS on pcgcn folder.

2. Compile PCGCN
   1. ```$ cd ..``` :point_right: Move to PCGCN root dir.
   2. ```$ python requirements.py``` :wrench: Quickly check requirements.
   3. ```$ make install``` :arrow_down: Compiles and installs the PCGCN module.

## Usage

  Run the model in two different ways:

  * ```$ make run``` :running: Executes the model with **default parameters** (check the Makefile).
  * ```$ python pcgcn/train.py``` :hand: Manually load the model **without parameters**.

  <details>
    <summary> Click ME to see the allowed parameters! </summary>

    --no-cuda                - Runs the model on the CPU. Currently PCGCN does not work in GPU, WIP.
    --dataset 'name'         - Determines the dataset to be used (default 'cora').
                               Where 'name' can be one ot the following values: 'cora', 'pubmed' and 'citeseer'.
    --epochs X               - Determines the amount of epochs to train the model (default 200).
    --nparts X               - Determines the amount of subgraphs to be generated (default 1). 
    --partition 'algorithm'  - Determines the partition algorithm (default 'random').
                               Where 'algorithm' can be one of the following values: 'metis' and 'random'.
    --sparsity_threshold X   - Determines the max. value of sparsity for edge_blocks (default is 60%).
    --gcn                    - Runs the GCN with the default implementation.

    Example #1:
      python pcgcn/train.py --no-cuda --dataset pubmed --epochs 100 --nparts 8 --sparsity_threshold 80
      (Runs PCGCN on the CPU with 100 epochs, using pubmed; randomly partitions the graph into 8 subgraphs and for those with a sparsity bigger than 80, will use sparse representation.)
  
    Example #2:
      python pcgcn/train.py --no-cuda --dataset cora --partition metis --epochs 50 --nparts 16 --sparsity_threshold 50
      (Runs PCGCN on the CPU with 50 epochs, using cora; using METIS as the partitioning algortithm, partitions the graph into 16 subgraphs and for those with a sparsity bigger than 50, will use sparse representation.)

    Example #3:
      python pcgcn/train.py --gcn --epochs 50
      (Runs the default GCN on the GPU with 50 epochs, using cora. No partitioning is done!)
  </details>

## Uninstall

1. Uninstall METIS
   1. ```$ cd metis-5.1.0/``` :point_right: Move to the METIS dir.
   2. ```$ make uninstall``` :warning: Removes all files installed by 'make install'.
   3. ```$ make distclean``` :recycle: Performs clean and completely removes the build directory.

2. Uninstall PCGCN
   1. ```$ cd /pcgcn``` :point_right: Move to PCGCN root dir.
      - ```$ make clean``` :recycle: Deletes the PCGCN module but keeps the dirs.
      - ```$ make distclean``` :recycle: Performs clean and completely removes PCGCN.

## Notes

1. Make sure to recompile PCGCN, ```$ make install```, whenever you change any module (i.e. layers, model, train, etc).

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

[3] [Tian et al., Partition-Centric Processing for Accelerating Graph Convolutional Network, 2020](https://ieeexplore.ieee.org/document/9139807)

[4] [Karypis & Kumar, A Fast and Highly Quality Multilevel Scheme for Partitioning Irregular Graphs, 1999](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

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
@article{karypis1999metis,
  title={A Fast and Highly Quality Multilevel Scheme for Partitioning Irregular Graphs}, 
  author={George Karypis and Vipin Kumar},
  pages={359-392},
  journal={SIAM Journal on Scientific Computing},
  volume={20},
  number={1},
  year={1999}
}
```

## Bugs & Future Work

1. If a dataset (graph) is NOT undirected, METIS cannot partition it (i.e. graphs in which for each edge (v,u) there is not an edge (u,v)).
2. Converting graph to METIS output (dataset.graph) using sparse representation not dense (faster).
2. Computing of subgraphs (edge_blocks) refactor to sparse (it'll be faster).