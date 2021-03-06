Partition Centric Graph Convolutional Networks in PyTorch, PCGCN
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1] and Partition Centric Processing (PCGCN) for Accelerating Graph Convolutional Network [3] using METIS [4] as the partitioning algorithm and a synthetic graph dataset generator [5] + PaRMAT dataset generator [7].

For a high-level introduction to GCNs, see: Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016).

This implementation makes use of the Cora dataset from [2]. Also includes PubMed and citeseer citation network datasets (see `data` folder) in a preprocess format, provided by [6] (https://github.com/kimiyoung/planetoid).

## Requirements

* C compiler that supports the C99 standard (e.g. gcc 7.5.0).
* CMake 2.8 or above.
* Python 3.6.x (rec. 15)
   * NumPy 1.19.5
   * PyTorch 1.2.0
   * SciPy 1.5.4

The dependencies of Python can be easily installed running the following command:
```bash
python -m pip install -r requirements.txt
```

## Installation

0. Clone the repo
   1. `$ git clone https://github.com/NicolasMeseguer/pcgcn` :open_file_folder: Clone the repo to local.
   2. `$ cd pcgcn` :point_right: Move to the PCGCN root dir.

1. Install METIS
   1. `$ cd metis-5.1.0/` :point_right: Move to the METIS dir.
   2. `$ make config cc=gcc prefix=../../../metis` :wrench: Configure METIS passing options to the Makefile.
   3. `$ make install` :arrow_down: Installs METIS on pcgcn folder.

2. Compile PCGCN
   1. `$ cd ..` :point_right: Move back to PCGCN root dir.
   2. `$ python requirements.py` :wrench: Quickly check requirements.
   3. `$ make install` :arrow_down: Compiles and installs the PCGCN module.

## PaRMAT requirements

PaRMAT is a multi-threaded RMAT graph generator. Using PaRMAT, you can create very large undirected RMAT graphs. PaRMAT is designed to exploit multiple threads and to avoid failing when there is not a lot of memory (RAM) available (https://github.com/farkhor/PaRMAT).

To compile PaRMAT, you'll need to have a C++ compiler that supports C++11 features.

3. Install PaRMAT.
   1. `$ make installrmat` :arrow_down: Compiles PaRMAT.

You can now use `--rmat <option>` (have a look at [parameters allowed](#Usage)).

PaRMAT will automatically generate an undirected graph, already sorted and with the specified amount of vertices and edges. Along with the graph file ended with .g(raph)rmat, it will come with two additional files: .f(eatures)rmat and .l(abels)rmat. These two corresponds to random generated values for the features and classes (subsequent runs with this dataset will use the random generated values).

If you still want to use the same datset but with different features and labels, simply remove the .frmat and .lrmat files (they will spawn again with new values!).

## Graphlaxy requirements

Graphlaxy is a tool used to create synthetic graph datasets with an even distribution over a set of metrics (or projection) using 'Nash Bargain Scheme' optimization. 

At this point (if executed the `requirements.py`), and you want to install Graphlaxy you should be able to do so without any problem. I encourage strongly to have a look at the README of Graphlaxy repository (https://github.com/BNN-UPC/graphlaxy/blob/master/README.md).

4. Install the requirements for Graphlaxy
   1. `$ cd graphlaxy` :point_right: Move to the Graphlaxy dir.
   2. `$ python -m pip install -r requirements.txt` :arrow_down: Installs the dependencies to use Graphlaxy.
   3. `$ cd ..` :point_right: Move back to PCGCN root dir.

You can now use `--graphlaxy <option>` (have a look at [parameters allowed](#Usage)).

In case you generate a new graph, along with the graph file ended with .g(raph)laxy, it will come with two additional files: .f(eatures)laxy and .l(abels)laxy. These two corresponds to random generated values for the features and classes (subsequent runs with this dataset will use the random generated values). 

If you still want to use the same datset but with different features and labels, simply remove the .flaxy and .llaxy files (they will spawn again with new values!).

## Usage

  Run the model in two different ways:

  * `$ make run` :running: Executes the model with **default parameters** (have a look at the Makefile).
  * `$ python pcgcn/train.py <parameters>` :hand: Manually load the model **without parameters**.

  <details>
    <summary> Click ME to see the allowed parameters! </summary>

    --no-cuda                - Runs the model on the CPU. Currently PCGCN does not work in GPU, WIP.
    --dataset 'name'         - Determines the dataset to be used (default 'cora').
                               Where 'name' can be one ot the following values: 'cora', 'pubmed' and 'citeseer'.
    --graphlaxy 'option'     - Uses the Graphlaxy dataset generator as the dataset. This parameter discards the --dataset and --rmat.
                               Where 'option' can be one of the follows:
                               'dataset_rand_name' uses a dataset that has already been generated by Graphlaxy (check the name once generated).
                               'edge_size_min,edge_size_max' generates a dataset with the specified edges.
    --rmat 'option'          - Uses RMAT dataset generator. This parameter discards the --dataset and --graphlaxy.
                               Where 'option' can be one of the follows:
                               'dataset_rand_name' uses a dataset that has already been generated by RMAT (check the name once generated).
                               'n_vertices,n_edges' generates a dataset with the specified vertices and edges.
    --epochs X               - Determines the amount of epochs to train the model (default 200).
    --no-epochs              - Disables the print of the epochs (acc, losses, fail, test...).
    --gcn                    - Runs the GCN with the default implementation (parameters below this are discarded).
    --nparts X               - Determines the amount of subgraphs to be generated (default 1). 
    --partition 'algorithm'  - Determines the partition algorithm (default 'random').
                               Where 'algorithm' can be one of the following values: 'metis' and 'random'.
    --sparsity_threshold X   - Determines the max. value of sparsity for edge_blocks (default is 60%).

    Example #1:
      python pcgcn/train.py --no-cuda --dataset pubmed --epochs 100 --nparts 8 --sparsity_threshold 80
      (Runs PCGCN on the CPU with 100 epochs, using pubmed; randomly partitions the graph into 8 subgraphs and for 
      those with a sparsity bigger than 80, will use sparse matrix-mult.)
  
    Example #2:
      python pcgcn/train.py --no-cuda --dataset cora --partition metis --epochs 50 --nparts 16 --sparsity_threshold 50
      (Runs PCGCN on the CPU with 50 epochs, using cora; using METIS as the partitioning algortithm, partitions the graph into 
      16 subgraphs and for those with a sparsity bigger than 50, will use sparse matrix-mult.)

    Example #3:
      python pcgcn/train.py --gcn --epochs 50
      (Runs the default GCN on the GPU with 50 epochs, using cora. No partitioning is done!)
    
    Example #4:
      python train.py --no-cuda --graphlaxy 1000,1000 --epochs 500 --partition metis --nparts 4
      (Runs PCGCN on the CPU with 500 epochs, using a random dataset just generated; using METIS as the partitioning algortithm, partitions the 
      graph into 4 subgraphs. DISCLAIMER: this example will generate a random dataset each run. See Example #5)

    Example #5 (using the dataset generated from Example #4, called Blue_Elephant):
      python train.py --no-cuda --graphlaxy Blue_Elephant --epochs 500 --partition metis --nparts 2 --sparsity_threshold 50
      (Runs PCGCN on the CPU with 500 epochs, using a random dataset that was previoulsy generated; using METIS as the partitioning algortithm, partitions the 
      graph into 2 subgraphs and for those with a sparsity bigger than 50, will use sparse matrix-mult.)

    Example #6:
      python train.py --no-cuda --rmat 1000,7000 --epochs 500 --gcn
      (Runs the default GCN on the GPU with 500 epochs, using a random dataset that was generated using PaRMAT; no partitioning is done!)

    Example #7 (using the dataset generated from Example #6, called Orange_Monkey):
      python train.py --no-cuda --rmat Orange_Monkey --epochs 500 --partition metis --nparts 4 --sparsity_threshold 70
      (Runs PCGCN on the CPU with 500 epochs, using a random dataset that was previoulsy generated; using METIS as the partitioning algortithm, partitions the 
      graph into 4 subgraphs and for those with a sparsity bigger than 70, will use sparse matrix-mult.)
  </details>

## Uninstall

1. Uninstall METIS
   1. `$ cd metis-5.1.0/` :point_right: Move to the METIS dir.
   2. `$ make uninstall` :warning: Removes all files installed.
   3. `$ make distclean` :recycle: Performs clean and completely removes the build directory.

2. Uninstall PCGCN
   1. `$ cd /pcgcn` :point_right: Move to PCGCN root dir.
      - `$ make clean` :recycle: Deletes the PCGCN module but keeps the dirs.
      - `$ make distclean` :recycle: Performs clean and completely removes PCGCN.

3. Uninstall PaRMAT
   1. `$ make uninstallrmat` :warning: Removes all files installed.
  
4. Uninstall Graphlaxy
   1. Graphlaxy has nothing compiled but just a set of modules. In case you want to uninstall the modules:
      1. `$ cd graphlaxy` :point_right: Move to the Graphlaxy dir.
      2. `$ python -m pip uninstall -r requirements.txt` :recycle: Deletes the modules used for Graphlaxy.
      3. `$ cd ..` :point_right: Move back to PCGCN root dir.

## Notes

1. Make sure to recompile PCGCN, `$ make install`, whenever you change any module (i.e. layers, model, train, etc).
2. [Graphlaxy @ requirements.py] Copies `pyconfig.h` from the directory of Python into the `Include/` directory (careful with permissions).
3. If a dataset (graph) is NOT undirected, METIS cannot partition it (i.e. graphs in which for each edge (v,u) there is not an edge (u,v)).
4. The random datasets (PaRMAT and Graphlaxy) have an specific amount of features, max features and classes. These parameters are randomly calculated, have a look [here](/proof_of_concepts/graphs_random_parameters.png). 

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

[3] [Tian et al., Partition-Centric Processing for Accelerating Graph Convolutional Network, 2020](https://ieeexplore.ieee.org/document/9139807)

[4] [Karypis & Kumar, A Fast and Highly Quality Multilevel Scheme for Partitioning Irregular Graphs, 1999](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

[5] [Axel Wassington, Graphlaxy Dataset Generator, 2022](https://github.com/BNN-UPC/graphlaxy)

[6] [Zhilin Yang et al., Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016](https://arxiv.org/abs/1603.08861)

[7] [Farzad Khorasani et al., Scalable SIMD-Efficient Graph Processing on GPUs, PACT 15](https://ieeexplore.ieee.org/document/7429293)

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
@inproceedings{wsvr,
 author = {Khorasani, Farzad and Gupta, Rajiv and Bhuyan, Laxmi N.},
 title = {Scalable SIMD-Efficient Graph Processing on GPUs},
 booktitle = {Proceedings of the 24th International Conference on Parallel Architectures and Compilation Techniques},
 series = {PACT 15},
 pages = {39--50},
 year = {2015}
}
```

## Future Work
- :heavy_check_mark: Refactor the function that converts a graph into the METIS input (using adj. sparse representation).
- :heavy_check_mark: Computing of subgraphs (edge_blocks) refactor to sparse (it'll be faster).
- :x: Adjust the randomness of features_size, max_features and labels_size for big graphs ([this](/proof_of_concepts/graphs_random_parameters.png)).