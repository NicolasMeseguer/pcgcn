import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from random import randrange
import numpy as np

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, nparts, bias=True):

        # Step no. 3 (twice --> 2 gcn layers)

        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.nparts = nparts
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # Simulates METIS. Subgraphs are not balanced, one may have more load than another
    def random_partition(self, nvectors, nparts):
        partitions = [[] for x in range(nparts)]
        for i in range(nvectors):
            partitions[randrange(nparts)].append(i)

        return partitions

    def forward(self, input, adj):
        # Step no. 6 (forwarding of the layers)

        #combinacion
        support = torch.mm(input, self.weight)

        # randomly partition the graph into self.nparts
        subgraphs = self.random_partition(int(adj.shape[0]), self.nparts)

        # edge blocks, calculate adj matrix for each and process it (look algorith PCGCN)

        #agregacion
        output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
