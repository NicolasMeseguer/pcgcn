import math

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, subgraphs, bias=True):

        # Step no. 3 (twice --> 2 gcn layers)

        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.subgraphs = subgraphs
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

    def split_support_mat(self, features_matrix):
        nparts = len(self.subgraphs)
        features_matrix_np = features_matrix.detach().numpy()
        splitted_features = [[] for x in range(nparts)]

        # Iterate over the number of subgraphs
        for i in range(nparts):
            # Iterate over the vectors of a subgraph
            for j in range(len(self.subgraphs[i])):
                splitted_features[i].append(features_matrix_np[self.subgraphs[i][j],:].tolist())

        return splitted_features

    def compute_edge_block(self, subgraph_k, subgraph_i, adj):
        edge_block = np.zeros((len(subgraph_k), len(subgraph_i)), dtype=float)
        
        for x in range(len(subgraph_k)):
            for y in range(len(subgraph_i)):
                if(adj[subgraph_k[x]][subgraph_i[y]] != 0):
                    edge_block[x][y] = adj[subgraph_k[x]][subgraph_i[y]]

        return torch.FloatTensor(edge_block)

    def sum_mat(self, mat1, mat2, nodes):
        for i in range(len(nodes)):
            mat1[nodes[i]] += mat2[i]
        
        return mat1


    def forward(self, input, adj):
        # Step no. 6 (forwarding of the layers)

        # Basic setup
        adj_numpy = adj.to_dense().numpy()

        # combination
        support = torch.mm(input, self.weight)

        # aggregation

        # 1. Split a of l (support) according to subgraphs
        support_subgraphs = self.split_support_mat(support)

        # 2. Execute subgrahs
        agg_subgraphs = np.zeros((support.shape[0], self.out_features), dtype=np.double)

        # Execute graph propagation for each subgraph
        for k in range(len(self.subgraphs)):
            # Gather and accumulate states from neighbor subgraphs
            for i in range(len(self.subgraphs)):
                # calculate edge block & transform it to torch
                edge_block = self.compute_edge_block(self.subgraphs[k], self.subgraphs[i], adj_numpy)
                accumulation = (torch.mm(edge_block, torch.FloatTensor(support_subgraphs[i]))).numpy()
                # 3. Combine hidden states
                agg_subgraphs = self.sum_mat(agg_subgraphs, accumulation, self.subgraphs[k])
        
        # pcgcn output
        output = torch.from_numpy(agg_subgraphs).float()

        # forward gcn output
        # output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
