import torch.nn as nn
import torch.nn.functional as F
from pcgcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, subgraphs, edge_blocks, compute_gcn):

        # Step no. 2

        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, subgraphs, edge_blocks, compute_gcn)
        self.gc2 = GraphConvolution(nhid, nclass, subgraphs, edge_blocks, compute_gcn)
        self.dropout = dropout

    def forward(self, x, adj):

        # Step no. 5 (forwarding of the model)
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
