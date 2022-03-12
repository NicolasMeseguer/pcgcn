import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, subgraphs):

        # Step no. 2

        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, subgraphs)
        self.gc2 = GraphConvolution(nhid, nclass, subgraphs)
        self.dropout = dropout

    def forward(self, x, adj):

        # Step no. 5 (forwarding of the model)
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
