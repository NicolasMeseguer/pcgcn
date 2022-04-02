from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pcgcn.utils import load_data, accuracy, random_partition, metis_partition, compute_edge_block, tcolors, print_color
from pcgcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nparts', type=int, default=1,
                    help='Number of subgraphs.')
parser.add_argument('--sparsity_threshold', type=float, default=60.00,
                    help='Determines the maximum sparsity for each edge block.')
parser.add_argument('--gcn', action='store_true', default=False,
                    help='Execute using GCN.')
parser.add_argument('--partition', type=str, default="",
                    help='Determines the partition algorithm')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# CUDA Available
if not args.cuda:
    print("Running WITHOUT CUDA, is it Available ?", end =" ")
    if torch.cuda.is_available():
        print_color(tcolors.OKGREEN, torch.cuda.is_available())
    else:
        print_color(tcolors.FAIL, "False")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, datasetname = load_data()

# Start partitioning the graph
subgraphs = None
edge_blocks = None
sparsity_blocks = None
if not args.gcn:
    print("Partitioning graph...")

    # partitions the graph into args.nparts
    if args.partition.lower() == "metis":
        subgraphs = metis_partition(adj, args.nparts, datasetname)
    else:
        subgraphs = random_partition(int(adj.shape[0]), args.nparts)
    print_color(tcolors.OKGREEN, "\tDone !")

    # based on the subgraphs and the adj matrix, get the edgeblocks (the edge_blocks representation can be either dense (float tensor) or sparse (coo tensor)).
    print_color(tcolors.OKCYAN, "\tComputing edge blocks...")
    edge_blocks, sparsity_blocks = compute_edge_block(subgraphs, adj, args.sparsity_threshold)
    print_color(tcolors.OKGREEN, "\tDone !")

# Model and optimizer
# Step no. 1
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
			subgraphs=subgraphs,
            edge_blocks=edge_blocks,
            sparsity_blocks=sparsity_blocks,
            sparsity_threshold=args.sparsity_threshold,
            compute_gcn=args.gcn)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda and args.gcn:
    print("-- Running on CUDA --")
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
elif args.cuda and not args.gcn:
    print("-- PCGCN not avaiable on CUDA (yet) --")
    exit(1)
else:
    print("-- Running on CPU --")


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # Step no. 4
    output = model(features, adj)
    # Step no. 7
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # droput = probability of training a given node in a layer.
        # 1.0 means no dropout (will train), and 0.0 means no outputs from the layer.
        model.eval()
        # Step no. 8, fastmode disabled
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print_color(tcolors.OKGREEN, "Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print("-- Now Testing --")
test()
