import numpy as np
import scipy.sparse as sp
import torch

# print full size of matrices
np.set_printoptions(threshold=np.inf)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""

    """ 
    CORA Dataset Details. https://paperswithcode.com/dataset/cora
    2708 rows (articulos cientificos), classified into one of 7 classes (última col)
    Each row, has 1433 (size of dict) columns indicating the presence (1) or absence (0) of a word

    The cora.cites file contains 5429 (edges) citations from one paper to another (nodes).
    """

    print('Loading {} dataset...'.format(dataset))

    # extract content (all) from cora (.content) and store it in a str matrix
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # NOTE, matrix accesing [x, y]:
    # x --> row
    # y --> column

    # NOTE, slicing a matrix
    # i.e. https://www.programiz.com/python-programming/matrix#matrix-slicing
    # [:,:] = everything
    # [4, :] = 4th row, all columns
    # [:, 1,-1] = all rows, from columns 1 ... till the end, minus 1
    # [:, -1] = all rows, and print last column
        # NOTE, nth column is = -1... nth-1 is equal to -2... and so on

    # extracts the features (all the content of the matrix) and represent it in CSR
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # extracts the final column with the labels (last column)
    # and represents it in a onehot vector
    labels = encode_onehot(idx_features_labels[:, -1])

    # extract the indices (column 0)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # --- start building the graph ---

    # index the indices
    idx_map = {j: i for i, j in enumerate(idx)}

    # extract content (all) from cora (.cites) and store it in a np (int) matrix
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    # converts the cora.cites so it uses the new indexes (the ones that idx_map assigned)
    # map: it maps the function dict.get() using as input the variable edges_unordered.flatten()
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    """ 
        Until now; what we have.

        features    --> CSR Format (scipy sparse csr matrix)
        labels      --> onehot-encoding format (numpy array)
        edges:      --> updated with the idx_map (numpy array)

        explanation:    (cora.content) the paper id (column 0) indexed in idx_map, so we have node 0(row0col0), node 1(row1col0), node n(rowncol0)
                        once done, we associate it with the raw edges, and update them so it points to the new references (new indexes)
    """

    # transform the edge 2col array into a COO sparse matrix
    # construct from three arrays like (each piece of data goes to 1 position (i, j)):
    #       coo_matrix((data, (i, j)), 
    #                   [shape=(M, N)])
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # printing it in this way will show the idx of the papers (cora.cites with the new indexes)
    # print(adj)
    # easier to (adjacency matrix starting from [0][0]):
    # print(adj.toarray())

    # assume this will make a COO matrix symmetric
    # build symmetric adjacency matrix
    # NOTE: why does it make it symmetric?
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # addition of an identity matrix of the same size (npapers*npapers)
    # then normalize it
    # NOTE: why does it sum a identity matrix ? are papers referencing themself ?
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # normalize fetures matrix
    # why normalize? -> it makes the features more consistent with each other, 
    #                   which allows the model to predict outputs 
    #                   more accurately
    features = normalize(features)

    # creates 3 ranges, one for training, another one as values, and a final one for testing
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    """ 
        Until now; what we have.

        adj             --> adjacency matrix normalized (scipy sparse coo matrix)
        features        --> features matrix normalized (scipy sparse csr matrix)
        three ranges    --> train, val & test (numpy arrays)
    """

    # convert matrices and arrays to tensors

    # features csr matrix to float tensor representation (the full matrix)
    features = torch.FloatTensor(np.array(features.todense()))

    # converts labels to a long tensor
    # np.where(labels)[1] is a vector indicating the paper class (0-6) of each paper (there are 2708 rows) 
    labels = torch.LongTensor(np.where(labels)[1])

    # convert adjacency (scipy sparse) coo matrix to a (torch sparse) coo tensor
    # print(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # print(adj.todense().numpy())
    # These two prints should be the same

    # creates arrays of length (range)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('{} dataset loaded...'.format(dataset))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
