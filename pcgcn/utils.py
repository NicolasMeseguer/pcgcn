import numpy as np
import scipy.sparse as sp
import torch
import os.path
import subprocess
import time
import sys
import random

# print full size of matrices
# np.set_printoptions(threshold=np.inf)

# Print useful messages in different colors
class tcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(color, msg):
    print(color + str(msg) + tcolors.ENDC)

def print_color_return(color, msg):
    return color + str(msg) + tcolors.ENDC

# Returns a random name + a suffix depending on the tool used to generate the random dataset (i.e. Graphlaxy or RMAT...)
def random_name():
    animal = get_animals_dic()
    animal = animal[random.randint(0, len(animal)-1)]
    color = get_color_dic()
    color = color[random.randint(0, len(color)-1)]

    return color + '_' + animal

# Returns either .glaxy or .grmat depending on the tool used
def tool_suffix(tool):
    eof = None

    if(tool == "Graphlaxy"):
        eof = '.glaxy'
    elif(tool == "RMAT"):
        eof = '.grmat'
    
    return eof

def dataset_generate(parameters, tool, path):
    print("\tCalling " + print_color_return(tcolors.UNDERLINE, tool) + "...")

    # Transform the input edges
    comma_idx = parameters.index(',')
    # Graphlaxy: param1 edge_size_min, param2 edge_size_max
    # RMAT: param1 n_vertices, param2 n_edges
    param1 = parameters[0:comma_idx]
    param2 = parameters[comma_idx+1:]

    # Prepare the random name
    dataset_name = random_name()

    # Ensure that random name is not already been used
    while(os.path.exists(path + dataset_name + tool_suffix(tool))):
        dataset_name = random_name()

    # Set path for the tools
    path += dataset_name + tool_suffix(tool)
    tool_location = None

    # Graphlaxy tool
    if(tool == "Graphlaxy"):
        # Get user's python bin
        python_bin = sys.executable
        tool_location = '../graphlaxy/'

        # Prepare CLI graphlaxy string
        graphlaxy_parameters = '-f ' + tool_location + ' -s 1 -e ' + param1 + ' ' + param2

        # Complete graphlaxy command
        command = python_bin + ' ' + tool_location + 'GraphlaxyDataGen.py generate ' + graphlaxy_parameters

        # Call graphlaxy
        graphlaxy = subprocess.Popen(command, shell=True, stdout = subprocess.PIPE)
        graphlaxy.wait()
        if(graphlaxy.returncode != 0):
            print_color(tcolors.FAIL, "\tGraphlaxy output was not the one expected.\nExiting now...")
            exit(1)
        
        # Update paths
        current_path = tool_location + 'graphs/RMAT_0.txt'

    # RMAT tool
    elif(tool == "RMAT"):
        # PaRMAT location
        tool_location = '../PaRMAT/Release/PaRMAT'

        if not os.path.isfile(tool_location):
            print_color(tcolors.FAIL, "\tYou MUST compile PaRMT in order to use '--rmat <option>' as the dataset generator.\nExiting now..." )
            exit(1)

        # Prepare CLI PaRMAT parameters
        parmat_parameters = '-sorted -undirected -nVertices ' + param1 + ' -nEdges ' + param2

        # Complete PaRMAT command
        command = './' + tool_location + ' ' + parmat_parameters

        # Call PaRMAT
        parmat = subprocess.Popen(command, shell=True, stdout = subprocess.PIPE)
        parmat.wait()
        if(parmat.returncode != 0):
            print_color(tcolors.FAIL, "\tPaRMAT output was not the one expected.\nERxiting now...")
            exit(1)

        # Update paths
        current_path = 'out.txt'

    # Prepare the command
    command = 'mv ' + current_path + ' ' + path

    # Move the graph
    move_command = subprocess.Popen(command, shell=True, stdout = subprocess.PIPE)
    move_command.wait()

    print_color(tcolors.OKBLUE, "\t" + tool + " has generated the graph:")
    print_color(tcolors.OKGREEN, "\t\t" + dataset_name)

    return dataset_name

def dataset_load(dataset, tool, path):

    # Parameters for easy handling
    dataset_name = dataset
    dataset_path = path
    graph_path = None
    features_path = None
    labels_path = None

    # Parameters to read the graph
    bias_idx = 0

    # Depending on the tool used, change the path of the dataset
    if tool == "Graphlaxy":

        # Sanity check of the dataset name, JUST IN CASE
        if '.glaxy' in dataset:
            dataset_name = dataset.replace(".glaxy", "")

        # Create the paths for each of the files
        graph_path = dataset_path + dataset_name + '.glaxy'
        features_path = dataset_path + dataset_name + '.flaxy'
        labels_path =  dataset_path + dataset_name + '.llaxy'

        # IDx of the vertices start from 1, so, substract 1
        bias_idx = -1

    elif tool == "RMAT":

        # Sanity check of the dataset name, JUST IN CASE
        if '.grmat' in dataset:
            dataset_name = dataset.replace(".grmat", "")

        # Create the paths for each of the files
        graph_path = dataset_path + dataset_name + '.grmat'
        features_path = dataset_path + dataset_name + '.frmat'
        labels_path =  dataset_path + dataset_name + '.lrmat'

    # Check if the graph exists
    if(not os.path.exists(graph_path)):
        print_color(tcolors.FAIL, "\tThe specified dataset: " + dataset_name + " could not be found in " + graph_path + " !\nExiting now...")
        exit(1)
    
    print('\tLoading ' + print_color_return(tcolors.UNDERLINE, dataset_name) + ' dataset...')

    # Read the graph
    with open(graph_path, 'r') as g:
        edges = [[(int(num) + bias_idx) for num in line.split()] for line in g]
        edges = np.array(edges)
        g.seek(0)
        n_edges = len([line.strip("\n") for line in g if line != "\n"])
    
    if(tool == "Graphlaxy"):
        n_vert = int(edges[edges.shape[0]-1, : 1,]+1)
    elif(tool == "RMAT"):
        n_vert  = int(np.amax(edges)) + 1

    # Adj matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n_vert, n_vert),
                            dtype=np.float32)

    # Prepare parameters
    features_size = None
    max_features = None
    labels_size = None
    features = None
    labels = None

    if(not os.path.exists(features_path) or not os.path.exists(labels_path)):
        # Create the files and store them

        # TODO: This parameters are random; at the moment are set in a predefined threshold.
        features_size = random.randint(int(n_vert*0.25), int(n_vert*0.75)) # between 25 - 75% of the vertices
        max_features = random.randint(int(features_size*0.05), int(features_size*0.1)) # between 5 - 10% of the size of the features
        labels_size = random.randint(int(n_vert*0.03), int(n_vert*0.07)) # between 3 - 7% of the vertices

        # Manual adjustment if the datset is too small
        if(max_features < 2):
            max_features = int(features_size*0.5)
        
        if(labels_size < 3):
            labels_size = int(n_vert*0.5)

        # Randomly generate the features
        n_features = int(features_size)
        features = np.empty((n_vert, n_features), dtype=np.float32)

        # Randomly generate the classes
        n_labels = int(labels_size)
        labels = np.empty((n_vert, n_labels), dtype=np.float32)

        # Randomly generate the features and labels
        for n in range(n_vert):
            # Features
            feature_row = np.zeros(n_features)
            feature_row[:random.randint(1, max_features)] = 1
            np.random.shuffle(feature_row)
            features[n] = feature_row

            # Labels
            label_row = np.zeros(n_labels)
            label_row[random.randint(0, n_labels-1)] = 1
            labels[n] = label_row
        
        # Finally store them
        np.savetxt(features_path, features, fmt='%.0f', header=str(features_size) + ' ' + str(max_features))
        np.savetxt(labels_path, labels, fmt='%.0f', header=str(labels_size))

    else:
        # Read the files and retrieve the features and labels
        features = np.loadtxt(features_path)
        labels = np.loadtxt(labels_path)

        # READ the features_size, max_features and labels_size
        features_header = str(open(features_path).readline()).replace('# ', '').rstrip("\n").split(' ', 1)
        features_size = int(features_header[0])
        max_features = int(features_header[1])

        labels_size = int(str(open(labels_path).readline()).replace('# ',''))

    ###################################
    ### NOW COMES THE POSTPROCESS ! ###
    ###################################

    # Make the Adj symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize it
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # convert adjacency (scipy sparse) coo matrix to a (torch sparse) coo tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Features matrix
    features = sp.csr_matrix(features, dtype=np.float32)

    # Normalize it
    features = normalize_features(features)

    # features csr matrix to float tensor representation (the full matrix)
    features = torch.FloatTensor(np.array(features.todense()))

    # converts labels to a long tensor
    labels = torch.LongTensor(np.where(labels)[1])

    # creates 3 ranges, one for training, another one as values, and a final one for testing
    idx_train = range(int(n_vert*0.2)) # 20% for training
    idx_val = range(int(n_vert*0.2)+1, int(n_vert*0.7)) # 20 - 70 as values 
    idx_test = range(int(n_vert*0.7)+1, n_vert) # 70 - 100 for testing

    # creates arrays of length (range)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Print general parameters about the dataset
    print("\t\tVertices  " + print_color_return(tcolors.UNDERLINE, "# " + str(n_vert)))
    print("\t\tEdges  " + print_color_return(tcolors.UNDERLINE, "# " + str(n_edges)))
    print("\t\tFeatures " + print_color_return(tcolors.UNDERLINE, "# " + str(features_size)) + " (max per vector " + print_color_return(tcolors.UNDERLINE, "# " + str(max_features)) + ")")
    print("\t\tLabels " + print_color_return(tcolors.UNDERLINE, "# " + str(labels_size)))

    # print("\tPreparing ranges of edges for labels (train, values and test)...")

    # print("\t\tidx_train " + print_color_return(tcolors.UNDERLINE, "# [0, " + str(int(n_vert*0.2)) + "]"))
    # print("\t\tidx_val " + print_color_return(tcolors.UNDERLINE, "# [" + str(int(n_vert*0.2)+1) + ", " + str(int(n_vert*0.7)) + "]"))
    # print("\t\tidx_test " + print_color_return(tcolors.UNDERLINE, "# [" + str(int(n_vert*0.7)+1) + ", " + str(n_vert) + "]"))
    
    print_color(tcolors.OKGREEN, "\tDone !")

    return adj, features, labels, idx_train, idx_val, idx_test, dataset_name

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# Random partition. Subgraphs are not balanced, one may have more load than another
def random_partition(nvectors, nparts):
    print_color(tcolors.OKCYAN, "\tRandomly partitioning the graph...")
    partitions = [[] for x in range(nparts)]
    for i in range(nvectors):
        partitions[randrange(nparts)].append(i)

    return partitions

# Calls METIS for partitioning the graph
def metis_partition(adj, nparts, dataset, path):

    # Path where the output of METIS is going to be
    graphpath = path + dataset + ".graph"

    # If the dataset is not transformed to METIS then, do it
    if not os.path.isfile(graphpath):
        print_color(tcolors.OKCYAN, "\tConverting to METIS...")

        # Get the vector and edge size
        nvectors = int(adj.shape[0])
        nedges = int(adj._nnz()/2)

        # Indices (sparse)
        indexes_sparse = adj.coalesce().indices().numpy()

        # If the number of edges is odd, remove one
        if(int(adj._nnz()) % 2 != 0):
            indexes_sparse = np.delete(indexes_sparse, 0, 1)
            print_color(tcolors.WARNING, "\tWARNING: The first edge [0][0] will be removed...\n\tNumber of edges is odd.")
        
        content = ""
        linetowrite = ""
        start_val = indexes_sparse[0][0]
        for i in range(indexes_sparse.shape[1]):
            if(indexes_sparse[0][i] > start_val):
                start_val = indexes_sparse[0][i]
                content += linetowrite.rstrip() + "\n"
                linetowrite = ""

            linetowrite += str(indexes_sparse[1][i] + 1) + " "
        
        # Write the last line
        content += linetowrite.rstrip() + "\n"

        graphfile = open(graphpath, "w")
        graphfile.write(str(nvectors) + " " + str(nedges) + "\n")
        graphfile.write(content)
        graphfile.close()
    
    # The file already exists, then call METIS with graphfile and nparts
    metispath = "../metis/bin/gpmetis"

    if not os.path.isfile(metispath):
        print_color(tcolors.FAIL, "\tYou MUST install METIS in order to use 'metis' as the partitioning algorithm.\nExiting now..." )
        exit(1)
    
    print_color(tcolors.OKCYAN, "\tCalling METIS...")

    # Prepare the METIS output
    outputpath = graphpath + ".part." + str(nparts)

    # If the output of METIS already exists, do not call it again.
    if not os.path.exists(outputpath):

        # Prepare the CLI command
        metis_parameters = ""
        command = metispath + ' ' + graphpath + ' ' + str(nparts) + ' ' + metis_parameters

        metis = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE)
        metis.wait()
        if(metis.returncode != 0):
            print_color(tcolors.FAIL, "\tMETIS could not partition the graph.\nExiting now...")
            exit(1)

        if not os.path.isfile(outputpath):
            print_color(tcolors.FAIL, "\tMETIS output not found, even when it was executed...\nExiting now...")
            exit(1)
    else:
        print("\t" + print_color_return(tcolors.UNDERLINE, "Previous output") + " found (" + outputpath + ").\n\t" + print_color_return(tcolors.UNDERLINE, "Delete it") + " to generate a new METIS output.")
            
    # At this point, either the file exists or was already generated
    graphfile = open(outputpath, "r")

    # Dump content of file
    fileDump = []
    for line in graphfile:
        fileDump.append(int(line))
    fileDump = np.array(fileDump)

    graphfile.close()

    partitions = [[] for x in range(nparts)]

    tmpVertex = 0
    for line in fileDump:
        partitions[int(line)].append(tmpVertex)
        tmpVertex += 1

    return partitions

# Computes the edge block given the subgraphs and the adj matrix
# 1. Make the edge_blocks & Calculate the sparsity for each one.
#   2a. Low sparsity, keep the edge_block as it is (dense).
#   2b. Notably high sparsity, then convert it to sparse representation (COO) and store it in the same idx.
def compute_edge_block(subgraphs, adj, sparsity_threshold):

    adj_numpy = adj.to_dense().numpy()

    # Array list to store the edge_blocks.
    edge_block = []
    sparsity_block = []
    connectivity_block = []

    # Iterate over a subgraph
    for k in range(len(subgraphs)):

        # Check subgraphs that are connected
        for i in range(len(subgraphs)):

            # Create a matrix of size (NodesK x NodesI) to store adj values
            sub_edge_block = np.zeros((len(subgraphs[k]), len(subgraphs[i])), dtype=float)

            # Variables to check the sparsity
            n_connections = 0
            vertices_of_sk = len(subgraphs[k])
            vertices_of_si = len(subgraphs[i])

            # Only iterate over the lower triangular
            if not i > k:
                # Iterate over all the nodes of the subgraphs and for those with a value, store them.
                for x in range(len(subgraphs[k])):
                    for y in range(len(subgraphs[i])):
                        if(adj_numpy[subgraphs[k][x]][subgraphs[i][y]] != 0):
                            sub_edge_block[x][y] = adj_numpy[subgraphs[k][x]][subgraphs[i][y]]
                            n_connections += 1

            # Append the subgraph edge block to the array list and the corresponding sparsity
            edge_block.append(sub_edge_block)
            sparsity_block.append( round((float(100) - ((n_connections/(vertices_of_sk*vertices_of_si))*100)), 2) )
            connectivity_block.append(n_connections)
    
    # Iterate over the superior triangular and transpose the lower matrices
    for k in range(len(subgraphs)):
        for i in range(len(subgraphs)):
            if i > k:
                edge_block[(k*int(len(subgraphs)))+i] = edge_block[(i*int(len(subgraphs)))+k].transpose()
                sparsity_block[(k*int(len(subgraphs)))+i] = sparsity_block[(i*int(len(subgraphs)))+k]
                connectivity_block[(k*int(len(subgraphs)))+i] = connectivity_block[(i*int(len(subgraphs)))+k]
    
    print_color(tcolors.OKCYAN, "\tComputing sparsity of edge blocks...")
    for i in range(pow(len(subgraphs),2)):
        # subgraph_k = int(i / len(subgraphs))
        # subgraph_i = i % len(subgraphs)
        # print("Sparsity of [" + str(subgraph_k) + "][" + str(subgraph_i) + "] -> " + str(sparsity_block[i]) + " = " + str(connectivity_block[i]) + "/(" + str(len(subgraphs[subgraph_k])) + "x" + str(len(subgraphs[subgraph_i])) + ").")
        
        # If the sparsity (of edge_block[i]) is bigger than sparsity_threshold, convert the given edge_block to sparse coo or csr representation
        if(sparsity_block[i] > sparsity_threshold ):
            # edge_block[i] = sparse_float_to_coo(torch.FloatTensor(edge_block[i]))
            # edge_block[i] = numpy_to_csr(edge_block[i])
            edge_block[i] = numpy_to_coo(edge_block[i])
        else:
            edge_block[i] = torch.FloatTensor(edge_block[i])

    return edge_block, sparsity_block

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""

    """ 
    CORA Dataset Details. https://paperswithcode.com/dataset/cora
    2708 rows (articulos cientificos), classified into one of 7 classes (última col)
    Each row, has 1433 (size of dict) columns indicating the presence (1) or absence (0) of a word

    The cora.cites file contains 5429 (edges) citations from one paper to another (nodes).
    """

    print('\tLoading ' + print_color_return(tcolors.UNDERLINE, dataset) + ' dataset...')

    if dataset == "cora":
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

    elif dataset == "pubmed" or dataset == "citeseer":
        # Adj matrix
        with open(path + dataset + '_adjacency.txt', 'r') as f:
            matrix_size = f.readline()
            matrix_size = matrix_size.split(" ")

            edges = [[int(num) for num in line.split(',')] for line in f]
            edges = np.array(edges)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(int(matrix_size[0]), int(matrix_size[1])),
                            dtype=np.float32)

        # Features data
        with open(path + dataset + '_features.txt', 'r') as f:
            matrix_size = f.readline()
            matrix_size = matrix_size.split(" ")

            edges = [[int(num) for num in line.split(',')] for line in f]
            edges = np.array(edges)
        
        with open(path + dataset + '_features_data.txt', 'r') as f:
            values = [float(line.rstrip("\n")) for line in f]
            values = np.array(values)

        features = sp.csr_matrix((values, (edges[:, 0], edges[:, 1])),
                                shape=(int(matrix_size[0]), int(matrix_size[1])),
                                dtype=np.float32)

        # Labels data
        with open(path + dataset + '_labels.txt', 'r') as f:
            labels = [[int(num) for num in line.split(',')] for line in f]
            labels = np.array(labels)
    
    else:
        print_color(tcolors.FAIL, "\tThe specified dataset does not exist\nExiting now..." )
        exit(1)

    ###################################
    ### NOW COMES THE POSTPROCESS ! ###
    ###################################

    # addition of an identity matrix of the same size (npapers*npapers)
    # then normalize it
    # NOTE: why does it sum a identity matrix ? are papers referencing themself ?
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # normalize fetures matrix
    # why normalize? -> it makes the features more consistent with each other, 
    #                   which allows the model to predict outputs 
    #                   more accurately
    features = normalize_features(features)

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
    
    print_color(tcolors.OKGREEN, "\tDone !")

    return adj, features, labels, idx_train, idx_val, idx_test, dataset


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

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

def numpy_to_csr(numpy_arr):
    return torch.tensor(numpy_arr, dtype = torch.float64).to_sparse_csr()

def numpy_to_coo(numpy_arr):
    indices = [[] for x in range(2)]
    values = []

    for i in range(numpy_arr.shape[0]):
        for j in range(numpy_arr.shape[1]):
            if(numpy_arr[i][j] != 0):
                indices[0].append(i)
                indices[1].append(j)
                values.append(numpy_arr[i][j])

    return torch.sparse_coo_tensor(indices, values, (numpy_arr.shape[0], numpy_arr.shape[1]))

def sparse_float_to_coo(sparse_float_mx):
    indices = [[] for x in range(2)]
    values = []

    for i in range(sparse_float_mx.shape[0]):
        for j in range(sparse_float_mx.shape[1]):
            if(sparse_float_mx[i][j] != 0):
                indices[0].append(i)
                indices[1].append(j)
                values.append(sparse_float_mx[i][j].item())

    return torch.sparse_coo_tensor(indices, values, (sparse_float_mx.shape[0], sparse_float_mx.shape[1]))

def get_animals_dic():
    return (
        "Aardvark",
        "Albatross",
        "Alligator",
        "Alpaca",
        "Ant",
        "Anteater",
        "Antelope",
        "Ape",
        "Armadillo",
        "Donkey",
        "Baboon",
        "Badger",
        "Barracuda",
        "Bat",
        "Bear",
        "Beaver",
        "Bee",
        "Bison",
        "Boar",
        "Buffalo",
        "Butterfly",
        "Camel",
        "Capybara",
        "Caribou",
        "Cassowary",
        "Cat",
        "Caterpillar",
        "Cattle",
        "Chamois",
        "Cheetah",
        "Chicken",
        "Chimpanzee",
        "Chinchilla",
        "Chough",
        "Clam",
        "Cobra",
        "Cockroach",
        "Cod",
        "Cormorant",
        "Coyote",
        "Crab",
        "Crane",
        "Crocodile",
        "Crow",
        "Curlew",
        "Deer",
        "Dinosaur",
        "Dog",
        "Dogfish",
        "Dolphin",
        "Dotterel",
        "Dove",
        "Dragonfly",
        "Duck",
        "Dugong",
        "Dunlin",
        "Eagle",
        "Echidna",
        "Eel",
        "Eland",
        "Elephant",
        "Elk",
        "Emu",
        "Falcon",
        "Ferret",
        "Finch",
        "Fish",
        "Flamingo",
        "Fly",
        "Fox",
        "Frog",
        "Gaur",
        "Gazelle",
        "Gerbil",
        "Giraffe",
        "Gnat",
        "Gnu",
        "Goat",
        "Goldfinch",
        "Goldfish",
        "Goose",
        "Gorilla",
        "Goshawk",
        "Grasshopper",
        "Grouse",
        "Guanaco",
        "Gull",
        "Hamster",
        "Hare",
        "Hawk",
        "Hedgehog",
        "Heron",
        "Herring",
        "Hippopotamus",
        "Hornet",
        "Horse",
        "Human",
        "Hummingbird",
        "Hyena",
        "Ibex",
        "Ibis",
        "Jackal",
        "Jaguar",
        "Jay",
        "Jellyfish",
        "Kangaroo",
        "Kingfisher",
        "Koala",
        "Kookabura",
        "Kouprey",
        "Kudu",
        "Lapwing",
        "Lark",
        "Lemur",
        "Leopard",
        "Lion",
        "Llama",
        "Lobster",
        "Locust",
        "Loris",
        "Louse",
        "Lyrebird",
        "Magpie",
        "Mallard",
        "Manatee",
        "Mandrill",
        "Mantis",
        "Marten",
        "Meerkat",
        "Mink",
        "Mole",
        "Mongoose",
        "Monkey",
        "Moose",
        "Mosquito",
        "Mouse",
        "Mule",
        "Narwhal",
        "Newt",
        "Nightingale",
        "Octopus",
        "Okapi",
        "Opossum",
        "Oryx",
        "Ostrich",
        "Otter",
        "Owl",
        "Oyster",
        "Panther",
        "Parrot",
        "Partridge",
        "Peafowl",
        "Pelican",
        "Penguin",
        "Pheasant",
        "Pig",
        "Pigeon",
        "Pony",
        "Porcupine",
        "Porpoise",
        "Quail",
        "Quelea",
        "Quetzal",
        "Rabbit",
        "Raccoon",
        "Rail",
        "Ram",
        "Rat",
        "Raven",
        "Reindeer",
        "Rhinoceros",
        "Rook",
        "Salamander",
        "Salmon",
        "Sandpiper",
        "Sardine",
        "Scorpion",
        "Seahorse",
        "Seal",
        "Shark",
        "Sheep",
        "Shrew",
        "Skunk",
        "Snail",
        "Snake",
        "Sparrow",
        "Spider",
        "Spoonbill",
        "Squid",
        "Squirrel",
        "Starling",
        "Stingray",
        "Stinkbug",
        "Stork",
        "Swallow",
        "Swan",
        "Tapir",
        "Tarsier",
        "Termite",
        "Tiger",
        "Toad",
        "Trout",
        "Turkey",
        "Turtle",
        "Viper",
        "Vulture",
        "Wallaby",
        "Walrus",
        "Wasp",
        "Weasel",
        "Whale",
        "Wildcat",
        "Wolf",
        "Wolverine",
        "Wombat",
        "Woodcock",
        "Woodpecker",
        "Worm",
        "Wren",
        "Yak",
        "Zebra"
    )

def get_color_dic():
    return ("Red", "Blue", "Yellow", "Grey", "Black", "Purple", "Orange", "Pink", "Green", "Cyan", "White", "Silver", "Lime", "Teal", "Aqua", "Chocolate", "Gold", "Magenta", "Olive", "Turquoise")