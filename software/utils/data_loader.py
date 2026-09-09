import os
import warnings
warnings.filterwarnings('ignore')

import urllib.request
import tarfile
import pickle
import numpy as np
import scipy.sparse as sp
import torch

# Base directory of the software folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix with self-loops:
    A_tilde = A + I_N
    D_tilde = sum(A_tilde, axis=1)
    A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
    """
    adj = sp.coo_matrix(adj)
    adj_tilde = adj + sp.eye(adj.shape[0])
    
    rowsum = np.array(adj_tilde.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_coo_tensor(indices, values, shape)

def load_planetoid_dataset(dataset_name="cora", root_dir=None):
    """
    Loads Planetoid dataset (Cora, Citeseer, Pubmed).
    Tries torch_geometric first; if unavailable, uses built-in loader.
    """
    dataset_name = dataset_name.lower()
    if root_dir is None:
        root_dir = DEFAULT_DATA_DIR
    os.makedirs(root_dir, exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from torch_geometric.datasets import Planetoid
            import torch_geometric.transforms as T
            
            dataset = Planetoid(root=root_dir, name=dataset_name.capitalize(), transform=T.NormalizeFeatures())
            data = dataset[0]
            
            num_nodes = data.num_nodes
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            
            edge_index = data.edge_index.numpy()
            adj_raw = sp.coo_matrix(
                (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                shape=(num_nodes, num_nodes),
                dtype=np.float32
            )
            
            adj_raw = adj_raw + adj_raw.T.multiply(adj_raw.T > adj_raw) - adj_raw.multiply(adj_raw.T > adj_raw)
            adj_norm_coo = normalize_adj(adj_raw)
            adj_norm_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm_coo)
            
            features = data.x
            labels = data.y
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            
            return {
                'name': dataset_name,
                'features': features,
                'labels': labels,
                'adj_norm': adj_norm_tensor,
                'adj_raw_coo': adj_raw.tocoo(),
                'adj_norm_coo': adj_norm_coo,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask,
                'num_nodes': num_nodes,
                'num_features': num_features,
                'num_classes': num_classes
            }
            
        except Exception:
            return load_planetoid_raw(dataset_name, root_dir)

def load_planetoid_raw(dataset_name, root_dir):
    url_base = "https://github.com/kimiyoung/planetoid/raw/master/data"
    files = ['ind.{}.x', 'ind.{}.tx', 'ind.{}.allx',
             'ind.{}.y', 'ind.{}.ty', 'ind.{}.ally',
             'ind.{}.graph', 'ind.{}.test.index']
    
    data_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    
    for f_pattern in files:
        fname = f_pattern.format(dataset_name)
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            url = f"{url_base}/{fname}"
            urllib.request.urlretrieve(url, fpath)
            
    objects = []
    for i in range(len(files) - 1):
        fname = files[i].format(dataset_name)
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))
            
    x, tx, allx, y, ty, ally, graph = objects
    
    test_idx_reorder = []
    with open(os.path.join(data_dir, f"ind.{dataset_name}.test.index"), 'r') as f:
        for line in f:
            test_idx_reorder.append(int(line.strip()))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_name == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_reorder), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_reorder), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    features = features.toarray()
    rowsum = features.sum(1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    features = features / rowsum
    features = torch.FloatTensor(features)

    adj = nx_to_adj(graph, features.shape[0])
    adj_norm_coo = normalize_adj(adj)
    adj_norm_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm_coo)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = torch.LongTensor(np.argmax(labels, 1))

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, features.shape[0])
    val_mask = sample_mask(idx_val, features.shape[0])
    test_mask = sample_mask(idx_test, features.shape[0])

    return {
        'name': dataset_name,
        'features': features,
        'labels': labels,
        'adj_norm': adj_norm_tensor,
        'adj_raw_coo': adj.tocoo(),
        'adj_norm_coo': adj_norm_coo,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': features.shape[0],
        'num_features': features.shape[1],
        'num_classes': int(labels.max().item()) + 1
    }

def nx_to_adj(dict_graph, num_nodes):
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for node, neighbors in dict_graph.items():
        for n in neighbors:
            adj[node, n] = 1
            adj[n, node] = 1
    return adj

def sample_mask(idx, l):
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = True
    return mask
