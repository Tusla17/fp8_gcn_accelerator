import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNConvManual(nn.Module):
    """
    Standard GCN Layer with explicit A(HW) computation order:
    Feature Transformation: Z = H * W
    Feature Aggregation:    H_out = A_hat * Z
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GCNConvManual, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):
        """
        x: Dense feature matrix [N, in_features]
        adj_norm: Normalized adjacency matrix (Sparse or Dense) [N, N]
        """
        # Step 1: Feature Transformation (Dense-Dense GEMM: H x W)
        # In hardware, this is compute-bound and will be accelerated by Approximate Multiplier
        z = torch.mm(x, self.weight)
        
        # Step 2: Feature Aggregation (Sparse-Dense SpMM: A_hat x Z)
        # In hardware, this is memory-bound and follows Gustavson's element-wise dataflow
        if adj_norm.is_sparse:
            out = torch.spmm(adj_norm, z)
        else:
            out = torch.mm(adj_norm, z)
            
        if self.bias is not None:
            out = out + self.bias
            
        return out


class GCN2Layer(nn.Module):
    """
    2-Layer Graph Convolutional Network matching the FP8GCN architecture:
    Layer 1: in_features -> hidden_dim (16) -> ReLU -> Dropout
    Layer 2: hidden_dim (16) -> num_classes -> LogSoftmax
    """
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5, bias=False):
        super(GCN2Layer, self).__init__()
        self.conv1 = GCNConvManual(in_features, hidden_dim, bias=bias)
        self.conv2 = GCNConvManual(hidden_dim, num_classes, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        # Layer 1
        h1 = self.conv1(x, adj_norm)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # Layer 2
        h2 = self.conv2(h1, adj_norm)
        return F.log_softmax(h2, dim=1)

    def get_layer_weights(self):
        """Returns W0 and W1 as numpy arrays or torch tensors"""
        return self.conv1.weight.detach(), self.conv2.weight.detach()
