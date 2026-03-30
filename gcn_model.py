"""
gcn_model.py
------------
Graph Convolutional Network with dual prediction heads:

  1. Binary head  → LTS / non-LTS classification (cross-entropy loss)
  2. Cox head     → continuous risk score for time-to-event survival
                    modelling (Cox partial likelihood loss)

Both heads share the same GCN backbone (two graph conv layers).
The binary head answers "is this patient a long-term survivor?"
The Cox head answers "what is this patient's relative survival risk?"
and enables Kaplan-Meier stratification.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONVOLUTION LAYER
# ─────────────────────────────────────────────────────────────────────────────
class GraphConvolution(nn.Module):
    """
    Single graph convolutional layer.
    Implements: H = A_norm · X · W + b
    (Kipf & Welling 2017 spectral GCN formulation)
    """

    def __init__(self, infeas: int, outfeas: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features  = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(infeas, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outfeas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        x1     = torch.mm(x, self.weight)    # X · W
        output = torch.mm(adj, x1)           # A · (X · W)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return (f"GraphConvolution("
                f"in={self.in_features}, out={self.out_features})")


# ─────────────────────────────────────────────────────────────────────────────
# GCN MODEL WITH DUAL HEAD
# ─────────────────────────────────────────────────────────────────────────────
class GCN(nn.Module):
    """
    Two-layer GCN with a binary classification head and a Cox survival head.

    Architecture:
        Input (N, n_in)
            ↓  GraphConvolution(n_in → n_hid) + ELU + Dropout
            ↓  GraphConvolution(n_hid → n_hid) + ELU + Dropout
            ↓  Shared node embeddings H  (N, n_hid)
            ├── Linear(n_hid → n_out)  → binary logits  (N, 2)
            └── Linear(n_hid → 1)      → Cox risk score  (N, 1)

    Parameters
    ----------
    n_in  : int    Input feature dimension.
    n_hid : int    Hidden layer dimension.
    n_out : int    Binary output classes (2 for LTS/non-LTS).
    dropout : float  Dropout rate after each GCN layer.
    """

    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 dropout: float = 0.5):
        super(GCN, self).__init__()
        self.gc1      = GraphConvolution(n_in,  n_hid)
        self.gc2      = GraphConvolution(n_hid, n_hid)
        self.dp1      = nn.Dropout(dropout)
        self.dp2      = nn.Dropout(dropout)
        # Binary classification head
        self.fc_bin   = nn.Linear(n_hid, n_out)
        # Cox survival head — outputs a scalar risk score per patient
        # No activation: risk scores are unbounded real numbers
        self.fc_cox   = nn.Linear(n_hid, 1)
        self.dropout  = dropout

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x   : (N, n_in)  node feature matrix
        adj : (N, N)     normalised adjacency matrix

        Returns
        -------
        bin_logits  : (N, 2)  — raw logits for binary head
        cox_risk    : (N,)    — scalar risk scores for Cox head
                               higher score = higher risk (shorter survival)
        embeddings  : (N, n_hid) — node embeddings (for inspection)
        """
        # Shared backbone
        h = self.gc1(x, adj)
        h = F.elu(h)
        h = self.dp1(h)
        h = self.gc2(h, adj)
        h = F.elu(h)
        h = self.dp2(h)

        # Dual heads
        bin_logits = self.fc_bin(h)              # (N, 2)
        cox_risk   = self.fc_cox(h).squeeze(-1)  # (N,)

        return bin_logits, cox_risk, h

    def __repr__(self):
        return (f"GCN(in={self.gc1.in_features}, "
                f"hid={self.gc2.in_features}, "
                f"out_bin={self.fc_bin.out_features}, "
                f"out_cox=1, dropout={self.dropout})")