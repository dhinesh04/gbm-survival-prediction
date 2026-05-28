"""
gcn_model.py
------------
Graph Convolutional Network with dual prediction heads:

  1. Binary head  → LTS / non-LTS classification (cross-entropy loss)
  2. Cox head     → continuous risk score for time-to-event survival
                    modelling (Cox partial likelihood loss)

Both heads share the same GCN backbone (two graph conv layers).

Optional per-modality encoders (Issue 5 response):
  When modality_dims is provided (e.g. [50, 50, 50, 4] for CNA/mRNA/meth/clin),
  each modality is projected through its own small MLP (Linear → ELU) into a
  shared ENC_DIM-dimensional space before concatenation and graph convolution.
  This lets each modality learn its own representation independently, which is
  important because CNA (discrete copy ratios), mRNA (log-expression), and
  methylation (β-values in [0,1]) have completely different distributions.
  When modality_dims=None the model behaves identically to the original
  (plain concatenated input → GCN), used for ablation A7.
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
        x1     = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return (f"GraphConvolution("
                f"in={self.in_features}, out={self.out_features})")


# ─────────────────────────────────────────────────────────────────────────────
# GCN MODEL WITH OPTIONAL MODALITY ENCODERS AND DUAL HEAD
# ─────────────────────────────────────────────────────────────────────────────
class GCN(nn.Module):
    """
    Two-layer GCN with optional per-modality encoders, a binary
    classification head, and a Cox survival head.

    Architecture (with modality encoders — ablation A8 / full model):
        CNA  (50) ──► Linear(50→enc_dim) + ELU ──┐
        mRNA (50) ──► Linear(50→enc_dim) + ELU ──┼── cat → GraphConv → GraphConv
        Meth (50) ──► Linear(50→enc_dim) + ELU ──┤                         │
        Clin  (4) ──► Linear( 4→enc_dim) + ELU ──┘                    embeddings H
                                                                        /          \
                                                             fc_bin (binary)   fc_cox (risk)

    Architecture (without encoders — ablation A7, plain concatenation):
        [CNA ∥ mRNA ∥ Meth ∥ Clin] (154-dim) → GraphConv → GraphConv
                                                                   │
                                                              embeddings H
                                                              /          \
                                                   fc_bin (binary)   fc_cox (risk)

    Parameters
    ----------
    n_in         : int        Total raw input feature dimension (154).
    n_hid        : int        Hidden dimension of GCN layers (64).
    n_out        : int        Number of binary output classes (2).
    dropout      : float      Dropout rate after each GCN layer.
    modality_dims: list[int]  Feature dimension of each modality block in
                              the order they appear in the feature matrix,
                              e.g. [50, 50, 50, 4] for CNA/mRNA/meth/clin.
                              Must sum to n_in.  If None, encoders are
                              disabled and the raw concatenation is used.
    enc_dim      : int        Output dimension of each modality encoder.
                              GCN input = enc_dim × len(modality_dims).
    """

    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 dropout: float = 0.5,
                 modality_dims: list = None,
                 enc_dim: int = 32):
        super(GCN, self).__init__()
        self.modality_dims = modality_dims
        self.enc_dim       = enc_dim

        if modality_dims is not None:
            assert sum(modality_dims) == n_in, (
                f"sum(modality_dims)={sum(modality_dims)} must equal n_in={n_in}"
            )
            # One encoder per modality: Linear → ELU
            self.mod_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, enc_dim),
                    nn.ELU(),
                )
                for d in modality_dims
            ])
            gcn_in = enc_dim * len(modality_dims)
        else:
            self.mod_encoders = None
            gcn_in = n_in

        self.gc1     = GraphConvolution(gcn_in, n_hid)
        self.gc2     = GraphConvolution(n_hid,  n_hid)
        self.dp1     = nn.Dropout(dropout)
        self.dp2     = nn.Dropout(dropout)
        self.fc_bin  = nn.Linear(n_hid, n_out)
        self.fc_cox  = nn.Linear(n_hid, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x   : (N, n_in)  node feature matrix
        adj : (N, N)     normalised adjacency matrix

        Returns
        -------
        bin_logits  : (N, 2)     raw logits for binary head
        cox_risk    : (N,)       scalar risk scores (higher = more risk)
        embeddings  : (N, n_hid) shared GCN node embeddings
        """
        # ── Per-modality encoding (optional) ─────────────────────────────────
        if self.mod_encoders is not None:
            parts  = []
            offset = 0
            for enc, d in zip(self.mod_encoders, self.modality_dims):
                parts.append(enc(x[:, offset: offset + d]))
                offset += d
            x = torch.cat(parts, dim=1)   # (N, enc_dim × n_modalities)

        # ── Graph convolution backbone ────────────────────────────────────────
        h = self.gc1(x, adj)
        h = F.elu(h)
        h = self.dp1(h)
        h = self.gc2(h, adj)
        h = F.elu(h)
        h = self.dp2(h)

        # ── Dual prediction heads ─────────────────────────────────────────────
        bin_logits = self.fc_bin(h)              # (N, 2)
        cox_risk   = self.fc_cox(h).squeeze(-1)  # (N,)

        return bin_logits, cox_risk, h

    def __repr__(self):
        if self.mod_encoders is not None:
            enc_str = (f"ModalityEncoders({self.modality_dims}→{self.enc_dim}each)"
                       f" → GCN({self.enc_dim*len(self.modality_dims)}"
                       f"→{self.gc1.out_features}→{self.gc2.out_features})")
        else:
            enc_str = (f"GCN({self.gc1.in_features}"
                       f"→{self.gc1.out_features}→{self.gc2.out_features})")
        return (f"GCN({enc_str}, "
                f"BinaryHead→{self.fc_bin.out_features}, "
                f"CoxHead→1, dropout={self.dropout})")