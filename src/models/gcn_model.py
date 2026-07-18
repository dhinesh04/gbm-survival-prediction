"""GCN with dual heads: AFT regression (predicted log-survival time) and Cox (risk score)."""

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter


class OmicsSelfAttention(nn.Module):
    """True self-attention (Q/K/V from the tokens themselves) across the omics
    modalities only -- clinical is excluded and concatenated in afterward by
    the caller. Mirrors MoACG (Qin et al. 2026)'s scope.

    Output dim is enc_dim * n_omics, uncompressed, so concatenating clinical's
    enc_dim afterward still lands on enc_dim * 4 == gcn_in.
    `last_attn_weights` ([N, n_omics, n_omics]) is a pairwise relatedness
    matrix, not a per-modality importance score.
    """
    def __init__(self, enc_dim: int, n_omics: int = 3):
        super(OmicsSelfAttention, self).__init__()
        self.n_omics = n_omics
        self.scale   = enc_dim ** -0.5

        self.W_Q = nn.Linear(enc_dim, enc_dim)
        self.W_K = nn.Linear(enc_dim, enc_dim)
        self.W_V = nn.Linear(enc_dim, enc_dim)
        self.compress = nn.Linear(enc_dim * n_omics, enc_dim * n_omics)
        self.softmax  = nn.Softmax(dim=-1)
        self.elu      = nn.ELU()

        self.last_attn_weights = None  # [N, n_omics, n_omics]

    def forward(self, omics_embeddings: list) -> torch.Tensor:
        """omics_embeddings: n_omics tensors of [N, enc_dim] (CNA, mRNA, Meth order)."""
        assert len(omics_embeddings) == self.n_omics, (
            f"expected {self.n_omics} omics embeddings, got {len(omics_embeddings)}"
        )
        N = omics_embeddings[0].shape[0]

        tokens = torch.stack(omics_embeddings, dim=1)   # [N, n_omics, enc_dim]
        Q = self.W_Q(tokens)
        K = self.W_K(tokens)
        V = self.W_V(tokens)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [N, n_omics, n_omics]
        weights = self.softmax(scores)
        self.last_attn_weights = weights.detach()

        ctx = torch.matmul(weights, V)                  # [N, n_omics, enc_dim] contextualized tokens
        h_omics = self.elu(self.compress(ctx.reshape(N, -1)))   # [N, n_omics*enc_dim]
        return h_omics


class GraphConvolution(nn.Module):
    """Single GCN layer: H = A_norm . X . W + b"""
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

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x1     = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return f"GraphConvolution(in={self.in_features}, out={self.out_features})"


class GCN(nn.Module):
    """Two-layer GCN with modality fusion, a log-normal AFT head, and a Cox head."""
    def __init__(self, n_in: int, n_hid: int,
                 dropout: float = 0.5,
                 modality_dims: list = None,
                 enc_dim: int = 32,
                 fusion_type: str = "none",
                 n_fusion_heads: int = 4,
                 n_self_attn_modalities: int = None):
        super(GCN, self).__init__()
        self.modality_dims = modality_dims
        self.enc_dim       = enc_dim
        self.fusion_type   = fusion_type if modality_dims is not None else None

        if modality_dims is not None:
            assert sum(modality_dims) == n_in, (
                f"sum(modality_dims)={sum(modality_dims)} must equal n_in={n_in}"
            )
            self.mod_encoders = nn.ModuleList([
                nn.Sequential(nn.Linear(d, enc_dim), nn.ELU())
                for d in modality_dims
            ])
            n_mod = len(modality_dims)

            if fusion_type == "none":
                self.fusion_layer = None
                self.n_self_attn  = None
                gcn_in = enc_dim * n_mod

            elif fusion_type == "omics_self_attn":
                # k = how many leading encoder outputs self-attend; the rest are
                # concatenated in unchanged. Default k=n_mod-1 means "omics
                # self-attend, clinical bypasses" when n_mod=4.
                k = (n_self_attn_modalities if n_self_attn_modalities is not None
                     else max(1, n_mod - 1))
                assert 1 <= k <= n_mod, (
                    f"n_self_attn_modalities={k} must be between 1 and n_mod={n_mod}"
                )
                self.fusion_layer = OmicsSelfAttention(enc_dim=enc_dim, n_omics=k)
                self.n_self_attn  = k
                gcn_in = enc_dim * n_mod

            else:
                raise ValueError(
                    f"Unknown or invalid fusion_type: {fusion_type!r} for "
                    f"{n_mod} modalities. Expected 'none', 'omics_self_attn' "
                    f"(any modality count), or 'attention' (4 modalities only)."
                )
        else:
            self.mod_encoders = None
            self.fusion_layer = None
            self.n_self_attn  = None
            gcn_in = n_in

        self.gc1 = GraphConvolution(gcn_in, n_hid)
        self.gc2 = GraphConvolution(n_hid,  n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

        self.fc_reg  = nn.Linear(n_hid, 1)
        self.fc_cox  = nn.Linear(n_hid, 1)

        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.dropout = dropout

    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        # per-modality encoding + fusion
        if self.mod_encoders is not None:
            parts, offset = [], 0
            for enc, d in zip(self.mod_encoders, self.modality_dims):
                parts.append(enc(x[:, offset: offset + d]))
                offset += d

            if self.fusion_type == "omics_self_attn" and self.fusion_layer is not None:
                k = self.n_self_attn
                h_attn  = self.fusion_layer(parts[:k])    # [N, k*enc_dim]
                h_plain = parts[k:]                        # remaining modalities, unchanged
                x = torch.cat([h_attn] + h_plain, dim=1) if h_plain else h_attn

            else:
                x = torch.cat(parts, dim=1)                # plain concat (fusion_type="none")

        h = self.gc1(x, adj)
        h = self.elu1(h)
        h = self.dp1(h)
        h = self.gc2(h, adj)
        h = self.elu2(h)
        h = self.dp2(h)

        pred_log_t = self.fc_reg(h).squeeze(-1)
        cox_risk   = self.fc_cox(h).squeeze(-1)

        return pred_log_t, cox_risk, h

    def __repr__(self):
        sigma_val = float(torch.exp(self.log_sigma).item())
        if self.mod_encoders is not None:
            if self.fusion_layer is not None:
                fusion_name = {
                    "attention":       "AttnFusion",
                    "omics_self_attn": "OmicsSelfAttnFusion",
                }.get(self.fusion_type, "Fusion")
                enc_str = f"ModEncoders→{fusion_name}(out={self.gc1.in_features})→GCN"
            else:
                enc_str = f"ModEncoders(concat)→GCN"
        else:
            enc_str = f"GCN(Raw)"
            
        return (f"GCN({enc_str}, AFTHead→mu=log(t̂) [σ={sigma_val:.3f}], "
                f"CoxHead→risk, dropout={self.dropout})")