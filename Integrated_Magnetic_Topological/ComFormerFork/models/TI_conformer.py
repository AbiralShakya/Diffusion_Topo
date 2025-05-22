"""Implementation based on the template of Matformer, extended for Topological-Insulator classification."""

from typing import Tuple
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter
from comformer.models.utils import RBFExpansion
from comformer.utils import BaseSettings
from comformer.features import angle_emb_mp
from comformer.models.transformer import ComformerConv, ComformerConv_edge, ComformerConvEqui


class iComformerConfig(BaseSettings):
    """Hyperparameter schema for TI-specialized iComformer."""
    name: str = "iComformer"
    conv_layers: int = 4
    edge_layers: int = 1
    atom_input_features: int = 92
    wyckoff_features: int = 16
    inversion_edge_features: int = 8
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    num_topo_classes: int = 4

    link: str = "identity"

    class Config:
        env_prefix = "jv_model"


class eComformerConfig(BaseSettings):
    """Hyperparameter schema for TI-specialized eComformer."""
    name: str = "eComformer"
    conv_layers: int = 3
    edge_layers: int = 1
    atom_input_features: int = 92
    wyckoff_features: int = 16
    inversion_edge_features: int = 8
    edge_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    num_topo_classes: int = 4

    link: str = "identity"

    class Config:
        env_prefix = "jv_model"


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    return torch.clamp(bond_cosine, -1, 1)


class iComformer(nn.Module):
    """Invariant ComFormer extended for TI classification."""

    def __init__(self, config: iComformerConfig = iComformerConfig()):
        super().__init__()
        # site-symmetry embedding
        self.wyckoff_emb = nn.Embedding(100, config.wyckoff_features)  # adjust num_embeddings
        # inversion-pair edge embedding
        self.inv_edge_emb = nn.Linear(1, config.inversion_edge_features)

        # atom embed takes atom_features + wyckoff_features
        self.atom_embedding = nn.Linear(
            config.atom_input_features + config.wyckoff_features,
            config.node_features
        )
        # RBF + inversion features
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=config.edge_features),
            nn.Linear(config.edge_features + config.inversion_edge_features, config.node_features),
            nn.Softplus(),
        )
        # original angle RBF
        self.rbf_angle = nn.Sequential(
            RBFExpansion(vmin=-1.0, vmax=1.0, bins=config.triplet_input_features),
            nn.Linear(config.triplet_input_features, config.node_features),
            nn.Softplus(),
        )
        # attention layers
        self.att_layers = nn.ModuleList([
            ComformerConv(
                in_channels=config.node_features,
                out_channels=config.node_features,
                heads=1,
                edge_dim=config.node_features,
            ) for _ in range(config.conv_layers)
        ])
        self.edge_update_layer = ComformerConv_edge(
            in_channels=config.node_features,
            out_channels=config.node_features,
            heads=1,
            edge_dim=config.node_features,
        )
        # classification head
        self.fc = nn.Sequential(nn.Linear(config.node_features, config.fc_features), nn.SiLU())
        self.fc_out = nn.Linear(config.fc_features, config.num_topo_classes)
        self.link = (lambda x: x) if config.link == "identity" else None

    def forward(self, data) -> torch.Tensor:
        # data: (graph_data, lattice_data, _)
        graph, _, _ = data
        x = graph.x                     # [N, atom_input_features]
        wy = self.wyckoff_emb(graph.wyckoff)  # [N, wyckoff_features]
        node_features = self.atom_embedding(torch.cat([x, wy], dim=-1))

        # original edges
        ef = -0.75 / torch.norm(graph.edge_attr, dim=1, keepdim=True)  # [E,1]
        # inversion edges
        inv_ef = graph.inv_edge_type.view(-1,1).float()               # [M,1]
        inv_emb = self.inv_edge_emb(inv_ef)                          # [M,InvF]
        # pad and concat
        pad_inv = torch.zeros(ef.size(0), inv_emb.size(1), device=ef.device)
        ef_all = torch.cat([ef, inv_ef], dim=0)                       # [E+M,1]
        inv_pad = torch.cat([pad_inv, inv_emb], dim=0)                # [E+M,InvF]
        edge_features = self.rbf(torch.cat([ef_all, inv_pad], dim=1)) # [E+M, node_features]
        # build combined edge_index
        edge_index = torch.cat([graph.edge_index, graph.inv_edge_index], dim=1)

        # angle/triplet features
        nei_len = -0.75 / torch.norm(graph.edge_nei, dim=-1)           # [E,3]
        nei_angle = bond_cosine(
            graph.edge_nei, graph.edge_attr.unsqueeze(1).repeat(1,3,1)
        )                                                             # [E,3]
        nei_len_emb = self.rbf(nei_len.view(-1,1)).view(-1,3,self.rbf[1].out_features)
        nei_ang_emb = self.rbf_angle(nei_angle.view(-1,1)).view(-1,3,self.rbf_angle[1].out_features)

        # message passing
        for i, conv in enumerate(self.att_layers):
            node_features = conv(node_features, edge_index, edge_features)
            if i==0:
                edge_features = self.edge_update_layer(edge_features, nei_len_emb, nei_ang_emb)

        # global readout + classification
        h = scatter(node_features, graph.batch, dim=0, reduce='mean')
        h = self.fc(h)
        out = self.fc_out(h)
        if self.link: out = self.link(out)
        return F.log_softmax(out, dim=-1)


class eComformer(nn.Module):
    """Equivariant ComFormer extended for TI classification."""

    def __init__(self, config: eComformerConfig = eComformerConfig()):
        super().__init__()
        # same wyckoff + inv-edge setup
        self.wyckoff_emb = nn.Embedding(100, config.wyckoff_features)
        self.inv_edge_emb = nn.Linear(1, config.inversion_edge_features)
        self.atom_embedding = nn.Linear(
            config.atom_input_features + config.wyckoff_features,
            config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=config.edge_features),
            nn.Linear(config.edge_features + config.inversion_edge_features, config.node_features),
            nn.Softplus(),
        )
        # attention layers
        self.att_layers = nn.ModuleList([
            ComformerConv(
                in_channels=config.node_features,
                out_channels=config.node_features,
                heads=1,
                edge_dim=config.node_features,
            ) for _ in range(config.conv_layers)
        ])
        # equivariant update
        self.equi_update = ComformerConvEqui(
            in_channels=config.node_features,
            out_channels=config.node_features,
            edge_dim=config.node_features,
            use_second_order_repr=True
        )
        # classification head
        self.fc = nn.Sequential(nn.Linear(config.node_features, config.fc_features), nn.SiLU())
        self.fc_out = nn.Linear(config.fc_features, config.num_topo_classes)
        self.link = (lambda x: x) if config.link == "identity" else None

    def forward(self, data) -> torch.Tensor:
        graph, _, _ = data
        x = graph.x
        wy = self.wyckoff_emb(graph.wyckoff)
        h = self.atom_embedding(torch.cat([x, wy], dim=-1))

        ef = -0.75 / torch.norm(graph.edge_attr, dim=1, keepdim=True)
        inv_ef = graph.inv_edge_type.view(-1,1).float()
        inv_emb = self.inv_edge_emb(inv_ef)
        pad_inv = torch.zeros(ef.size(0), inv_emb.size(1), device=ef.device)
        ef_all = torch.cat([ef, inv_ef], dim=0)
        inv_pad = torch.cat([pad_inv, inv_emb], dim=0)
        edge_features = self.rbf(torch.cat([ef_all, inv_pad], dim=1))
        edge_index = torch.cat([graph.edge_index, graph.inv_edge_index], dim=1)

        h = self.att_layers[0](h, edge_index, edge_features)
        h = self.equi_update(graph, h, edge_index, edge_features)
        for lay in self.att_layers[1:]:
            h = lay(h, edge_index, edge_features)

        h = scatter(h, graph.batch, dim=0, reduce='mean')
        h = self.fc(h)
        out = self.fc_out(h)
        if self.link: out = self.link(out)
        return F.log_softmax(out, dim=-1)
