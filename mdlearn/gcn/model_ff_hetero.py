import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from .model import WeightedAverage, MLPModel


class EdgeGATLayer(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, out_dim):
        super(EdgeGATLayer, self).__init__()
        self.fc_node = nn.Linear(in_dim_node, out_dim, bias=False)
        self.fc_edge = nn.Linear(in_dim_edge, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

        for layer in self.fc_node, self.fc_edge, self.attn_fc:
            nn.init.normal_(layer.weight, std=0.5)

    def edge_attention(self, edges):
        z3 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['z']], dim=1)
        a = self.attn_fc(z3)
        return {'e': F.leaky_relu(a, negative_slope=0.2)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, feats_node, feats_edge, etype):
        g.ndata['z'] = self.fc_node(feats_node)
        g.edges[etype].data['z'] = self.fc_edge(feats_edge)
        g.apply_edges(self.edge_attention, etype=etype)
        g.update_all(self.message_func, self.reduce_func, etype=etype)
        return g.ndata.pop('h')


class MultiHeadEdgeGATLayer(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, out_dim, n_head):
        super(MultiHeadEdgeGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(n_head):
            self.heads.append(EdgeGATLayer(in_dim_node, in_dim_edge, out_dim))

    def forward(self, g, feats_node, feats_edge, etype):
        heads_out = [gat(g, feats_node, feats_edge, etype) for gat in self.heads]
        return torch.cat(heads_out, dim=1)


class FFGATLayer(nn.Module):
    def __init__(self, in_dim_node, in_dim_bond, in_dim_angle, in_dim_dihedral, out_dim, n_head):
        super(FFGATLayer, self).__init__()
        self.gat_bond = MultiHeadEdgeGATLayer(in_dim_node, in_dim_bond, out_dim, n_head)
        self.gat_angle = MultiHeadEdgeGATLayer(in_dim_node, in_dim_angle, out_dim, n_head)
        # self.gat_dihedral = MultiHeadEdgeGATLayer(in_dim_node, in_dim_dihedral, out_dim, n_head)

    def forward(self, g, feats_node, feats_bond, feats_angle, feats_dihedral):
        hb = self.gat_bond(g, feats_node, feats_bond, 'bond')
        ha = self.gat_angle(g, feats_node, feats_angle, 'angle')
        # hd = self.gat_dihedral(g, feats_node, feats_dihedral, 'dihedral')
        # return torch.cat([hb, ha, hd], dim=1)
        return torch.cat([hb, ha], dim=1)


class ForceFieldGATModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_bond, in_dim_angle, in_dim_dihedral, in_dim_graph, out_dim, n_head):
        super(ForceFieldGATModel, self).__init__()
        self.gat1 = FFGATLayer(in_dim_node, in_dim_bond, in_dim_angle, in_dim_dihedral, out_dim, n_head)
        # self.gat2 = FFGATLayer(out_dim * n_head * 3, in_dim_bond, in_dim_angle, in_dim_dihedral, out_dim, n_head)
        # self.gat3 = FFGATLayer(out_dim * n_head * 3, in_dim_bond, in_dim_angle, in_dim_dihedral, out_dim, 1)
        # self.readout = WeightedAverage(out_dim * 3)
        # self.mlp = MLPModel(out_dim * 3 + in_dim_graph, 1, [out_dim * 2, out_dim])
        self.gat2 = FFGATLayer(out_dim * n_head * 2, in_dim_bond, in_dim_angle, in_dim_dihedral, out_dim, n_head)
        self.gat3 = FFGATLayer(out_dim * n_head * 2, in_dim_bond, in_dim_angle, in_dim_dihedral, out_dim, 1)
        self.readout = WeightedAverage(out_dim * 2)
        self.mlp = MLPModel(out_dim * 2 + in_dim_graph, 1, [out_dim * 2, out_dim])

    def forward(self, g, feats_node, feats_bond, feats_angle, feats_dihedral, feats_graph):
        x = F.selu(self.gat1(g, feats_node, feats_bond, feats_angle, feats_dihedral))
        x = F.selu(self.gat2(g, x, feats_bond, feats_angle, feats_dihedral))
        x = F.selu(self.gat3(g, x, feats_bond, feats_angle, feats_dihedral))
        embedding = self.readout(g, x)
        return self.mlp(torch.cat([embedding, feats_graph], dim=1))
