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
    def __init__(self, in_dim_node, in_dim_edges, out_dim, n_head):
        super(FFGATLayer, self).__init__()
        self.module_dict = nn.ModuleDict()
        for edge_type, in_dim_edge in in_dim_edges.items():
            self.module_dict[edge_type] = MultiHeadEdgeGATLayer(in_dim_node, in_dim_edge, out_dim, n_head)

    def forward(self, g, feats_node, feats_edges):
        out_list = [module(g, feats_node, feats_edges[etype], etype) for etype, module in self.module_dict.items()]
        return torch.cat(out_list, dim=1)


class ForceFieldGATModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_edges, in_dim_graph, out_dim, n_head):
        super(ForceFieldGATModel, self).__init__()
        self.project_node = nn.Sequential(nn.Linear(in_dim_node, out_dim),
                                          nn.SELU())
        self.gat1 = FFGATLayer(out_dim, in_dim_edges, out_dim, n_head)
        # self.gat2 = FFGATLayer(out_dim * n_head * len(in_dim_edges), in_dim_edges, out_dim, n_head)
        # self.gat3 = FFGATLayer(out_dim * n_head * len(in_dim_edges), in_dim_edges, out_dim, 1)
        self.gru = nn.GRU(out_dim * n_head * len(in_dim_edges), out_dim)
        self.readout = WeightedAverage(out_dim)
        self.mlp = MLPModel(out_dim + in_dim_graph, 1, [out_dim * 2, out_dim])

    def forward(self, g, feats_node, feats_edges, feats_graph):
        x = self.project_node(feats_node)  # (V, out_dim)
        hidden = x.unsqueeze(0)  # (1, V, out_dim)
        for i in range(3):
            x = F.selu(self.gat1(g, x, feats_edges))  # (V, out_dim * n_head * n_etype)
            x, hidden = self.gru(x.unsqueeze(0), hidden)  # (1, V, out_dim)
            x = x.squeeze(0)  # (V, out_dim)
        # x = F.selu(self.gat2(g, x, feats_edges))
        # x = F.selu(self.gat3(g, x, feats_edges))
        embedding = self.readout(g, x)
        return self.mlp(torch.cat([embedding, feats_graph], dim=1))
