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
        self.fc_edge = nn.Linear(in_dim_edge, in_dim_edge, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim + in_dim_edge, 1, bias=False)

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

    def forward(self, g, feats_node, feats_edge):
        g.ndata['z'] = self.fc_node(feats_node)
        g.edata['z'] = self.fc_edge(feats_edge)
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h'), g.edata.pop('z')


class MultiHeadEdgeGATLayer(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, out_dim, n_head):
        super(MultiHeadEdgeGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(n_head):
            self.heads.append(EdgeGATLayer(in_dim_node, in_dim_edge, out_dim))

    def forward(self, g, feats_node, feats_edge):
        ndata, edata = zip(*[gat(g, feats_node, feats_edge) for gat in self.heads])
        return torch.cat(ndata, dim=1), torch.mean(torch.stack(edata, dim=0), dim=0)


class ForceFieldGATModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, in_dim_graph, out_dim, n_head):
        super(ForceFieldGATModel, self).__init__()
        self.gat1 = MultiHeadEdgeGATLayer(in_dim_node, in_dim_edge, out_dim, n_head)
        self.gat2 = MultiHeadEdgeGATLayer(out_dim * n_head, in_dim_edge, out_dim, n_head)
        self.gat3 = MultiHeadEdgeGATLayer(out_dim * n_head, in_dim_edge, out_dim, 1)
        self.readout = WeightedAverage(out_dim)
        self.mlp = MLPModel(out_dim + in_dim_graph, 1, [out_dim * 2, out_dim])

    def forward(self, g, feats_node, feats_edge, feats_graph):
        ndata, edata = self.gat1(g, feats_node, feats_edge)
        ndata, edata = self.gat2(g, F.selu(ndata), F.selu(edata))
        ndata, edata = self.gat3(g, F.selu(ndata), F.selu(edata))
        embedding = self.readout(g, F.selu(ndata))
        return self.mlp(torch.cat([embedding, feats_graph], dim=1))
