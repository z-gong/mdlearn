import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from .model import WeightedAverage, MLPModel


class EdgeGATLayer(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, out_dim, n_head):
        super(EdgeGATLayer, self).__init__()
        self.fc_node = nn.Linear(in_dim_node, out_dim * n_head, bias=False)
        self.fc_edge = nn.Sequential(nn.Linear(in_dim_edge, out_dim),
                                     nn.SELU(),
                                     )
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, n_head, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, n_head, out_dim)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, n_head, out_dim)))

        nn.init.normal_(self.fc_node.weight, std=0.5)
        for layer in self.fc_edge:
            if hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, std=0.5)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.attn_l, std=0.5)
        nn.init.normal_(self.attn_r, std=0.5)
        nn.init.normal_(self.attn_e, std=0.5)

        self._n_head = n_head
        self._out_dim = out_dim

    def forward(self, g, feats_node, feats_edge, etype):
        graph = g[etype]
        with graph.local_scope():
            feat_n = self.fc_node(feats_node).view(-1, self._n_head, self._out_dim)
            feat_e = self.fc_edge(feats_edge).repeat_interleave(self._n_head, dim=0) \
                .view(-1, self._n_head, self._out_dim)  # (E, n_head, out_dim)
            # compute attention
            el = (feat_n * self.attn_l).sum(dim=-1, keepdim=True)
            er = (feat_n * self.attn_r).sum(dim=-1, keepdim=True)
            ee = (feat_e * self.attn_e).sum(dim=-1, keepdim=True)
            graph.ndata.update({'ft': feat_n, 'el': el, 'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'en'))
            e = F.leaky_relu(graph.edata.pop('en') + ee, negative_slope=0.2)
            # soft max attention
            graph.edata['a'] = dgl.ops.edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            return graph.ndata.pop('ft').view(-1, self._n_head * self._out_dim)


class FFGATLayer(nn.Module):
    def __init__(self, in_dim_node, in_dim_edges, out_dim, n_head):
        super(FFGATLayer, self).__init__()
        self.module_dict = nn.ModuleDict()
        for edge_type, in_dim_edge in in_dim_edges.items():
            self.module_dict[edge_type] = EdgeGATLayer(in_dim_node, in_dim_edge, out_dim, n_head)

    def forward(self, g, feats_node, feats_edges):
        out_list = [module(g, feats_node, feats_edges[etype], etype) for etype, module in self.module_dict.items()]
        return torch.cat(out_list, dim=1)


class ForceFieldGATModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_edges, in_dim_graph, out_dim, n_head):
        super(ForceFieldGATModel, self).__init__()
        n_edge_type = len(in_dim_edges)
        self.gat1 = FFGATLayer(in_dim_node, in_dim_edges, out_dim, n_head)
        self.gat2 = FFGATLayer(out_dim * n_head * n_edge_type, in_dim_edges, out_dim, n_head)
        self.gat3 = FFGATLayer(out_dim * n_head * n_edge_type, in_dim_edges, out_dim, 1)
        self.readout = WeightedAverage(out_dim * n_edge_type)
        self.mlp = MLPModel(out_dim * len(in_dim_edges) + in_dim_graph, 1, [out_dim * 2, out_dim])

    def forward(self, g, feats_node, feats_edges, feats_graph):
        x = F.selu(self.gat1(g, feats_node, feats_edges))  # (V, out_dim * n_head * n_edge_type)
        x = F.selu(self.gat2(g, x, feats_edges))  # (V, out_dim * n_head * n_edge_type)
        x = F.selu(self.gat3(g, x, feats_edges))  # (V, out_dim * n_edge_type)
        embedding = self.readout(g, x)
        return self.mlp(torch.cat([embedding, feats_graph], dim=1))


class ForceFieldGatedGATModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_edges, in_dim_graph, out_dim, n_head, n_conv=3):
        super(ForceFieldGatedGATModel, self).__init__()
        self.project_node = nn.Sequential(nn.Linear(in_dim_node, out_dim),
                                          nn.SELU())
        self.gat_list = nn.ModuleList()
        for i in range(n_conv):
            self.gat_list.append(FFGATLayer(out_dim, in_dim_edges, out_dim, n_head))
        self.gru = nn.GRU(out_dim * n_head * len(in_dim_edges), out_dim)
        self.readout = WeightedAverage(out_dim)
        self.mlp = MLPModel(out_dim + in_dim_graph, 1, [out_dim * 2, out_dim])

        nn.init.normal_(self.project_node[0].weight, std=0.5)
        nn.init.zeros_(self.project_node[0].bias)

    def forward(self, g, feats_node, feats_edges, feats_graph):
        x = self.project_node(feats_node)  # (V, out_dim)
        hidden = x.unsqueeze(0)  # (1, V, out_dim)
        for gat in self.gat_list:
            x = F.selu(gat(g, x, feats_edges))  # (V, out_dim * n_head * n_edge_type)
            x, hidden = self.gru(x.unsqueeze(0), hidden)  # (1, V, out_dim)
            x = x.squeeze(0)  # (V, out_dim)
        embedding = self.readout(g, x)
        return self.mlp(torch.cat([embedding, feats_graph], dim=1))
