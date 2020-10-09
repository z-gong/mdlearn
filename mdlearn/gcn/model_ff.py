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
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, n_head, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, n_head, out_dim)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, n_head, in_dim_edge)))
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.normal_(self.fc_node.weight, std=0.5)
        nn.init.normal_(self.attn_l, std=0.5)
        nn.init.normal_(self.attn_r, std=0.5)
        nn.init.normal_(self.attn_e, std=0.5)

        self._n_head = n_head
        self._out_dim = out_dim
        self._in_dim_edge = in_dim_edge

    def forward(self, g, feats_node, feats_edge, etype):
        graph = g[etype]
        with graph.local_scope():
            feat_n = self.fc_node(feats_node).view(-1, self._n_head, self._out_dim)
            feat_e = feats_edge.repeat_interleave(self._n_head, dim=0).view(-1, self._n_head, self._in_dim_edge)
            # compute attention
            el = (feat_n * self.attn_l).sum(dim=-1, keepdim=True)
            er = (feat_n * self.attn_r).sum(dim=-1, keepdim=True)
            ee = (feat_e * self.attn_e).sum(dim=-1, keepdim=True)
            graph.ndata.update({'ft': feat_n, 'el': el, 'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'en'))
            e = self.leaky_relu(graph.edata.pop('en') + ee)
            # soft max attention
            graph.edata['a'] = dgl.ops.edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            x = graph.ndata.pop('ft').view(-1, self._n_head * self._out_dim)
            return x


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
    def __init__(self, in_dim_node, in_dim_edges, in_dim_graph, out_dim, n_head_list):
        super(ForceFieldGATModel, self).__init__()
        n_edge_type = len(in_dim_edges)
        self.fc_edges = nn.ModuleDict()
        for edge_type, in_dim_edge in in_dim_edges.items():
            self.fc_edges[edge_type] = nn.Sequential(nn.Linear(in_dim_edge, out_dim),
                                                     nn.SELU(),
                                                     nn.Linear(out_dim, out_dim),
                                                     nn.SELU(),
                                                     )
        _dim_edges = {etype: out_dim for etype in in_dim_edges}

        self.n_conv = len(n_head_list)

        self.gat_list = nn.ModuleList()
        for i in range(self.n_conv):
            n_head = n_head_list[i]
            if i == 0:
                layer = FFGATLayer(in_dim_node, _dim_edges, out_dim, n_head)
            else:
                n_head_last = n_head_list[i - 1]
                layer = FFGATLayer(out_dim * n_head_last * n_edge_type, _dim_edges, out_dim, n_head)
            self.gat_list.append(layer)

        self.readout = WeightedAverage(out_dim * n_head_list[-1] * n_edge_type)
        self.mlp = MLPModel(out_dim * n_head_list[-1] * n_edge_type + in_dim_graph, 1, [out_dim * 2, out_dim])

        for layer in self.fc_edges.values():
            if hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, std=0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, g, feats_node, feats_edges, feats_graph):
        feats_e = {etype: module(feats_edges[etype]) for etype, module in self.fc_edges.items()}
        x = feats_node
        for i in range(self.n_conv):
            x = F.relu(self.gat_list[i](g, x, feats_e))  # (V, out_dim * n_head * n_edge_type)
        embedding = self.readout(g, x)
        return self.mlp(torch.cat([embedding, feats_graph], dim=1))
