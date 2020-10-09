import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers):
        super().__init__()
        layers = []
        for i in range(len(hidden_layers) + 1):
            _in = in_dim if i == 0 else hidden_layers[i - 1]
            _out = out_dim if i == len(hidden_layers) else hidden_layers[i]

            linear = nn.Linear(_in, _out)
            torch.nn.init.normal_(linear.weight, std=0.5)
            torch.nn.init.zeros_(linear.bias)
            layers.append(linear)

            if i != len(hidden_layers):
                layers.append(nn.SELU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, feats):
        return self.mlp(feats)


class WeightedAverage(nn.Module):
    def __init__(self, in_feats, out_feats=1):
        '''
        Parameters
        ----------
        in_feats : int
        out_feats : int
            out_feats should be 1 or equal to in_feats
        '''
        super(WeightedAverage, self).__init__()
        self.score = nn.Linear(in_feats, out_feats)

        torch.nn.init.normal_(self.score.weight, std=0.5)
        torch.nn.init.zeros_(self.score.bias)

    def forward(self, graph, feats_node):
        w = torch.sigmoid(self.score(feats_node))
        with graph.local_scope():
            graph.ndata['n'] = feats_node
            graph.ndata['w'] = w
            embedding = dgl.mean_nodes(graph, 'n', 'w')

        return embedding


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, n_head_list, extra_feats):
        super().__init__()
        self.n_conv = len(n_head_list)

        self.gat_list = nn.ModuleList()
        for i in range(self.n_conv):
            n_head = n_head_list[i]
            if i == 0:
                layer = dglnn.GATConv(in_feats, hidden_size, n_head)
            else:
                n_head_last = n_head_list[i - 1]
                layer = dglnn.GATConv(n_head_last * hidden_size, hidden_size, n_head)

            torch.nn.init.normal_(layer.attn_l, std=0.5)
            torch.nn.init.normal_(layer.attn_r, std=0.5)
            torch.nn.init.normal_(layer.fc.weight, std=0.5)
            self.gat_list.append(layer)

        self.readout = WeightedAverage(hidden_size * n_head_list[-1])

        self.mlp = MLPModel(hidden_size * n_head_list[-1] + extra_feats, 1, [2 * hidden_size, hidden_size])

    def forward(self, g, feats_node, feats_graph):
        x = feats_node
        for i in range(self.n_conv):
            x = F.relu(self.gat_list[i](g, x)).view(g.number_of_nodes(), -1)
        embedding = self.readout(g, x)
        return self.mlp(torch.cat((embedding, feats_graph), dim=1))
