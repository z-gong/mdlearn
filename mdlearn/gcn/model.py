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


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, extra_feats):
        super().__init__()
        gcn1 = dglnn.GraphConv(in_feats, hidden_size)
        gcn2 = dglnn.GraphConv(hidden_size, hidden_size)
        gcn3 = dglnn.GraphConv(hidden_size, hidden_size)

        for layer in gcn1, gcn2, gcn3:
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.zeros_(layer.bias)

        self.conv = nn.Sequential(gcn1, nn.SELU(), gcn2, nn.SELU(), gcn3)
        self.readout = dglnn.AvgPooling()
        self.mlp = MLPModel(hidden_size + extra_feats, 1, [2 * hidden_size, hidden_size])

    def forward(self, g, feats_node, feats_graph):
        x = F.selu(self.conv(g, feats_node))
        embedding = self.readout(g, x)
        return self.mlp(torch.cat((embedding, feats_graph), dim=1))


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
    def __init__(self, in_feats, hidden_size, n_head, extra_feats):
        super().__init__()
        self.gat1 = dglnn.GATConv(in_feats, hidden_size, n_head)
        self.gat2 = dglnn.GATConv(n_head * hidden_size, hidden_size, n_head)
        self.gat3 = dglnn.GATConv(n_head * hidden_size, hidden_size, 1)

        for layer in self.gat1, self.gat2, self.gat3:
            torch.nn.init.normal_(layer.attn_l, std=0.5)
            torch.nn.init.normal_(layer.attn_r, std=0.5)
            torch.nn.init.normal_(layer.fc.weight, std=0.5)

        self.readout = WeightedAverage(hidden_size)

        self.mlp = MLPModel(hidden_size + extra_feats, 1, [2 * hidden_size, hidden_size])

    def forward(self, g, feats_node, feats_graph):
        x = F.selu(self.gat1(g, feats_node)).view(g.number_of_nodes(), -1)
        x = F.selu(self.gat2(g, x)).view(g.number_of_nodes(), -1)
        x = F.selu(self.gat3(g, x)).view(g.number_of_nodes(), -1)
        embedding = self.readout(g, x)
        return self.mlp(torch.cat((embedding, feats_graph), dim=1))
