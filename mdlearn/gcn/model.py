import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, extra_feats):
        super().__init__()
        self.gcn1 = dglnn.GraphConv(in_feats, hidden_size)
        self.gcn2 = dglnn.GraphConv(hidden_size, hidden_size)
        self.gcn3 = dglnn.GraphConv(hidden_size, hidden_size)
        self.readout = dglnn.AvgPooling()
        self.linear1 = nn.Linear(hidden_size + extra_feats, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.activation = F.selu
        for layer in self.gcn1, self.gcn2, self.gcn3, \
                     self.linear1, self.linear2, self.linear3:
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, g, feats_node, feats_graph):
        x = self.activation(self.gcn1(g, feats_node))
        x = self.activation(self.gcn2(g, x))
        x = self.activation(self.gcn3(g, x))
        embedding = self.readout(g, x)
        y = torch.cat((embedding, feats_graph), dim=1)
        y = self.activation(self.linear1(y))
        y = self.activation(self.linear2(y))
        return self.linear3(y)


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, n_head, extra_feats):
        super().__init__()
        self.gat1 = dglnn.GATConv(in_feats, hidden_size, n_head)
        self.gat2 = dglnn.GATConv(n_head * hidden_size, hidden_size, n_head)
        self.gat3 = dglnn.GATConv(n_head * hidden_size, hidden_size, n_head)

        # self.readout = dglnn.AvgPooling()
        self.readout = WeightedAverage(n_head * hidden_size)

        self.linear1 = nn.Linear(n_head * hidden_size + extra_feats, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size
        self.n_head = n_head
        self.activation = F.selu

        for layer in self.gat1, self.gat2, self.gat3:
            torch.nn.init.normal_(layer.attn_l, std=0.5)
            torch.nn.init.normal_(layer.attn_r, std=0.5)
            torch.nn.init.normal_(layer.fc.weight, std=0.5)

        for layer in self.linear1, self.linear2, self.linear3:
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, g, feats_node, feats_graph, target=None, weight=None):
        x = self.activation(self.gat1(g, feats_node)).view(-1, self.hidden_size * self.n_head)
        x = self.activation(self.gat2(g, x)).view(-1, self.hidden_size * self.n_head)
        x = self.activation(self.gat3(g, x)).view(-1, self.hidden_size * self.n_head)
        embedding = self.readout(g, x)
        y = torch.cat((embedding, feats_graph), dim=1)
        y = self.activation(self.linear1(y))
        y = self.activation(self.linear2(y))
        predict = self.linear3(y)
        if target is not None:
            return (predict - target) * weight
        else:
            return predict


class WeightedAverage(nn.Module):
    def __init__(self, in_feats):
        super(WeightedAverage, self).__init__()
        self.score = nn.Linear(in_feats, 1)

        torch.nn.init.normal_(self.score.weight, std=0.5)
        torch.nn.init.zeros_(self.score.bias)

    def forward(self, graph, feats_node):
        w = torch.sigmoid(self.score(feats_node))
        with graph.local_scope():
            graph.ndata['n'] = feats_node
            graph.ndata['w'] = w
            embedding = dgl.mean_nodes(graph, 'n', 'w')

        return embedding
