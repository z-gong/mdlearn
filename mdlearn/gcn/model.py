import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super().__init__()
        self.gcn1 = GraphConv(in_feats, hidden_size)
        self.gcn2 = GraphConv(hidden_size, hidden_size)
        self.gcn3 = GraphConv(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size + 3, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        for layer in self.gcn1, self.gcn2, self.gcn3, \
                     self.linear1, self.linear2, self.linear3:
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.normal_(layer.bias, mean=0.01, std=0.001)

    def forward(self, g, feats_node, feats_graph):
        x = F.selu(self.gcn1(g, feats_node))
        x = F.selu(self.gcn2(g, x))
        g.ndata['y'] = self.gcn3(g, x)
        y_sum = dgl.readout_nodes(g, 'y', op='sum')
        y = torch.cat((y_sum, feats_graph), dim=1)
        y = F.selu(self.linear1(y))
        y = F.selu(self.linear2(y))
        return self.linear3(y).view(-1)


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super().__init__()
        self.gcn1 = GATConv(in_feats, hidden_size, 3)
        self.gcn2 = GATConv(3 * hidden_size, hidden_size, 3)
        self.gcn3 = GATConv(3 * hidden_size, hidden_size, 1)

        self.bias1 = nn.Parameter(torch.Tensor(3, hidden_size))
        self.bias2 = nn.Parameter(torch.Tensor(3, hidden_size))
        self.bias3 = nn.Parameter(torch.Tensor(hidden_size))

        self.linear1 = nn.Linear(hidden_size + 3, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        for layer in self.gcn1, self.gcn2, self.gcn3:
            torch.nn.init.normal_(layer.attn_l, std=0.5)
            torch.nn.init.normal_(layer.attn_r, std=0.5)
            torch.nn.init.normal_(layer.fc.weight, std=0.5)

        torch.nn.init.normal_(self.bias1, mean=0.01, std=0.001)
        torch.nn.init.normal_(self.bias2, mean=0.01, std=0.001)
        torch.nn.init.normal_(self.bias3, mean=0.01, std=0.001)

        for layer in self.linear1, self.linear2, self.linear3:
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.normal_(layer.bias, mean=0.01, std=0.001)

    def forward(self, g, feats_node, feats_graph):
        x = F.relu(self.gcn1(g, feats_node) + self.bias1).view(-1, 48)
        x = F.relu(self.gcn2(g, x) + self.bias2).view(-1, 48)
        g.ndata['y'] = self.gcn3(g, x).view(-1, 16) + self.bias3
        y = dgl.readout_nodes(g, 'y', op='mean')
        y = torch.cat((y, feats_graph), dim=1)
        y = F.relu(self.linear1(y))
        y = F.relu(self.linear2(y))
        return self.linear3(y).flatten()
