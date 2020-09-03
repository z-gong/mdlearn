import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, extra_feats):
        super().__init__()
        self.gcn1 = GraphConv(in_feats, hidden_size)
        self.gcn2 = GraphConv(hidden_size, hidden_size)
        self.gcn3 = GraphConv(hidden_size, hidden_size)
        self.readout = Readout()
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
        x = self.gcn3(g, x)
        embedding = self.readout(g, x)
        y = torch.cat((embedding, feats_graph), dim=1)
        y = self.activation(self.linear1(y))
        y = self.activation(self.linear2(y))
        return self.linear3(y).view(-1)


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, n_head, extra_feats):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_size, n_head)
        self.gat2 = GATConv(n_head * hidden_size, hidden_size, n_head)
        self.gat3 = GATConv(n_head * hidden_size, hidden_size, n_head)

        # from dgllife.model.readout import *
        # self.readout = AttentiveFPReadout(n_head * hidden_size)
        # self.readout = MLPNodeReadout(n_head * hidden_size, n_head * hidden_size, n_head * hidden_size)
        self.readout = Readout()

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

    def forward(self, g, feats_node, feats_graph):
        x = self.activation(self.gat1(g, feats_node)).view(-1, self.hidden_size * self.n_head)
        x = self.activation(self.gat2(g, x)).view(-1, self.hidden_size * self.n_head)
        x = self.gat3(g, x).view(-1, self.hidden_size * self.n_head)
        embedding = self.readout(g, x)
        y = torch.cat((embedding, feats_graph), dim=1)
        y = self.activation(self.linear1(y))
        y = self.activation(self.linear2(y))
        return self.linear3(y).view(-1)


class Readout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, feats_node):
        with g.local_scope():
            g.ndata['h'] = feats_node
            embedding = dgl.readout_nodes(g, 'h', op='mean')
        return embedding
