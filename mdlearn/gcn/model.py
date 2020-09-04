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
        self.readout = MeanReadout()
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
        self.gat1 = GATConv(in_feats, hidden_size, n_head)
        self.gat2 = GATConv(n_head * hidden_size, hidden_size, n_head)
        self.gat3 = GATConv(n_head * hidden_size, hidden_size, n_head)

        # self.readout = AttentiveFPReadout(n_head * hidden_size, num_timesteps=1)
        self.readout = MeanReadout()

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


class MeanReadout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, feats_node):
        with g.local_scope():
            g.ndata['h'] = feats_node
            embedding = dgl.readout_nodes(g, 'h', op='mean')
        return embedding


class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        # self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """Perform one-step readout

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')

            return g_repr
            # context = F.elu(g_repr)
            #
            # if get_node_weight:
            #     return self.gru(context, g_feats), g.ndata['a']
            # else:
            #     return self.gru(context, g_feats)


class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class computes graph representations out of node features.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.

        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.mean_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats
