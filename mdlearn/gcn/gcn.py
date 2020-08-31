import pandas as pd
from rdkit.Chem import AllChem as Chem
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GATConv
import matplotlib.pyplot as plt


def mk_graph(rdk_mol):
    bonds = []
    for bond in rdk_mol.GetBonds():
        id1, id2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((id1, id2))
        bonds.append((id2, id1))
    edges = list(zip(*bonds))
    u = th.tensor(edges[0])
    v = th.tensor(edges[1])

    graph = dgl.graph((u, v))
    graph.ndata['x'] = th.zeros(graph.num_nodes(), 6)
    graph.ndata['y'] = th.zeros(graph.num_nodes(), 16)

    for atom in rdk_mol.GetAtoms():
        idx = 0
        if atom.GetAtomicNum() == 6:
            idx = 1
        if atom.GetIsAromatic():
            idx = 2
        else:
            n_neigh = len(atom.GetNeighbors())
            if n_neigh == 3:
                idx = 3
            if n_neigh == 2:
                idx = 4
        graph.ndata['x'][atom.GetIdx()][idx] = 1
        if atom.IsInRing():
            graph.ndata['x'][-1] = 1

    return graph


def get_shortest_wiener(rdk_mol):
    wiener = 0
    max_shortest = 0
    mol = Chem.RemoveHs(rdk_mol)
    n_atoms = mol.GetNumAtoms()
    for i in range(0, n_atoms):
        for j in range(i + 1, n_atoms):
            shortest = len(Chem.GetShortestPath(mol, i, j)) - 1
            wiener += shortest
            max_shortest = max(max_shortest, shortest)
    return max_shortest, int(np.log(wiener) * 10)


graph_list = []
tvap_list = []
heavy_list = []
shortest_list = []
rotatable_list = []

df = pd.read_csv('../../data/nist-CH-tvap.txt', sep='\s+')
for row in df.itertuples():
    m = Chem.MolFromSmiles(row.SMILES)
    m = Chem.AddHs(m)
    graph_list.append(mk_graph(m))
    tvap_list.append(row.tvap)
    shortest_list.append(get_shortest_wiener(m)[0])
    heavy_list.append(m.GetNumHeavyAtoms())
    rotatable_list.append(Chem.CalcNumRotatableBonds(m))

print(graph_list[0])
print(graph_list[0].ndata['x'])
print(tvap_list[0])

batch_graph = dgl.batch(graph_list[:])
print(batch_graph)

tvap = th.tensor(tvap_list[:], dtype=th.float32)
shortest = th.tensor(shortest_list[:], dtype=th.float32).view(-1, 1)
heavy = th.tensor(heavy_list[:], dtype=th.float32).view(-1, 1)
rotatable = th.tensor(rotatable_list[:], dtype=th.float32).view(-1, 1)
feats_extra = th.cat((shortest, heavy, rotatable), dim=1)


class Net(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super().__init__()
        self.gcn1 = GraphConv(in_feats, hidden_size)
        self.gcn2 = GraphConv(hidden_size, hidden_size)
        self.gcn3 = GraphConv(hidden_size, hidden_size)
        # self.gcn1 = GATConv(in_feats, hidden_size, 1)
        # self.gcn2 = GATConv(hidden_size, hidden_size, 1)
        # self.gcn3 = GATConv(hidden_size, hidden_size, 1)
        self.linear1 = nn.Linear(hidden_size + 3, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.linear3 = nn.Linear(2 * hidden_size, 1)

    def forward(self, g, feats_node, feats_graph):
        x = F.relu(self.gcn1(g, feats_node))
        x = F.relu(self.gcn2(g, x))
        g.ndata['y'] = self.gcn3(g, x)
        y = dgl.readout_nodes(g, 'y', op='mean')
        y = th.cat((y, feats_graph), dim=1)
        y = F.relu(self.linear1(y))
        y = F.relu(self.linear2(y))
        return self.linear3(y).flatten()


net = Net(6, 16)
print(net)

optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
for epoch in range(200):
    net.train()
    val = net(batch_graph, batch_graph.ndata['x'], feats_extra)
    loss = F.mse_loss(val, tvap)
    print(epoch, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
print(val)
plt.plot(tvap_list, val.detach().numpy(), '.')
plt.plot([100, 800], [100, 800], '-')
plt.show()
