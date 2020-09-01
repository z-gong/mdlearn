import pandas as pd
from rdkit.Chem import AllChem as Chem
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv
import matplotlib.pyplot as plt

from mdlearn.gcn.featurizer import rdk2dgl, get_extra_features
from mdlearn.gcn.model import GCNModel, GATModel

graph_list = []
extra_list = []
y_list = []

df = pd.read_csv('../data/nist-CH-tvap.txt', sep='\s+')
for row in df.itertuples():
    rdkm = Chem.MolFromSmiles(row.SMILES)
    rdkm = Chem.AddHs(rdkm)
    graph_list.append(rdk2dgl(rdkm))
    extra_list.append(get_extra_features(rdkm))
    y_list.append(row.tvap)

for graph in graph_list:
    graph.ndata['y'] = torch.zeros(graph.num_nodes(), 16)

print(graph_list[0])
print(graph_list[0].ndata['x'])
print(y_list[0])

batch_graph = dgl.batch(graph_list[:])
print(batch_graph)

from mdlearn.preprocessing import Scaler
fp_dict = {}
fp_file = '../run/out-ch-tvap/fp_simple'
d = pd.read_csv(fp_file, sep='\s+', header=None, names=['SMILES', 'fp'], dtype=str)
for i, row in d.iterrows():
    if row.SMILES not in fp_dict:
        fp_dict[row.SMILES] = list(map(float, row.fp.split(',')))
    else:
        fp_dict[row.SMILES] += list(map(float, row.fp.split(',')))
fp_list = []
for smiles in df.SMILES:
    fp_list.append(fp_dict[smiles])
fp_array = np.array(fp_list, dtype=np.float32)
scaler = Scaler()
scaler.fit(fp_array)
fp_array = scaler.transform(fp_array)

fp_simple = torch.tensor(fp_array)

y = torch.tensor(y_list[:], dtype=torch.float32)
feats_extra = torch.tensor(extra_list, dtype=torch.float32)

model = GCNModel(batch_graph.ndata['x'].shape[-1], batch_graph.ndata['y'].shape[-1])
print(model)
for name, param in model.named_parameters():
        print(name, param.data.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
for epoch in range(400):
    model.train()
    predict = model(batch_graph, batch_graph.ndata['x'], feats_extra)
    loss = F.mse_loss(predict, y)
    print(epoch, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
print(predict)
plt.plot(y_list, predict.detach().numpy(), '.')
plt.plot([100, 800], [100, 800], '-')
plt.show()
