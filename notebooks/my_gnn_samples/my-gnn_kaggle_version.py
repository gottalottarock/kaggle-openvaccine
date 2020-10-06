#!/usr/bin/env python
# coding: utf-8

# In[3]:


# %% [code]
# %% [code]
# %% [code]
# %% [code]
# ------------------ install torch_geometric begin -----------------
try:
    import torch_geometric
except:
    import subprocess
    import torch

    nvcc_stdout = str(subprocess.check_output(['nvcc', '-V']))
    tmp = nvcc_stdout[nvcc_stdout.rfind('release') + len('release') + 1:]
    cuda_version = tmp[:tmp.find(',')]
    cuda = {
            '9.2': 'cu92',
            '10.1': 'cu101',
            '10.2': 'cu102',
            }

    CUDA = cuda[cuda_version]
    TORCH = torch.__version__.split('.')
    TORCH[-1] = '0'
    TORCH = '.'.join(TORCH)

    install1 = 'pip install torch-scatter==latest+' + CUDA + ' -f https://pytorch-geometric.com/whl/torch-' + TORCH + '.html'
    install2 = 'pip install torch-sparse==latest+' + CUDA + ' -f https://pytorch-geometric.com/whl/torch-' + TORCH + '.html'
    install3 = 'pip install torch-cluster==latest+' + CUDA + ' -f https://pytorch-geometric.com/whl/torch-' + TORCH + '.html'
    install4 = 'pip install torch-spline-conv==latest+' + CUDA + ' -f https://pytorch-geometric.com/whl/torch-' + TORCH + '.html'
    install5 = 'pip install torch-geometric'
    install6 = 'pip install neptune-client'

    subprocess.run(install1.split())
    subprocess.run(install2.split())
    subprocess.run(install3.split())
    subprocess.run(install4.split())
    subprocess.run(install5.split())
    subprocess.run(install6.split())


# In[4]:


import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader, Batch
from torch.optim import Adam
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import json
from scipy.linalg import block_diag


# In[5]:


def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == "("]
    closed = [idx for idx, i in enumerate(structure) if i == ")"]

    assert len(opened) == len(closed)

    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append((candidate, close_idx))
        assigned.append(close_idx)
        couples.append((close_idx, candidate))

    assert len(couples) == 2 * len(opened)

    return couples


def build_matrix(couples, size):
    mat = np.zeros((size, size))

    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1

    for i, j in couples:
        mat[i, j] = 2
        mat[j, i] = 2

    return mat


# In[6]:


def seq2nodes(sequence, loop_type, bpp):
    type_dict = {"A": 0, "G": 1, "U": 2, "C": 3}
    type_loop = {"f": 0, "s": 1, "h": 2, "m": 3, "i": 4, "t": 5}
    nodes = np.zeros((len(sequence), 4))
    loops = np.zeros((len(sequence), len(type_loop)))
    for i, (s, lt) in enumerate(zip(sequence, loop_type)):
        nodes[i, type_dict[s]] = 1
        loops[i, type_loop[lt]] = 1
    nodes = np.concatenate(
        [nodes, loops, np.stack([bpp.max(0), bpp.sum(0)], axis=1)], axis=-1
    )
    return nodes


def seq2edge_index(structure):
    couples = sorted(set(get_couples(structure)))
    couples = np.array(couples).T
    neig = np.array([np.arange(0, len(structure) - 1), np.arange(1, len(structure))])
    neig2 = neig[::-1, ::]
    if couples.shape[-1] > 0:
        edge_index = np.concatenate([couples, neig, neig2], axis=1)
    else:
        edge_index = np.concatenate([neig, neig2], axis=1)

    edges_type = np.array([1] * couples.shape[-1] + [2] * neig.shape[1] * 2)

    return edge_index, edges_type


def edge_index2features(edge_index, edges_type, node_features, bpp):
    edge_type_f = np.zeros((edge_index.shape[1], 2))
    for ty in [1, 2]:
        edge_type_f[:, ty - 1] = (edges_type == ty).astype(int)
    edge_direction = np.stack(
        [
            (
                edge_index[
                    1,
                ]
                - edge_index[
                    0,
                ]
                == 1
            ).astype(int),
            (
                edge_index[
                    0,
                ]
                - edge_index[
                    1,
                ]
                == 1
            ).astype(int),
        ]
    ).T
    bpps = np.array([bpp[i, j] for i, j in edge_index.T]).T
    edge_features = np.concatenate(
        [edge_type_f, edge_direction, np.expand_dims(bpps, 1)], axis=-1
    )
    return edge_features


def seq2edges(structure, node_features, bpp):
    edge_index, edges_type = seq2edge_index(structure)
    edge_features = edge_index2features(edge_index, edges_type, node_features, bpp)
    return edge_index, edge_features


def cg2edges(cg_graph, node2idx):
    features = []
    indexes = []
    for node_name, segments in cg_graph["nodes"].items():
        node_idx = node2idx[node_name]
        for seg in segments:
            for idx in range(*seg):
                indexes.append((node_idx, idx))
                features.append([1, 0, 0])
                indexes.append((idx, node_idx))
                features.append([0, 1, 0])
    for node_1, node_2 in cg_graph["edges"]:
        indexes.append((node2idx[node_1], node2idx[node_2]))
        features.append([0, 0, 1])
    indexes = np.array(indexes).T
    features = np.array(features)
    return indexes, features


def create_edges(structure, node_features, cg_graph, node2idx, bpp):
    edge_index_nuc, edge_features_nuc = seq2edges(structure, node_features, bpp)
    edge_index_bungle, edge_features_bungle = cg2edges(cg_graph, node2idx)
    edge_index = np.concatenate([edge_index_nuc, edge_index_bungle], axis=1)
    edge_from = node_features[
        edge_index[
            0,
        ]
    ]
    edge_to = node_features[
        edge_index[
            1,
        ]
    ]
    edge_features = block_diag(edge_features_nuc, edge_features_bungle)
    edge_features = np.concatenate([edge_features, edge_from, edge_to], axis=1)
    return edge_index, edge_features


def bungle_features(cg_nodes):
    type_dict = {"f": 0, "t": 1, "s": 2, "i": 3, "m": 4, "h": 5}
    features = np.zeros((len(cg_nodes), len(type_dict) + 1))
    for index, (node_name, segments) in enumerate(cg_nodes.items()):
        features[index][type_dict[node_name[0]]] = 1
        num_b = sum(seg[1] - seg[0] for seg in segments)
        features[index][-1] = num_b
    return features


def add_bungle_nodes(x, cg_graph):
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    cg_nodes = cg_graph["nodes"]
    x_len, x_dim = x.shape
    node2idx = {node: index for index, node in enumerate(cg_nodes, start=x_len)}
    cg_x = bungle_features(cg_nodes)
    cg_x = np.concatenate([cg_x, np.ones((cg_x.shape[0], 1))], axis=1)
    features = block_diag(x, cg_x)
    return features, node2idx


def create_single_graph(
    sequence, targets, errors, seq_scored, structure, loop_type, cg_graph, energy, bpp
):
    x = seq2nodes(sequence, loop_type, bpp)
    x, node2idx = add_bungle_nodes(x, cg_graph)
    edge_index, edge_features = create_edges(structure, x, cg_graph, node2idx, bpp)
    edge_index = torch.LongTensor(edge_index)
    edge_features = torch.FloatTensor(edge_features)
    x = torch.FloatTensor(x)
    data = Data(x, edge_index, edge_features)
    is_nuc = np.zeros(x.shape[0])
    is_nuc[: len(sequence)] = 1
    data.is_nuc = torch.BoolTensor(is_nuc)
    return data


def build_data(df, sampled_structures, max_graphs=5, target_cols=None, error_cols=None):
    target_cols = target_cols or []
    error_cols = error_cols or []
    assert len(error_cols) == len(target_cols)
    data_list = []
    for (id_, sequence, seq_scored), targets, errors in zip(
        tqdm(df[["id", "sequence", "seq_scored"]].values),
        df[target_cols].values,
        df[error_cols].values,
    ):
        sampled_structs = sampled_structures[id_]
        sampled_structs = sorted(
            sampled_structs, key=lambda d: d["energy"], reverse=False
        )[:max_graphs]
        bpp = np.load(BPP_FOLDER + id_ + ".npy")
        sampled_datas = []
        for sample_struct in sampled_structs:
            data = create_single_graph(
                sequence=sequence,
                targets=targets,
                errors=errors,
                seq_scored=seq_scored,
                bpp=bpp,
                **sample_struct
            )
            sampled_datas.append(data)
        data_batch = Batch.from_data_list(sampled_datas)
        del data_batch.batch
        data_batch.num_samples = len(sampled_datas)
        data_batch.seq_scored = seq_scored
        data_batch.energies = torch.FloatTensor([d["energy"] for d in sampled_structs])
        data_batch.seq_len = len(sequence)
        if targets is not None:
            targets = np.stack(targets).T
            errors = np.stack(errors).T
            targets = np.pad(
                targets,
                ((0, len(sequence) - targets.shape[0]), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
            errors = np.pad(
                errors,
                ((0, len(sequence) - errors.shape[0]), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
            targets = torch.FloatTensor(targets)
            errors = torch.FloatTensor(errors)

            targets = torch.stack([targets, errors], dim=2)
        else:
            targets = None
        mask = ~targets[:, 0, 0].isnan()
        data_batch.y = targets
        data_batch.mask = mask
        data_list.append(data_batch)
        assert len({data.seq_scored for data in data_list}) == 1
    return data_list


# In[14]:


MAP2D_FOLDER = "../data/nsp_distances_angles2/"
BPP_FOLDER = "../input/stanford-covid-vaccine/bpps/"
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
ERROR_COLS = ["reactivity_error", "deg_error_Mg_pH10", "deg_error_Mg_50C"]
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODQwOTM5MjItYWQ2Mi00ODRhLTgxOTUtMzA4NzNhMzI3OGIwIn0="
device = "cuda"


# In[11]:


class MyDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, sampled_structures, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.df_train = df_train
        self.df_val = df_val
        self.sampled_structures = sampled_structures

    def setup(self, stage=None):
        self.train_data_list = build_data(
            self.df_train,
            self.sampled_structures,
            target_cols=TARGET_COLS,
            error_cols=ERROR_COLS,
        )
        self.val_data_list = build_data(
            self.df_val,
            self.sampled_structures,
            target_cols=TARGET_COLS,
            error_cols=ERROR_COLS,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data_list, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_data_list, batch_size=self.batch_size)


# In[13]:


with open("../input/openvaccine-cv/train_cv_splits.json", "r") as f:
    cv_split = json.load(f)
# with open("../input/cg-graphs/cg_graphs/cg_train_graphs.json",'r') as f:
#     cg_graphs = json.load(f)
with open("../input/openvaccine-sampled-structures/sampled_structures.json", "r") as f:
    sampled_structures = json.load(f)
df = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)


# In[15]:


data_sanity = MyDataModule(df.iloc[:100], df.iloc[:100], sampled_structures, 16)
data_sanity.setup()
batch = next(data_sanity.val_dataloader().__iter__())


# In[16]:


def compute_MCRMSE_simple(pred, y, columns, per_column, prefix=""):
    losses = torch.sqrt(torch.mean((pred - y) ** 2, dim=0) + 1e-6)
    if not per_column:
        return losses.mean()
    metrics = {f"{prefix}mcrmse_{col}": loss for col, loss in zip(columns, losses)}
    metrics[f"{prefix}mcrmse"] = losses.mean()
    return metrics


# In[17]:


def compute_MCRMSE(pred, y, columns, per_column=False, prefix=""):
    mask = ~y[:, 0, 0].isnan()
    #     pred = pred[mask]
    y = y[:, :, 0][mask]
    if DEBUG:
        print("y-shape: ", y.shape)
        print("pred-shape ", pred.shape)
    return compute_MCRMSE_simple(pred, y, columns, per_column, prefix)


# In[18]:


def compute_gauss_loss(pred, y, columns, per_column=False, prefix=""):
    mask = ~y[:, 0, 0].isnan()
    #     pred = pred[mask]
    y_mean = y[:, :, 0][mask]
    errors = y[:, :, 1][mask]
    losses = torch.sqrt(torch.mean((y_mean - pred) ** 2 / (errors ** 2), dim=0) + 1e-6)
    if not per_column:
        return losses.mean()
    metrics = {f"{prefix}gauss_{col}": loss for col, loss in zip(columns, losses)}
    metrics[f"{prefix}gauss"] = losses.mean()
    return metrics


# In[19]:


num_layers = 4
dropout1 = 0.1
dropout2 = 0.1
dropout3 = 0.1
hidden_channels3 = 32


# In[20]:


class MapE2NxN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(MapE2NxN, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MyDeeperGCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        node_hidden_channels,
        num_edge_features,
        edge_hidden_channels,
        num_layers=num_layers,
        output_dim=3,
        seq_len=107,
    ):
        super(MyDeeperGCN, self).__init__()
        self.lstm1 = nn.LSTM(
            num_node_features,
            int(node_hidden_channels / 2),
            bidirectional=True,
            batch_first=True,
        )
        self.node_mlp = nn.Linear(num_node_features, node_hidden_channels)
        self.lstm2 = nn.LSTM(
            node_hidden_channels,
            int(node_hidden_channels / 2),
            bidirectional=True,
            batch_first=True,
        )
        self.node_hidden_channels = node_hidden_channels
        self.seq_len = seq_len
        #         self.node_encoder = ChebConv(node_hidden_channels, node_hidden_channels, T)
        self.edge_encoder = nn.Linear(num_edge_features, edge_hidden_channels)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = gnn.NNConv(
                node_hidden_channels,
                node_hidden_channels,
                MapE2NxN(
                    edge_hidden_channels,
                    node_hidden_channels * node_hidden_channels,
                    hidden_channels3,
                ),
            )
            norm = nn.LayerNorm(node_hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = gnn.DeepGCNLayer(
                conv, norm, act, block="res+", dropout=dropout1, ckpt_grad=i % 3
            )
            self.layers.append(layer)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            node_hidden_channels,
            nhead=4,
            dim_feedforward=node_hidden_channels,
        )
        self.lin = nn.Linear(node_hidden_channels, output_dim)
        self.dropout = nn.Dropout(dropout2)

    def forward(self, data):
        seq_len = data.seq_len[0]
        x = data.x.float()

        new_x = torch.zeros((x.shape[0], self.node_hidden_channels), device=x.device)
        x_lstm = x[data.is_nuc].view(-1, seq_len, x.shape[-1])
        x_lstm = self.lstm1(x_lstm)[0].reshape(-1, self.node_hidden_channels)
        x_mlp = self.node_mlp(x[~data.is_nuc])
        new_x[data.is_nuc] = x_lstm
        new_x[~data.is_nuc] = x_mlp
        x = new_x

        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # edge for paired nodes are excluded for encoding node
        #         seq_edge_index = edge_index[:, edge_attr[:,0] == 0]
        #         x = self.node_encoder(x, seq_edge_index)

        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)

        x = x[data.is_nuc]
        x = x.view(-1, seq_len, x.shape[-1])
        x = self.transformer_encoder(x)
        x = self.lstm2(x)[0]
        x = x.reshape(-1, x.shape[-1])
        x = self.lin(x)
        batch_is_nuc = data.batch[data.is_nuc]
        predictions = []
        for i, num_samples in zip(range(data.num_graphs), data.num_samples):
            mask = batch_is_nuc == i
            x_graph = x[mask]
            x_graph = x_graph.view(num_samples, seq_len, x.shape[-1]).mean(axis=0)
            predictions.append(x_graph)
        return torch.cat(predictions, dim=0)


# In[21]:


def sanity_check(batch, device=device):
    batch.to(device)
    model = MyDeeperGCN(
        batch.x.shape[1],
        100,
        batch.edge_attr.shape[1],
        16,
    )
    model.to(device)
    return model(batch)


# In[22]:

sanity_check(batch,'cpu').shape[0]/107



# In[24]:


class MyGNNLighting(pl.LightningModule):
    def __init__(self, model, seq_len=107, seq_scored=68, metrics_prefix=""):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.seq_scored = seq_scored
        self.metrics_prefix = metrics_prefix

    def forward(self, data):
        return self.model(data)

    def predict(self, batch, device, mode="test"):
        self.eval()
        with torch.no_grad():
            batch = batch.to(device)
            return self.forward(batch).detach()

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        y = batch.y
        pred = pred[batch.mask]
        loss = self.compute_loss(pred, y)
        result = pl.TrainResult()
        result.log_dict(
            {"train_" + self.metrics_prefix + "gauss": loss}, on_step=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        y = batch.y
        pred = pred[batch.mask]
        return (pred, y)

    def validation_epoch_end(self, validation_step_outputs):
        if DEBUG:
            print("num step outputs :", len(validation_step_outputs))
        pred, y = zip(*validation_step_outputs)
        if DEBUG:
            print("num pred: ", len(pred))
        pred = torch.cat(pred, dim=0)
        y = torch.cat(y, dim=0)
        if DEBUG:
            print("val y-hsape: ", y.shape)
            print("val pred-shape: ", pred.shape)
        metrics = self.compute_val_metrics(pred, y)
        result = pl.EvalResult(
            checkpoint_on=metrics["val_" + self.metrics_prefix + "mcrmse"]
        )
        result.log_dict(
            metrics, on_step=False, on_epoch=True, logger=True, prog_bar=False
        )
        return result

    def compute_val_metrics(self, pred, y):
        metrics = compute_MCRMSE(
            pred,
            y,
            columns=TARGET_COLS,
            per_column=True,
            prefix="val_" + self.metrics_prefix,
        )
        metrics.update(
            compute_gauss_loss(
                pred,
                y,
                columns=TARGET_COLS,
                per_column=True,
                prefix="val_" + self.metrics_prefix,
            )
        )
        return metrics

    def compute_loss(self, pred, y):
        return compute_gauss_loss(pred, y, columns=[], per_column=False, prefix="")

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        return opt


# In[25]:


def get_callbacks(fold_i):
    mc_cb = pl.callbacks.ModelCheckpoint(
        filepath="/kaggle/working/models/{epoch}",
        mode="min",
        save_top_k=1,
        prefix=f"{fold_i}_",
        save_weights_only=True,
    )
    return mc_cb


def get_best_model_fn(mc_cb, fold_i):
    for k, v in mc_cb.best_k_models.items():
        if (v == mc_cb.best) and Path(k).stem.startswith(str(fold_i)):
            return k


# In[26]:


def sanity_check_2(batch, device="cuda"):
    batch.to(device)
    model = MyDeeperGCN(
        batch.x.shape[1],
        100,
        batch.edge_attr.shape[1],
        16,
    )
    model.to(device)
    module = MyGNNLighting(model)
    return (
        module.training_step(batch, 0),
        module.validation_step(batch, 0),
    )


# In[27]:

sanity_check_2(batch, "cpu")


# In[29]:


def memory_check(batch_size=16, device=device):
    data_list = build_data(
        df,
        sampled_structures,
        target_cols=TARGET_COLS,
        error_cols=ERROR_COLS,
    )
    batch_data_list = sorted(data_list, key=lambda a: a.x.shape, reverse=True)[:5]
    batch = Batch.from_data_list(batch_data_list)
    print(batch.x.shape, batch.num_samples)
    model = MyDeeperGCN(
        batch.x.shape[1],
        100,
        batch.edge_attr.shape[1],
        16,
    )
    model.to(device)
    module = MyGNNLighting(model)
    module.to(device)
    batch.to(device)
    opt = module.configure_optimizers()
    loss = module.training_step(batch, 0)
    loss.backward()
    opt.step()
    opt.zero_grad()


# In[31]:


memory_check()


# In[32]:


# df_filter = df[df.SN_filter==1]
# for col_error in ERROR_COLS:
#     errors = np.stack(df_filter[col_error].values)
#     errors.reshape(-1)
#     mean_error = errors.mean(axis=None)
#     var = ((errors-mean_error)**2).sum()/(errors.shape[0] - 1)
#     print(col_error, mean_error, var)
#     errors = np.stack(df[col_error].values)
#     errors = errors / mean_error
#     df[col_error] = list(errors)


# In[33]:


# df_filter = df[df.SN_filter==1]
# for col_error in ERROR_COLS:
#     errors = np.stack(df_filter[col_error].values)
#     errors.reshape(-1)
#     mean_error = errors.mean(axis=None)
#     var = ((errors-mean_error)**2).sum()/(errors.shape[0] - 1)
#     print(col_error, mean_error, var)


# In[37]:


def cv(df, experiment_name="test_cv", num_epochs=100):
    subprocess.run('rm -r ./models'.split())
    logger = pl.loggers.neptune.NeptuneLogger(
        NEPTUNE_API_TOKEN,
        "gottalottarock/openVaccine",
        experiment_name=experiment_name,
        close_after_fit=False,
        upload_source_files=["__notebook_source__.ipynb",'__notebook__.ipynb'],
    )
    cv_val_predictions = []
    df_filter = df[df.SN_filter == 1]
    for fold_i, (train_ids, val_ids) in enumerate(cv_split):
        df_train = df[df.id.isin(set(train_ids))]
        df_val = df_filter[df_filter.id.isin(set(val_ids))]
        data = MyDataModule(df_train, df_val, sampled_structures, 16)
        data.setup()
        model = MyDeeperGCN(
            data.train_data_list[0].x.shape[1],
            100,
            data.train_data_list[0].edge_attr.shape[1],
            16,
        )
        module = MyGNNLighting(model, metrics_prefix="cv" + str(fold_i) + "_")
        mc_cb = get_callbacks(fold_i)
        trainer = pl.trainer.Trainer(
            checkpoint_callback=mc_cb, logger=logger, max_epochs=num_epochs, gpus=1
        )
        trainer.fit(module, data)
        torch.cuda.empty_cache()
        best_model_path = mc_cb.best_model_path
        print(best_model_path)
        module.load_state_dict(torch.load(best_model_path)["state_dict"])
        module.to("cuda")
        predictions = []
        for batch in data.val_dataloader():
            pred = module.predict(batch, device="cuda").cpu().numpy()
            predictions.append(pred)
        predictions = np.concatenate(predictions, axis=0).T
        df_val_pred = pd.DataFrame(df_val.id)
        for col_name, pred_col in zip(TARGET_COLS, predictions):
            pred_col = pred_col.reshape(-1, 107)
            print(pred_col.shape, df_val.shape)
            assert pred_col.shape[0] == df_val_pred.shape[0]
            df_val_pred["cv_" + col_name] = list(pred_col)
        cv_val_predictions.append(df_val_pred)
    cv_val_predictions = pd.concat(cv_val_predictions, axis=0)
    cv_val_predictions.to_json("val_predictions.json", orient="records", lines=True)
    logger.experiment.log_artifact("val_predictions.json")
    cv_val_predictions = cv_val_predictions.set_index("id")
    cv_val_predictions = cv_val_predictions.loc[df_filter.id]
    vals = []
    ys = []
    for col in TARGET_COLS:
        y = np.stack(np.array(l) for l in df_filter[col].values)
        ys.append(y.reshape(-1))
        val = np.stack(np.array(l) for l in cv_val_predictions["cv_" + col].values)[
            :, :68
        ]
        vals.append(val.reshape(-1))
    vals = np.stack(vals).T
    ys = np.stack(ys).T
    vals = torch.FloatTensor(vals)
    ys = torch.FloatTensor(ys)
    metrics = compute_MCRMSE_simple(
        vals, ys, columns=TARGET_COLS, per_column=True, prefix="cv_"
    )
    logger.log_metrics(metrics)
    del module
    torch.cuda.empty_cache()
    logger.experiment.log_artifact("val_predictions.json")
    logger.experiment.log_artifact("./")
    logger.experiment.stop()


# In[38]:


DEBUG=False


# In[39]:


cv(df, "deepergnn_transformer_lstm_on_my_graph_bpp_gauss_loss", num_epochs=200)


# In[ ]:



