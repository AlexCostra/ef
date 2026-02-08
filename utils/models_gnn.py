from sqlite3 import apilevel
from telnetlib import EXOPL
from tkinter.messagebox import NO
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
# from loader import BioDataset
# from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
class DrugGAT(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=64, num_layers=3, heads=4, dropout=0.3):
        super(DrugGAT, self).__init__()

        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        # 每层输出 hidden_dim * heads（拼接）
        for _ in range(1,num_layers - 1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # 最后一层 GAT 仍然输出 64×4
        self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # FeedForward 网络：将 64×4 → 64
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim)
        )

    def forward(self, x, edge_index):
        """
        x: [num_drugs, in_dim]
        edge_index: [2, num_edges]
        """
        x = F.relu(self.input_linear(x))
        #print('x1',x.shape)
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            #print('xxxxxxxxxxxx',x.shape)
            x = F.elu(x)
            # 残差连接时需要维度匹配
            if residual.shape[-1] != x.shape[-1]:
                residual = F.pad(residual, (0, x.shape[-1] - residual.shape[-1]))
                #print('residual',residual.shape)
            #print('residualelse', residual.shape)
            x = self.norms[i](x + residual)
            x = self.dropout(x)

        # FeedForward 映射回 64维
        x = self.feed_forward(x)

        return x  # [num_drugs, 64]
class DrugSyergisticGAT(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=64, num_layers=3, heads=4, dropout=0.3):
        super(DrugSyergisticGAT, self).__init__()

        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        # 每层输出 hidden_dim * heads（拼接）
        for _ in range(1,num_layers - 1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # 最后一层 GAT 仍然输出 64×4
        self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # FeedForward 网络：将 64×4 → 64
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim)
        )

    def forward(self, x, edge_index):
        """
        x: [num_drugs, in_dim]
        edge_index: [2, num_edges]
        """
        x = F.relu(self.input_linear(x))

        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            x = F.elu(x)
            # 残差连接时需要维度匹配
            if residual.shape[-1] != x.shape[-1]:
                residual = F.pad(residual, (0, x.shape[-1] - residual.shape[-1]))
            x = self.norms[i](x + residual)
            x = self.dropout(x)

        # FeedForward 映射回 64维
        x = self.feed_forward(x)

        return x  # [num_drugs, 64]
class DrugGCN(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=64, num_layers=3):
        super(DrugGCN, self).__init__()

        # 如果药物本身已有输入特征，就用线性层统一到 hidden_dim
        self.input_linear = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        """
        x: [num_drugs, in_dim]   药物原始特征（不是id）
        edge_index: [2, num_edges]  药物-药物交互图
        """
        x = self.input_linear(x)  # 映射到 hidden_dim
        x = F.relu(x)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x   # [num_drugs, out_dim]
class DrugSyergisticGCN(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=64, num_layers=3):
        super(DrugSyergisticGCN, self).__init__()

        # 如果药物本身已有输入特征，就用线性层统一到 hidden_dim
        self.input_linear = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        """
        x: [num_drugs, in_dim]   药物原始特征（不是id）
        edge_index: [2, num_edges]  药物-药物交互图
        """
        x = self.input_linear(x)  # 映射到 hidden_dim
        x = F.relu(x)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x   # [num_drugs, out_dim]
class GATConv1(MessagePassing):
    def __init__(self, emb_dim, p_or_m, device, heads=2, negative_slope=0.2, aggr = "add", input_layer = False):
        super(GATConv1, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, emb_dim))

        self.device = device
        self.plus_or_minus = p_or_m

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.node_embeddings = None

        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        #add self loops in the edge space
        #edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        print('mlp',edge_index.shape)
        print('原始形状', x.shape)
        '''
        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))
        '''
        print('embedding形状', x.shape)
        #x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        print('embedding形状1', x.shape)
        #print('zxcvbnm',x.shape)#torch.Size([8384, 2, 64])
        #out = self.propagate(self.aggr, edge_index, x=x)
        #print("edge_index max:", edge_index.max())
        #print("edge_index min:", edge_index.min())
        #print("x.shape[0]:", x.shape[0])
        #print("Number of nodes in edge_index:", edge_index.max().item() + 1)
        #print("Number of nodes in x:", x.size(0))
        out = self.propagate(edge_index, x=x)
        self.node_embeddings = out
        return out

    def message(self, edge_index, x_i, x_j):
        print(x_i.shape)
        print(x_j.shape)
        alpha = (x_j * self.att).sum(dim=-1)


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        if self.plus_or_minus == 'minus':
            out = x_i - x_j * alpha.view(-1, self.heads, 1)
            return out 

        elif self.plus_or_minus == 'plus':
            out = x_i + x_j * alpha.view(-1, self.heads, 1)
            return out

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """
    def __init__(self, p_or_m, device, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, p_or_m, device, input_layer = input_layer))

    def forward(self, x, edge_index):
        h_list = [x]
        print('x',x)
        print('1111',h_list[0].shape)
        #print('xs',x.shape)
        for layer in range(self.num_layer):
            print('zxcvbnm',h_list[layer])
            h = self.gnns[layer](h_list[layer], edge_index)
            print(h)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return node_representation


if __name__ == "__main__":
    pass