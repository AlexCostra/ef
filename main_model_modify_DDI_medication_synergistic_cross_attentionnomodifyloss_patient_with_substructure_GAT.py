from collections import defaultdict
from copy import deepcopy
import os
from tkinter.messagebox import NO
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from dnc import DNC
from layers import FALayer, GCNLayer
import dgl
import math
import pdb
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor
import torch
from GNN import GNNGraph, GNN
from SetTransformer import SAB

molecule_para = {
    'num_layer': 4, 'emb_dim': 64, 'graph_pooling': 'mean',
    'drop_ratio': 0.7, 'gnn_type': 'gin', 'virtual_node': False
}
substruct_para = {
    'num_layer': 4, 'emb_dim': 64, 'graph_pooling': 'mean',
    'drop_ratio': 0.7, 'gnn_type': 'gin', 'virtual_node': False
}

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)#131xD   451xD    16x451
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        #print('ATTN',Attn.shape)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)


        # print(Attn[0])
        # print(mask[0])
        fix_feat = torch.matmul(fix_feat.T, fix_feat)
        fix_feat=torch.sqrt(fix_feat)
        fix_feat = fix_feat / fix_feat.sum(dim=0, keepdim=True)

        #print('fix_feat',fix_feat.shape)
        #fix_feat = torch.diag(fix_feat)
        #print('AA', fix_feat.shape)
        other_feat = torch.matmul(fix_feat, other_feat)

        O = torch.matmul(Attn, other_feat)

        return O
class CrossAttentionPredictor(nn.Module):
    def __init__(self, d_model, n_heads, n_med):
        super(CrossAttentionPredictor, self).__init__()
        self.n_med = n_med
        self.d_model = d_model
        self.n_heads = n_heads

        # 多头交叉注意力层
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # LayerNorm 层
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

        # 全连接层：将药物相关的患者表示映射为131维多标签输出
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_med)
        )

        # Sigmoid 输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, normed_query, normed_MPNN_emb):
        """
        normed_query: (B, D) → 患者特征
        normed_MPNN_emb: (N_med, D) → 药物特征
        """
        B, D = normed_query.shape
        N_med = normed_MPNN_emb.shape[0]

        # 添加序列维度，符合 MultiheadAttention 输入格式
        q = self.ln_q(normed_query).unsqueeze(1)     # (B, 1, D)
        k = normed_MPNN_emb.unsqueeze(0).expand(B, N_med, D)  # (B, N_med, D)
        v = k  # 通常 cross-attn 的 V 与 K 相同

        # Cross-Attention：患者查询药物表示
        attn_output, attn_weights = self.cross_attn(q, k, v)  # attn_output: (B, 1, D)
        attn_output = self.ln_out(attn_output.squeeze(1))     # (B, D)

        # 全连接层 + Sigmoid
        logits = self.fc(attn_output)   # (B, N_med)
        probs = self.sigmoid(logits)    # (B, N_med)

        return probs, attn_weights


class Fagcn_main(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=1):
        super().__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))
            # self.layers.append(GCNLayer(self.g, hidden_dim, dropout))

        self.t0 = nn.Linear(in_dim, hidden_dim)
        self.t1 = nn.Linear(hidden_dim, out_dim)
        self.context_attn = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t0.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.context_attn.weight, gain=1.414)

    def forward(self, h, context=None):
        raw = h
        for i in range(self.layer_num):
            m = self.layers[i](h)  # eq 6 h:torch.Size([55544, 64])
            #print('m',m.shape)
            # update h with context
            attn = torch.tanh(self.context_attn(context))  # eq 7
            m = attn * m

            h = self.eps * h + m  # eq 8 torch.Size([55544, 64])
            #print(self.eps)#0.3
            #print(h.shape)
            h = torch.relu(h)#torch.Size([55544, 64])
            #print('111111',h.shape)
        return h


class MolecularGraphNeuralNetwork_record(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device, fingers, avg_projection, g=None, args=None):
        super().__init__()
        self.device = device
        self.avg_projection = avg_projection.to(device)
        self.embed_fingerprint = nn.Embedding(N_fingerprint + 1, dim, padding_idx=N_fingerprint).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = fingers
        self.fingerprints = torch.cat(fingerprints)
        self.molecular_sizes = molecular_sizes
        if g is None:
            adjacencies = [adjacencies[i].cpu() for i in range(len(adjacencies))]
            self.adjacencies = self.pad(adjacencies, 0)
            # build graph for fagcn
            edges = self.adjacencies.nonzero()
            num_nodes = self.fingerprints.shape[0]
            U, V = edges[:, 0], edges[:, 1]
            g = dgl.graph((U, V), num_nodes=num_nodes).to('cpu')
            g = dgl.to_simple(g)
            g = dgl.remove_self_loop(g)
            g = dgl.to_bidirected(g)
            dill.dump(g, open("g.pkl", 'wb'))

        g = g.to(self.device)
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        self.encoder: Fagcn_main = Fagcn_main(
            g, dim, dim, dim, dropout=0.5, eps=0.3, layer_num=2)

        self.beta = 1
        Nmed = avg_projection.shape[0]
        self.viewcat = nn.Linear(2 * dim, dim)
        self.fc_selector = nn.Linear(dim, dim)



    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N)))  # .to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def max(self, vectors, axis):
        max_vectors = [torch.max(v, 0).values for v in torch.split(vectors, axis)]
        return torch.stack(max_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, *rec_args):
        """
        visit_emb(:Tensor) with shape (Nbatch, dim)
        labels(:Tensor) with shape (Nbatch, Nmed) each row is a mult-hot vector
        """

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(self.fingerprints)#替换成药物分子子结构
        context = self.update_recemb(*rec_args)
        dim = context.shape[1]
        context = context.repeat(1, self.molecular_sizes[0])
        context = context.reshape(-1, dim)
        fingerprint_vectors = self.encoder(fingerprint_vectors, context)  # eq 7, 8 9

        # Molecular vector by sum or mean of the fingerprint vectors
        molecular_vectors = self.sum(fingerprint_vectors, self.molecular_sizes)
        #print('self.avg_projection',self.avg_projection.shape)#torch.Size([131, 131])
        #print('molecular_vectors', molecular_vectors.shape)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
        mpnn_emb = torch.mm(self.avg_projection, molecular_vectors)
        #print('mmpnn', mpnn_emb.shape)
        return mpnn_emb, 0

    def update_recemb(self, embeddings, med2diag, med2pro, ehradj_idx):
        diag_emb, pro_emb = embeddings[0], embeddings[1]
        Ndiag, Npro = med2diag.shape[1], med2pro.shape[1]
        diag_emb = diag_emb(torch.arange(Ndiag).to(self.device))
        pro_emb = pro_emb(torch.arange(Npro).to(self.device))
        # pdb.set_trace()
        med_diagview = torch.mm(med2diag, diag_emb)
        med_proview = torch.mm(med2pro, pro_emb)
        med_rec = torch.cat((med_diagview, med_proview), -1)
        med_rec = self.viewcat(med_rec)
        med_rec = med_rec + self.cooccu_aug(med_rec, ehradj_idx)
        return med_rec

    def cooccu_aug(self, context, ehr_adj):
        aug_emb = torch.mm(ehr_adj, context)
        sel_attn = self.fc_selector(context.clone()).tanh()
        aug_emb = sel_attn * aug_emb
        return aug_emb


class main_model(nn.Module):
    def __init__(self, vocab_size, ddi_adj, ddi_synergistic_adj,encoder, ddi_encoder,ddi_synergistic_encoder,
                 emb_dim=256,
                 device=torch.device('cpu:0'),
                 use_aug=True,
                 ehr_adj=None,
                 med2diag=None,
                 med2pro=None,args=None,substruct_num=64,
                 global_para=molecule_para,
                 substruct_para=substruct_para,substruct_dim=64,
                 global_dim=64):
        super().__init__()
        self.use_aug = use_aug
        self.args = args
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_synergistic_adj = torch.FloatTensor(ddi_synergistic_adj).to(device)
        self.med2diag = med2diag
        self.med2pro = med2pro
        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device) if ehr_adj is not None else None
        self.device = device
        self.vocab_size = vocab_size

        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )

        #self.CrossAttentionPredictor=CrossAttentionPredictor(emb_dim,4,131)
        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] + 1, emb_dim, padding_idx=vocab_size[i]) for i in range(2)])
        self.embeddings.append(
            nn.Embedding(vocab_size[2] + 1, emb_dim, padding_idx=vocab_size[2])  # 新增药物 embedding
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim, batch_first=True),  # diag
            nn.GRU(emb_dim, emb_dim, batch_first=True),  # pro
            nn.GRU(emb_dim, emb_dim, batch_first=True)  # med (新增)
        ])
        #self.query_proj = nn.Linear(emb_dim, emb_dim)
        #self.key_proj = nn.Linear(emb_dim, emb_dim)
        #self.value_proj = nn.Linear(emb_dim, emb_dim)
        self.cross_fusion_diag = nn.Linear(3 * emb_dim, emb_dim)
        self.cross_fusion_pro = nn.Linear(3 * emb_dim, emb_dim)
        self.cross_fusion_med = nn.Linear(3 * emb_dim, emb_dim)
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3* emb_dim, emb_dim)
        )

        self.molecular_network = encoder
        self.f=nn.Linear(64,64)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.aug_MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.aug_MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.fc_selector = nn.Linear(emb_dim, emb_dim)
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)
        self.global_encoder = GNNGraph(**global_para)
        self.ddi_encoder = ddi_encoder
        self.ddi_synergistic_encoder=ddi_synergistic_encoder
        self.substruct_encoder = GNNGraph(**substruct_para)
        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        '''
        adj_tensor = torch.tensor(ddi_adj)
        self.edge_index = adj_tensor.nonzero().t().contiguous()
        # x = np.ones((np.size(ddi_adj, 0),1))
        x = np.random.rand(np.size(ddi_adj, 0), 1)#(131, 1)
        #print('xxxxxx',x.shape)
        self.x = torch.Tensor(x)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        '''

        ddi_adj = np.array(ddi_adj)  # 确保 numpy array
        adj_tensor = torch.tensor(ddi_adj, dtype=torch.long)
        '''
        src, dst = adj.nonzero(as_tuple=True)
        self.edge_index = SparseTensor(row=torch.tensor(src, dtype=torch.long),
                                       col=torch.tensor(dst, dtype=torch.long),
                                       value=torch.ones(len(src), device=adj.device),
                                       sparse_sizes=(adj.shape[0], adj.shape[1])).to(device)
        '''
        edge_index = adj_tensor.nonzero().t().contiguous()
        self.edge_index=torch.tensor(edge_index,dtype=torch.long)
        #print(self.edge_index.dtype)

        self.ddi_embedding = None
        self.mmaapp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(substruct_dim, emb_dim)
        )
        ddi_synergistic_adj = np.array(ddi_synergistic_adj)  # 确保 numpy array
        adj_synergistic_tensor = torch.tensor(ddi_synergistic_adj, dtype=torch.long)
        edge_synergistic_index = adj_synergistic_tensor.nonzero().t().contiguous()
        self.edge_synergistic_index = torch.tensor(edge_synergistic_index, dtype=torch.long)
        self.ddi_synergistic_embedding = None
    def get_inputs(self, dataset, MaxVisit=5):
        diag_list, pro_list, med_hist_list = [], [], []
        med_bce_list, med_ml_list, len_list = [], [], []

        max_visit = min(max(len(p) for p in dataset), MaxVisit)
        ml_diag = max(len(p[j][0]) for p in dataset for j in range(len(p)))
        ml_pro = max(len(p[j][1]) for p in dataset for j in range(len(p)))
        ml_med = max(len(p[j][2]) for p in dataset for j in range(len(p)))

        for patient in dataset:
            # 初始化 PAD
            cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
            cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
            cur_med_hist = torch.full((max_visit, ml_med), self.vocab_size[2])  # 历史药物

            for t, (d_list, p_list, m_list) in enumerate(patient[:max_visit]):
                # ---------------- 当前时刻的输入 ----------------
                cur_diag[t, :len(d_list)] = torch.LongTensor(d_list)
                cur_pro[t, :len(p_list)] = torch.LongTensor(p_list)

                # ---------------- 历史药物信息 ----------------
                if t > 0:  # 从第二次访问开始才有历史药物
                    cur_med_hist[t, :len(patient[t - 1][2])] = torch.LongTensor(patient[t - 1][2])

                # ---------------- 目标：当前时刻的药物 ----------------
                cur_bce = torch.zeros(self.vocab_size[2])
                cur_bce[m_list] = 1
                med_bce_list.append(cur_bce)

                cur_ml = torch.full((self.vocab_size[2],), -1)
                cur_ml[:len(m_list)] = torch.LongTensor(m_list)
                med_ml_list.append(cur_ml)

                # visit 长度
                len_list.append(t + 1)

                # 存拷贝（防止后面覆盖）
                diag_list.append(cur_diag.clone().long())
                pro_list.append(cur_pro.clone().long())
                med_hist_list.append(cur_med_hist.clone().long())

        # ---------------- 打包 tensor ----------------
        diag_tensor = torch.stack(diag_list).to(self.device)
        pro_tensor = torch.stack(pro_list).to(self.device)
        med_tensor = torch.stack(med_hist_list).to(self.device)  # 前一时刻历史药物
        med_tensor_bce_target = torch.stack(med_bce_list).to(self.device)
        med_tensor_ml_target = torch.stack(med_ml_list).to(self.device)
        len_tensor = torch.LongTensor(len_list).to(self.device)

        return diag_tensor, pro_tensor, med_tensor_bce_target, med_tensor_ml_target, len_tensor, med_tensor
    def compute_mahalanobis_distance(self, x, mu, sigma_inv):
        """计算马氏距离"""
        delta = x - mu  # 计算偏差
        dist = torch.sqrt(torch.sum(delta @ sigma_inv * delta, dim=1))  # 按行计算距离
        return dist
    def get_batch(self, data, batchsize=None):
        # diag_tensor, pro_tensor, med_tensor, len_tensor
        # data = self.get_inputs(dataset)
        if batchsize is None:
            yield data
        else:
            N = data[0].shape[0]
            idx = np.arange(N)
            np.random.shuffle(idx)
            i = 0
            while i < N:
                cur_idx = idx[i:i + batchsize]
                res = [cur_data[cur_idx] for cur_data in data]
                yield res
                i += batchsize

    def _get_query(self, diag, pro,med, visit_len,):
        """
               Args:
                   diag: (B, T, Ndiag)
                   pro:  (B, T, Npro)
                   med:  (B, T, Nmed)   # 新增药物输入
                   visit_len: (B,)
               """

        # Step 1: embedding lookup + pooling
        diag_emb_seq = self.dropout(self.embeddings[0](diag).sum(-2))  # (B, T, D)torch.Size([16, 5, 64])
        #print('diag_emb_seq',diag_emb_seq.shape)
        pro_emb_seq = self.dropout(self.embeddings[1](pro).sum(-2))  # (B, T, D)
        med_emb_seq = self.dropout(self.embeddings[2](med).sum(-2))  # (B, T, D)

        # Step 2: 跨模态交互 (在每个时间步把 diag/pro/med 拼接再映射)
        fused_diag_seq, fused_pro_seq, fused_med_seq = [], [], []
        for t in range(diag_emb_seq.shape[1]):
            diag_t = diag_emb_seq[:, t, :]
            pro_t = pro_emb_seq[:, t, :]
            med_t = med_emb_seq[:, t, :]

            # 交叉融合：每个模态都结合另外两个
            fused_diag_seq.append(self.cross_fusion_diag(torch.cat([diag_t, pro_t, med_t], dim=-1)))
            fused_pro_seq.append(self.cross_fusion_pro(torch.cat([pro_t, diag_t, med_t], dim=-1)))
            fused_med_seq.append(self.cross_fusion_med(torch.cat([med_t, diag_t, pro_t], dim=-1)))

        fused_diag_seq = torch.stack(fused_diag_seq, dim=1)  # (B, T, D)
        fused_pro_seq = torch.stack(fused_pro_seq, dim=1)  # (B, T, D)
        fused_med_seq = torch.stack(fused_med_seq, dim=1)  # (B, T, D)

        # Step 3: GRU 编码
        o1, _ = self.encoders[0](fused_diag_seq)  # (B, T, D)
        o2, _ = self.encoders[1](fused_pro_seq)  # (B, T, D)
        o3, _ = self.encoders[2](fused_med_seq)  # (B, T, D)

        # Step 4: 根据 visit_len 取最后一个时间步
        o1 = torch.stack([o1[i, visit_len[i] - 1, :] for i in range(visit_len.shape[0])])
        o2 = torch.stack([o2[i, visit_len[i] - 1, :] for i in range(visit_len.shape[0])])
        o3 = torch.stack([o3[i, visit_len[i] - 1, :] for i in range(visit_len.shape[0])])

        # Step 5: 拼接三个模态，形成最终 query
        patient_representations = torch.cat([o1, o2, o3], dim=-1)  # (B, 3D)
        query = self.query(patient_representations)  # (B, D)

        # 归一化
        norm_of_query = torch.norm(query, 2, 1, keepdim=True)
        normed_query = (norm_of_query / (1 + norm_of_query)) * (query / norm_of_query)

        return query, normed_query


    def forward(self, input, visit_len,mol_data,average_projection,substruct_data,ddi_mask_H):
        """
        Args:
            input(:list) with shape [(B, M, N_x)]. x can be diag, pro, med
            len(:list/LongTensor) with shape (B, 1)
        """
        diag, pro, med,labels = input
        query, normed_query = self._get_query(diag, pro, med,visit_len)  # (Batch, dim)
        substruct_weight = torch.sigmoid(self.substruct_rela(query))

        global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)
        substruct_embeddings = self.sab(self.substruct_encoder(**substruct_data).unsqueeze(0)
        ).squeeze(0)
        #print('ddi_mask_H',ddi_mask_H.shape)
        molecule_embeddings = self.aggregator(
            global_embeddings, substruct_embeddings,
            substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
        )
        molecule_embeddings=self.mmaapp(molecule_embeddings)

        MPNN_emb, rec_loss = self.molecular_network(self.embeddings, self.med2diag, self.med2pro,
                                                    self.ehr_adj)  # (N_medication, dim)
        #print('hi',MPNN_emb.shape)
        MPNN_emb=torch.cat([MPNN_emb,molecule_embeddings], dim=-1)
        #print('m',MPNN_emb.shape)
        if self.ddi_encoder:
            #print('hittttttttttttttttttttt')
            ddi_embedding = self.ddi_encoder(MPNN_emb, self.edge_index)
            self.ddi_e=self.f(ddi_embedding)
            ddi_synergistic_embedding=self.ddi_synergistic_encoder(MPNN_emb, self.edge_synergistic_index)
            self.ddi_synergistic_embedding=self.f(ddi_synergistic_embedding)
            self.ddi_embedding = ddi_synergistic_embedding-ddi_embedding
            # print("self.ddi_embedding", self.ddi_embedding)
            MPNN_emb = self.ddi_embedding
            mu = torch.mean(self.ddi_synergistic_embedding, dim=0)  # 正常样本的均值
            cov_matrix = torch.cov(self.ddi_synergistic_embedding.T)  # 计算协方差矩阵
            # 为了确保协方差矩阵可逆，添加正则化项
            cov_matrix += 1e-6 * torch.eye(cov_matrix.size(0))
            sigma_inv = torch.inverse(cov_matrix)  # 计算协方差矩阵的逆

            # Step 5: 计算马氏距离
            # 测试样本的异常检测分数
            # anomaly_score_test = self.compute_mahalanobis_distance(x112, mu, sigma_inv)
            # print('测试样本的异常检测分数:', anomaly_score_test)

            # 正常样本的异常检测分数（应该接近零）
            # anomaly_score_normal = self.compute_mahalanobis_distance(x111, mu, sigma_inv)
            # print('正常样本的异常检测分数:', anomaly_score_normal)
            score_synergistic = self.compute_mahalanobis_distance(self.ddi_synergistic_embedding, mu, sigma_inv)
            score_ddi = self.compute_mahalanobis_distance(self.ddi_e, mu, sigma_inv)

        #  cosine samilarity
        # MPNN_emb: (M, dim), normed_query (dim,)
        normed_MPNN_emb = MPNN_emb / torch.norm(MPNN_emb, 2, 1, keepdim=True)#torch.Size([131, 64])
        #print(' normed_MPNN_emb', normed_MPNN_emb.shape)
        # normed_MPNN_emb = self.ddi_embedding
        # print("normed_MPNN_emb", normed_MPNN_emb)

        MPNN_match = (torch.mm(normed_query, normed_MPNN_emb.t()))  # (B, N_med)torch.Size([16, 131])
        #print('MPNN_match',MPNN_match.shape)
        MPNN_att = self.MPNN_layernorm(MPNN_match)#torch.Size([16, 131])
        #print('MPNN_att',MPNN_att.shape)
        result = MPNN_att  # result: (M,)torch.Size([16, 131])
        #print('result',result.shape)

        #result,_ =self.CrossAttentionPredictor(query, MPNN_emb)
        #attn_output = torch.bmm(attn_weights, k)  # (B, 1, D)
        # --- Cross-Attention 改进版本 ---
        # normed_query: (B, D)
        # normed_MPNN_emb: (N_med, D)

        # Linear projections
        #Q = self.query_proj(normed_query)  # (B, D)
        #K = self.key_proj(normed_MPNN_emb)  # (N_med, D)
        #V = self.value_proj(normed_MPNN_emb)  # (N_med, D)

        # Attention计算 (B, N_med)
        #attn_scores = torch.matmul(Q, K.T) / (K.size(-1) ** 0.5)
        #attn_weights = F.softmax(attn_scores, dim=-1)  # 每个患者对所有药物的权重

        # 加权求和得到融合特征 (B, D)
        #attn_output = torch.matmul(attn_weights, V)

        # 残差与归一化
        #attn_output = self.MPNN_layernorm(attn_output + normed_query)
        #MPNN_match = (torch.mm(attn_output, normed_MPNN_emb.t()))
        #MPNN_att = self.MPNN_layernorm(MPNN_match)  # torch.Size([16, 131])
        # print('MPNN_att',MPNN_att.shape)
        #result = MPNN_att  # result: (M,)torch.Size([16, 131])
        # 预测层输入
        #result = attn_output  # (B, D)

        if self.args.ddi:
            neg_pred_prob = F.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)  # (B, Nmed, 1)
            tmp_right = neg_pred_prob.unsqueeze(1)  # (B, 1, Nmed)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)  # (N, Nmed, Nmed)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()#标量
        else:
            batch_neg = 0


        return result, batch_neg, 0,score_synergistic,score_ddi

    def save_embedding(self):
        MPNN_emb, rec_loss = self.molecular_network(self.embeddings, self.med2diag, self.med2pro, self.ehr_adj)
        normed_MPNN_emb = MPNN_emb / torch.norm(MPNN_emb, 2, 1, keepdim=True)
        med_emb = MPNN_emb.detach().cpu().numpy()
        normed_med_emb = normed_MPNN_emb.detach().cpu().numpy()

        diag_emb = self.embeddings[0].weight[:-1]  # .detach().cpu().numpy()
        print("save no pad diag_emb: {} -> {}".format(self.embeddings[0].weight.shape, diag_emb.shape))
        normed_diag_emb = diag_emb / torch.norm(diag_emb, 2, 1, keepdim=True)
        diag_emb = diag_emb.detach().cpu().numpy()
        normed_diag_emb = normed_diag_emb.detach().cpu().numpy()

        pro_emb = self.embeddings[1].weight[:-1].detach().cpu().numpy()

        diag_file = os.path.join('saved', self.args.model_name, 'diag.tsv')
        normed_diag_file = os.path.join('saved', self.args.model_name, 'diag_normed.tsv')
        pro_file = os.path.join('saved', self.args.model_name, 'pro.tsv')
        med_file = os.path.join('saved', self.args.model_name, 'med.tsv')
        normed_med_file = os.path.join('saved', self.args.model_name, 'med_normed.tsv')

        if self.ddi_embedding != None:
            normed_ddi_embedding = self.ddi_embedding / torch.norm(self.ddi_embedding, 2, 1, keepdim=True)
            normed_ddi_emb = normed_ddi_embedding.detach().cpu().numpy()
            ddi_emb = self.ddi_embedding.detach().cpu().numpy()
            normed_ddi_file = os.path.join('saved', self.args.model_name, 'ddi_normed.tsv')
            ddi_file = os.path.join('saved', self.args.model_name, 'ddi_emb.tsv')
            np.savetxt(normed_ddi_file, normed_ddi_emb, fmt="%.4f", delimiter='\t')
            np.savetxt(ddi_file, ddi_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(diag_file, diag_emb, fmt="%.4f", delimiter='\t')
        np.savetxt(normed_diag_file, normed_diag_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(pro_file, pro_emb, fmt="%.4f", delimiter='\t')

        np.savetxt(med_file, med_emb, fmt="%.4f", delimiter='\t')
        np.savetxt(normed_med_file, normed_med_emb, fmt="%.4f", delimiter='\t')

        print("saved embedding files")
        return

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
            item.weight.data[:, -1] = 0.
