import torch
import torch.nn as nn
import math
import torch.nn.init as init
import torch.nn.functional as F


class Atten(nn.Module):
    def __init__(self, batch_size=32, N=116, dropout=0.3):
        super(Atten, self).__init__()
        self.node = N
        self.dropout = dropout
        self.batch_size = batch_size
        self.cluster_atten = nn.Parameter(torch.ones(batch_size, self.node))
        self.atten = nn.Sequential(
            nn.Linear(N, N),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.atten_gra = nn.Sequential(
            nn.Linear(116, 116),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(116, 116),
            nn.ReLU()
        )
        self.batn = nn.BatchNorm1d(N)

    def forward(self, gcn_out):   # gcn_out: bat 116 32
        out1 = torch.amax(gcn_out, 2)  # batch * 116
        out1 = torch.ones((out1.shape[0], self.node)).to(torch.float32)
        atten_out = self.atten(out1)
        # atten_gra = self.atten_gra(gra).reshape(-1, self.node, 1)
        atten_out = atten_out.reshape(-1, self.node, 1)  # get bat * 116 * 1
        out2 = self.batn(gcn_out * atten_out + gcn_out)  # bat * 116 * 32
        return out2


class SpatialGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SpatialGCNLayer, self).__init__()
        self.w1 = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.reset_para()

    # def reset_para(self):
    #     nn.init.xavier_normal_(self.w1)
    #     # stdv = 1. / math.sqrt(self.w1.size(1))
    #     if self.bias is not None:
    #         nn.init.constant_(self.bias, 0)
    #         # nn.init.normal_(self.bias, 0, 1)

    def reset_para(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        init.kaiming_uniform_(self.w1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_matrix):
        # x: Node features (N x in_features)
        # adj_matrix: Adjacency matrix (N x N)
        batch_size, num_nodes, in_features = x.size()

        # Flatten batch and node dimensions
        x = x.reshape(-1, in_features)
        # adj_matrix = adj_matrix.view(-1, num_nodes)

        support = torch.mm(x, self.w1)  # AxW
        support = support.view(batch_size, num_nodes, -1)
        output = torch.bmm(adj_matrix, support)  # A^T(AxW)

        # Reshape back to original shape
        # output = output.view(batch_size, num_nodes, -1)
        # print("-----------", self.bias)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class community_local_cluster(nn.Module):
    def __init__(self, node_num, node_length=64, dropout=0.3):
        super(community_local_cluster, self).__init__()
        self.cluster = nn.Sequential(
            nn.Conv1d(node_num, 1, kernel_size=15, padding=15 // 2, stride=1),  #
            nn.LayerNorm(node_length),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_cluster = self.cluster(x)
        return x_cluster


class SpatialGCN(nn.Module):
    def __init__(self, node_num_comm, k=7, num_features=116, hidden_size=64, out_size=32, dropout=0.3, bias=True):
        super(SpatialGCN, self).__init__()
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.layer1 = SpatialGCNLayer(num_features, hidden_size, bias)
        self.activate = nn.Tanh()
        self.layernorm1 = nn.LayerNorm(64)
        self.layernorm2 = nn.LayerNorm(32)
        self.atten_cluster = Atten(N=k)
        self.atten_node = Atten(N=116)
        # community represent getting
        self.comm_layers = nn.ModuleList([])
        for comm_i in range(k):
            self.comm_layers.append(community_local_cluster(node_num=node_num_comm[comm_i]))
        self.conv1 = nn.Conv2d(1, 16, (self.k, 64), stride=1)
        self.linear1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.brain_linear = nn.Sequential(
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def pcc_cal(self, X_total):
        corrcoef_total = torch.tensor([])
        for x in X_total:
            x_reducemean = x - torch.mean(x, dim=1, keepdim=True)
            numerator = torch.matmul(x_reducemean, x_reducemean.T)
            no = torch.norm(x_reducemean, p=2, dim=1).unsqueeze(1)
            denominator = torch.matmul(no, no.T)
            # corrcoef = (numerator / denominator).fill_diagonal_(1.0).unsqueeze(0)
            corrcoef = (numerator / (denominator + 1e-8)).clone().fill_diagonal_(1.0).unsqueeze(0)
            corrcoef_total = torch.cat((corrcoef_total, corrcoef), dim=0)
        return corrcoef_total

    def forward(self, ROIs_belong, x, gra_edge):
        # x: Node features  (fc)
        # gra_edge: 116 116 (adj)
        # ROIs_belong: cluster_num 116
        x = self.activate(self.dropout(self.layer1(x, gra_edge)))  # bat 116 64
        x = self.atten_node(x)
        # bat 116 64
        comm_fea = torch.zeros((32, 1, 64))
        for comm_i in range(self.k):
            if comm_i==0:
                comm_fea=self.comm_layers[comm_i](x[:, ROIs_belong[comm_i]])
            else:
                comm_temp = self.comm_layers[comm_i](x[:, ROIs_belong[comm_i]])
                comm_fea = torch.cat((comm_fea, comm_temp), dim=1)
        brain_fea = torch.unsqueeze(comm_fea, dim=1)   # bat 1 k 64
        brain_fea_1 = self.conv1(brain_fea).reshape(-1, 16)
        # brain_fea_1 = torch.amax(brain_fea_1, dim=2)
        brain_fea = self.linear1(brain_fea_1)
        brain_repre_out = self.brain_linear(brain_fea)
        return brain_repre_out, brain_fea

