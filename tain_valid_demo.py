import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.io
import scipy.linalg
import numpy as np
from random import shuffle
from ASD_HNet import SpatialGCN
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn import preprocessing
from gradient_getting import gradient_avg_mask
# from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class myDataset(Dataset):
    def __init__(self, hfc_adj, hfc, fc):
        self.hfc_adj = hfc_adj
        self.hfc = hfc
        self.fc = fc
        # self.label = label
        # self.gradient = gradient
        # self.gra = gra
        # self.kendall_gra = kendall_gra

    def __getitem__(self, index):
        return self.hfc_adj[index], self.hfc[index], self.fc[index]

    def __len__(self):
        return len(self.hfc)


def confusion(g_turth, predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)
    return accuracy, sensitivity, specificty


def training_validating(net, data_iter, valid_iter, lossFunc, epochs=120, i=0):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)  # *len(data_iter)
    train_loss = []

    net.train()
    train_prototype = []
    train_label = []
    print("--------------------start to train!--------------------")
    best_valid_acc, best_valid_sen, best_valid_spe, best_valid_f1 = 0.0, 0.0, 0.0, 0.0
    best_valid_acc_type, best_valid_sen_type, best_valid_spe_type, best_valid_f1_type = 0.0, 0.0, 0.0, 0.0
    for e in tqdm(range(epochs)):
        ASD_prototype = []
        CN_prototype = []
        running_loss = 0
        running_total_loss = 0
        train_acc = 0
        for edge_train, labels, fc_fscore in data_iter:
            label = labels.long()  # .cuda()
            fc_fscore = fc_fscore.float()   # .cuda()
            edge_train = edge_train.to(torch.float32)
            y_hat, node_fea = net(ROI_belong, fc_fscore, edge_train)
            optimizer.zero_grad()
            # label = torch.argmax(labels, dim=1)
            loss = lossFunc(y_hat, label)
            total_loss = loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            predict = torch.argmax(y_hat, dim=1)
            # running_closs+=closs.item()
            running_loss += loss.item()
            running_total_loss += total_loss.item()
            equals = predict == label

            train_acc += torch.mean(equals.type(torch.FloatTensor))
            train_fea_type = node_fea.cpu().detach().numpy()
            for sub_i in range(len(label)):
                if label[sub_i] == 1:
                    ASD_prototype.append(train_fea_type[sub_i])
                else:
                    CN_prototype.append(train_fea_type[sub_i])
                train_prototype.append(train_fea_type[sub_i])
                train_label.append(label[sub_i].tolist())

        train_loss.append(running_loss / len(data_iter))

        print("epoch: {}/{}------ ".format(e + 1, epochs),
              "train_loss: {:.4f}------ ".format(running_loss / len(data_iter)),
              "train_acc: {:.4f}------".format(train_acc / len(data_iter)),
              "total_loss: {:.4f}".format(running_total_loss / len(data_iter)),
              )
        ASD_prototype = np.array(ASD_prototype)
        CN_prototype = np.array(CN_prototype)
        ASD_centers = ASD_prototype.mean(axis=0)
        CN_centers = CN_prototype.mean(axis=0)
        true_label = []
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            net.eval()
            labels_valid_ls = []
            predict_valid_ls = []
            valid_fea_type = []
            for edge_valid, labels_valid, fc_fscore_valid in valid_iter:
                label = labels_valid.long()  # .cuda()
                fc_fscore_valid = fc_fscore_valid.float()
                edge_valid = edge_valid.to(torch.float32)
                y_hat, node_fea_valid = net(ROI_belong, fc_fscore_valid, edge_valid,)  # 0 stands for CN，1 denotes ASD
                predict = torch.argmax(y_hat, dim=1)
                labels_valid_ls.append(label.tolist())
                predict_valid_ls.append(predict.tolist())
                # print(predict, label)
                valid_loss += lossFunc(y_hat, label)
                equals = predict == label
                true_label.append(label.tolist())
                valid_acc += torch.mean(equals.type(torch.FloatTensor))
                fea_valid_type = node_fea_valid.cpu().detach().numpy().reshape(-1)
                valid_fea_type.append(fea_valid_type)
            acc, senci, spec = confusion(labels_valid_ls, predict_valid_ls)
            f1_s = f1_score(labels_valid_ls, predict_valid_ls)
            print("linear--acc: {:.4f}...sen: {:.4f}...spec: {:.4f}...f1: {:.4f}".format(acc, senci, spec, f1_s))
            valid_fea_type = np.array(valid_fea_type)
            label_type = []
            for sub_i in range(len(valid_fea_type)):
                sub_i_dis = []
                asd0_dis = np.sqrt(
                    sum(np.power((ASD_centers - valid_fea_type[sub_i]), 2)))
                # cn0_dis = hyperbolic_distance_np(CN_centers, valid_fea_type[sub_i])
                cn0_dis = np.sqrt(
                    sum(np.power((CN_centers - valid_fea_type[sub_i]), 2)))
                sub_i_dis.append(asd0_dis)
                sub_i_dis.append(cn0_dis)
                sub_i_dis = np.array(sub_i_dis)
                index_min = np.argmin(sub_i_dis)
                if index_min == 0:  # or index_min == 1
                    label_type.append([1])
                else:
                    label_type.append([0])
            equals_type = 0
            for sub_i in range(len(label_type)):
                if label_type[sub_i] == true_label[sub_i]:
                    equals_type = equals_type + 1
            # print("prototype acc: {:.4f}".format(equals_type / len(label_type)))
            acc_type, senci_type, spec_type = confusion(true_label, label_type)
            f1_s_type = f1_score(true_label, label_type)
            print("prototype---acc: {:.4f}...sen: {:.4f}...spec: {:.4f}...f1: {:.4f}".format(acc_type, senci_type,
                                                                                         spec_type, f1_s_type))
            if acc >= best_valid_acc:
                best_valid_acc = acc
                best_valid_sen = senci
                best_valid_spe = spec
                best_valid_f1 = f1_s
                dir = './model/ASD-HNet_linear_I_fold' + str(i) + '.pth'
                torch.save(net.state_dict(), dir)
            if acc_type >= best_valid_acc_type:
                best_valid_acc_type = acc_type
                best_valid_sen_type = senci_type
                best_valid_spe_type = spec_type
                best_valid_f1_type = f1_s_type
                dir = './model/ASD-HNet_type_I_fold' + str(i) + '.pth'
                torch.save(net.state_dict(), dir)
            # print("hyplinear---acc: {:.4f}...sen: {:.4f}...spec: {:.4f}...f1: {:.4f}".format(acc, senci, spec, f1_s))
        if e==epochs-1:
            vote_F1_mean.append(f1_s_type)
            vote_ACC_mean.append(acc_type)
            vote_SEN_mean.append(senci_type)
            vote_SPE_mean.append(spec_type)
    total_valid_sen.append(best_valid_sen)
    total_valid_f1.append(best_valid_f1)
    total_valid_spe.append(best_valid_spe)
    total_valid_acc.append(best_valid_acc)

    total_validtype_senci.append(best_valid_sen_type)
    total_validtype_spec.append(best_valid_spe_type)
    total_validtype_f1.append(best_valid_f1_type)
    total_validtype_acc.append(best_valid_acc_type)
    return train_prototype, train_label, valid_fea_type, true_label, np.vstack((CN_centers, ASD_centers))


def kfold_data(hfc_adj, hfc, fc, k):
    total_train_loader = []
    total_valid_loader = []
    fold_size = hfc_adj.shape[0] // k
    for i in range(k):
        hfc_adj_train, hfc_train, fc_train = None, None, None
        hfc_adj_valid, hfc_valid, fc_valid = None, None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)

            hfc_adj_part, hfc_part, fc_part = hfc_adj[idx, :], hfc[idx], fc[idx,:]
            if j == i:  # i-th fold as testing data
                hfc_adj_valid, hfc_valid, fc_valid = hfc_adj_part, hfc_part, fc_part
            elif hfc_adj is None:
                hfc_adj_train, hfc_train, fc_train = hfc_adj_part, hfc_part, fc_part
            else:  # remaining fold as training data
                hfc_adj_train = torch.cat((hfc_adj_train, hfc_adj_part), dim=0)
                hfc_train = torch.cat((hfc_train, hfc_part), dim=0)
                fc_train = torch.cat((fc_train, fc_part), dim=0)

        train_data = myDataset(hfc_adj_train, hfc_train, fc_train)
        valid_data = myDataset(hfc_adj_valid, hfc_valid, fc_valid)

        train_iter = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
        valid_iter = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=4)
        total_train_loader.append(train_iter)
        total_valid_loader.append(valid_iter)

    return total_train_loader, total_valid_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def shuffle_data(hfc, asy, fc_aal, fc_cc200):
    shuffle_flag = True
    if shuffle_flag:
        q = list(zip(hfc, asy, fc_aal, fc_cc200))
        shuffle(q)
        hfc, asy, fc_aal, fc_cc200 = zip(*q)
    return np.array(hfc), np.array(asy), np.array(fc_aal), np.array(fc_cc200)


def data_norm(data):
    max_value = data.max()
    min_value = data.min()
    ranges = max_value - min_value
    data_norm = (data - min_value) / ranges

    return data_norm



from sklearn.model_selection import KFold
def split_indices(indices, n_splits, fold):
    """
    根据 fold 对指定 indices 进行 KFold 划分
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(indices))
    train_idx, test_idx = splits[fold]
    return indices[train_idx], indices[test_idx]


from brainspace.gradient import GradientMaps
def gradient_getting(Fc_all,  node_num=116):
    gradient_total = []
    for i in range(len(Fc_all)):
        gm1 = GradientMaps(n_components=1, approach='dm', kernel='cosine')
        gm1.fit(Fc_all[i])
        gradient_total.append(gm1.gradients_)
    gradient_total = np.array(gradient_total)
    return gradient_total

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    global total_valid_acc
    total_valid_acc = []
    global total_validtype_acc, total_validtype_senci, total_validtype_spec, total_validtype_f1
    total_validtype_acc = []
    total_validtype_senci = []
    total_validtype_spec = []
    total_validtype_f1 = []
    global total_valid_sen
    total_valid_sen = []
    global total_valid_spe
    total_valid_spe = []
    global total_valid_f1
    total_valid_f1 = []
    global cluster_gra
    global SVM_ACC_mean, SVM_SEN_mean, SVM_SPE_mean, SVM_F1_mean
    global vote_ACC_mean, vote_SEN_mean, vote_SPE_mean, vote_F1_mean
    SVM_ACC_mean, SVM_SEN_mean, SVM_SPE_mean, SVM_F1_mean = [], [], [], []
    vote_ACC_mean, vote_SEN_mean, vote_SPE_mean, vote_F1_mean = [], [], [], []
    proto_ACC_mean, proto_SEN_mean, proto_SPE_mean, proto_F1_mean = [], [], [], []
    global ROI_belong
    setup_seed(3407)

    FC_fscore = np.load(r"FC_fscore.npy")
    FC_aal = np.load(r"FC_aal.npy")
    labels = np.load(r"labels.npy")
    labels = np.argmax(labels, axis=1)
    gra_edge_corr = np.load(r"Fscore_edge.npy")

    gra_edge_corr = torch.tensor(gra_edge_corr)
    k_cluster = 7
    gra_edge_mask = gradient_avg_mask(FC_aal, k_cluster, mask=True)
    gra_edge = np.zeros(FC_aal.shape)
    for num_i in range(len(FC_aal)):
        gra_edge[num_i] = gra_edge_mask * FC_aal[num_i]
    cluster_gra = gradient_avg_mask(FC_aal, k_cluster, mask=False)
    gra_edge = torch.tensor(gra_edge)
    ROI_belong = {}
    node_num4comm = []
    for community_i in range(len(cluster_gra)):
        ROI_belong[community_i] = torch.tensor(np.where(cluster_gra[community_i] == 1)[0])
        node_num4comm.append(len(torch.tensor(np.where(cluster_gra[community_i] == 1)[0])))
    node_num4comm = torch.tensor(node_num4comm)

    cluster_gra = torch.tensor(cluster_gra).to(torch.float32)

    FC_aal = torch.tensor(FC_aal)

    FC_fscore = torch.tensor(FC_fscore)
    labels = torch.tensor(labels)
    train_iter, valid_iter = kfold_data(gra_edge_corr, labels, FC_fscore, 10)  # gra_edge_corr

    for i in range(10):
        print("fold: {}".format(i))
        lossFunc = nn.CrossEntropyLoss()
        net = SpatialGCN(node_num_comm=node_num4comm, k=k_cluster)    # .cuda()
        train_prototype, train_label, test_fea_type, test_label, proto_centers = training_validating(net, train_iter[i],
                                                                                                     valid_iter[i],
                                                                                                     lossFunc, 95, i)
        lossFunc1 = nn.CrossEntropyLoss()

        del net
        del lossFunc
    print(total_valid_acc)
    print(total_validtype_acc)
    total_valid_acc = torch.tensor(np.array(total_valid_acc))
    print(torch.mean(total_valid_acc))
    total_valid_acc = torch.tensor(np.array(total_valid_acc))
    total_validtype_acc = torch.tensor(np.array(total_validtype_acc))
    total_validtype_f1 = torch.tensor(np.array(total_validtype_f1))
    total_validtype_senci = torch.tensor(np.array(total_validtype_senci))
    total_validtype_spec = torch.tensor(np.array(total_validtype_spec))
    total_valid_sen = torch.tensor(np.array(total_valid_sen))
    total_valid_spe = torch.tensor(np.array(total_valid_spe))
    total_valid_f1 = torch.tensor(np.array(total_valid_f1))

    total_vote_f1 = torch.tensor(np.array(vote_F1_mean))
    total_vote_sen = torch.tensor(np.array(vote_SEN_mean))
    total_vote_spec = torch.tensor(np.array(vote_SPE_mean))
    total_vote_acc = torch.tensor(np.array(vote_ACC_mean))

    proto_ACC_mean = torch.tensor(np.array(proto_ACC_mean))
    proto_SEN_mean = torch.tensor(np.array(proto_SEN_mean))
    proto_SPE_mean = torch.tensor(np.array(proto_SPE_mean))
    proto_F1_mean = torch.tensor(np.array(proto_F1_mean))
    print(
        "valid_acc: {:.4f}...valid_sen: {:.4f}...valid_spec: {:.4f}...valid_f1: {:.4f}".format(
            torch.mean(total_valid_acc),
            torch.mean(total_valid_sen),
            torch.mean(total_valid_spe),
            torch.mean(total_valid_f1)))

    print(
        "type_valid_acc: {:.4f}...type_valid_sen: {:.4f}...type_valid_spec: {:.4f}...type_valid_f1: {:.4f}".format(
            torch.mean(total_validtype_acc),
            torch.mean(total_validtype_senci),
            torch.mean(total_validtype_spec),
            torch.mean(total_validtype_f1)))

    print(
        "last-----type_valid_acc: {:.4f}...type_valid_sen: {:.4f}...type_valid_spec: {:.4f}...type_valid_f1: {:.4f}".format(
            torch.mean(total_vote_acc),
            torch.mean(total_vote_sen),
            torch.mean(total_vote_spec),
            torch.mean(total_vote_f1)))


