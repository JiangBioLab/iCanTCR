# coding: utf-8
import os
import sys

import pandas
import torch
from torch import nn
import argparse
import my_utils
import numpy as np
import math
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time


torch.manual_seed(102)  # reproducible
MIN_AA_LEN = 11
MAX_AA_LEN = 19
K = 1
PC = 18

CFG = {  # CNN config
    'cfg00': [16, 'M', 16, 'M'],
    'cfg01': [16, 'M', 32, 'M'],
    'cfg02': [8, 'M', 8, 'M'],
    'cfg03': [64, 'M'],
    'cfg04': [8, 'M', 16, 'M'],
}
CANCER_3_DICT = np.load('./feature/3mer_abundance.npy', allow_pickle=True)
CANCER_4_DICT = np.load('./feature/4mer_abundance.npy', allow_pickle=True)
CANCER_5_DICT = np.load('./feature/5mer_abundance.npy', allow_pickle=True)

def load_aa_pc():
    # 将PCA的内容读取为一个dictionary(key=amino acid， value=list of pc values)
    aa_pc_map = {}
    n = 1
    file_path = './feature/AAidx_PCA.txt'
    with open(file_path, 'r') as f:
        for line in f:
            if n == 1:
                n = 2
                continue
            items = line.strip().split('\t')
            # items = line.strip().split(',')
            aa_name = items[0]
            aa_pca = [float(x) for x in items[1:]]
            aa_pc_map[aa_name] = aa_pca[:18]  # 前18个包含信息>99%, 前14个包含信息> 95%
    f.close()
    return aa_pc_map


# 将序列转化为AA_index的PCA 特征
def get_aaidx_feature(seq, k=1):
    aa_pc_map = load_aa_pc()
    aaidx_features = my_utils.get_aaseq_kmer(seq, aa_pc_map, k=k)
    return aaidx_features


# 将序列转化为k_mer的丰度特征
def get_k_feature(seq, k):
    # CANCER_3_DICT = np.load('./feature/3mer_abundance.npy', allow_pickle=True)
    # CANCER_4_DICT = np.load('./feature/4mer_abundance.npy', allow_pickle=True)
    # CANCER_5_DICT = np.load('./feature/5mer_abundance.npy', allow_pickle=True)
    cancer_l = ['BRCA', 'NSCLC', 'ESCA', 'GBM', 'LIHC', 'SARC', 'MELA', 'PRC', 'BLCA', 'HNSCC', 'MCC', 'healthy']
    k_feature, k_feature2 = [], []
    if k == 3:
        cancer_k_dict = CANCER_3_DICT.tolist()
    elif k == 4:
        cancer_k_dict = CANCER_4_DICT.tolist()
    elif k == 5:
        cancer_k_dict = CANCER_5_DICT.tolist()
    for i in range(len(seq) - k + 1):
        a_kmer = seq[i:i + k]
        one_feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        one_feature2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        n = 0
        for cancer in cancer_l:
            k_mer_d1 = cancer_k_dict[cancer][0]
            k_mer_d2 = cancer_k_dict[cancer][1]
            if a_kmer in k_mer_d1:
                one_feature[n] = k_mer_d1[a_kmer]
            if a_kmer in k_mer_d2:
                one_feature2[n] = k_mer_d2[a_kmer]
            n += 1
        k_feature.extend(one_feature)
        k_feature2.extend(one_feature2)
    assert len(k_feature) == len(k_feature2)
    joint_features = [k_feature, k_feature2]
    return joint_features


def split_file(x, y, test_size=0.2, is_random=False):
    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(x) is not np.ndarray:
        y = np.array(y)
    random_state = None
    if is_random:
        random_state = 111
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


class DataLoad(Dataset):
    def __init__(self, data, data_label, is_gpu=False, transform=None):
        self.b_feature_matrixs = []
        self.k_features = []
        self.cloneFracs = []
        self.mutil_labels = []
        self.binary_labels = []
        self.max_seq_len = MAX_AA_LEN - K + 1 - 5
        self.max_k3_len = MAX_AA_LEN - 3 + 1 - 5
        self.max_k4_len = MAX_AA_LEN - 4 + 1 - 5
        self.max_k5_len = MAX_AA_LEN - 5 + 1 - 5

        for i in range(len(data)):
            b_feature_matrix = data[i][0]
            b_feature_matrix = np.array(b_feature_matrix, dtype=object)
            if (b_feature_matrix.shape[0]) < self.max_seq_len:  # 为序列长度小于MAX_AA_LEN 的序列补零
                b_feature_matrix = np.pad(b_feature_matrix,
                                          ((0, self.max_seq_len - (b_feature_matrix.shape[0])), (0, 0)),
                                          'constant', constant_values=0)
            b_feature_matrix = np.array(b_feature_matrix, dtype=float)
            b_feature_matrix = torch.from_numpy(b_feature_matrix).type(torch.FloatTensor)

            k_feature_list = data[i][1]

            k3_feature = np.pad(np.array(k_feature_list[0][0]), (0, 12 * self.max_k3_len - len(k_feature_list[0][0])),
                               'constant')
            k3_feature2 = np.pad(np.array(k_feature_list[0][1]), (0, 12 * self.max_k3_len - len(k_feature_list[0][1])),
                                'constant')
            k4_feature = np.pad(np.array(k_feature_list[1][0]), (0, 12 * self.max_k4_len - len(k_feature_list[1][0])),
                               'constant')
            k4_feature2 = np.pad(np.array(k_feature_list[1][1]), (0, 12 * self.max_k4_len - len(k_feature_list[1][1])),
                                'constant')
            k5_feature = np.pad(np.array(k_feature_list[2][0]), (0, 12 * self.max_k5_len - len(k_feature_list[2][0])),
                               'constant')
            k5_feature2 = np.pad(np.array(k_feature_list[2][1]), (0, 12 * self.max_k5_len - len(k_feature_list[2][1])),
                                'constant')

            k_features = np.concatenate((k3_feature, k3_feature2, k4_feature, k4_feature2, k5_feature, k5_feature2))
            k_features = np.array(k_features, dtype=float)
            k_features = torch.from_numpy(k_features).type(torch.FloatTensor)

            if is_gpu:
                self.b_feature_matrixs.append(b_feature_matrix.cuda())
                self.k_features.append(k_features.cuda())
            else:
                self.b_feature_matrixs.append(b_feature_matrix)
                self.k_features.append(k_features)

        for i in range(len(data_label)):
            binary_label = data_label[i]
            self.binary_labels.append(binary_label)

        if is_gpu:
            self.binary_labels = torch.LongTensor(self.binary_labels).cuda()
        else:
            self.binary_labels = torch.LongTensor(self.binary_labels)

    def __getitem__(self, idx):
        idx = idx % len(self)
        b_feature_matrix = self.b_feature_matrixs[idx]
        k_features = self.k_features[idx]
        binary_label = self.binary_labels[idx]
        return b_feature_matrix, k_features, binary_label

    def __len__(self):
        return len(self.binary_labels)


class Seq_model(nn.Module):
    def __init__(self):
        super(Seq_model, self).__init__()
        self.cnn2d_b_3 = self.Conv2d(CFG['cfg04'], 3)
        self.cnn2d_b_4 = self.Conv2d(CFG['cfg04'], 4)
        self.cnn2d_b_5 = self.Conv2d(CFG['cfg04'], 5)
        self.rnnlayer_b = nn.LSTM(
            input_size=18,  # 17 for one-hot, 13 for pca
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.rnn_dense = nn.Linear(64, 32)
        self.FClayer_1 = nn.Linear(1664, 512)
        self.FClayer_2 = nn.Linear(512, 64)

        self.outlayer = nn.Linear(64, 2)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x_b):
        cnn_x_b = torch.unsqueeze(x_b, 1)
        cnn_b_3 = self.cnn2d_b_3(cnn_x_b)
        cnn_b_3 = cnn_b_3.view(cnn_b_3.size(0), -1)  # 展开多维的卷积图
        cnn_b_4 = self.cnn2d_b_4(cnn_x_b)
        cnn_b_4 = cnn_b_4.view(cnn_b_4.size(0), -1)  # 展开多维的卷积图
        cnn_b_5 = self.cnn2d_b_5(cnn_x_b)
        cnn_b_5 = cnn_b_5.view(cnn_b_5.size(0), -1)  # 展开多维的卷积图

        rnn_out_b, (h_n, h_c) = self.rnnlayer_b(x_b, None)
        out_b1 = self.dropout1(self.rnn_dense(rnn_out_b[:, -1, :]))   #

        cnn_combine = torch.cat((cnn_b_3, cnn_b_4, cnn_b_5, out_b1), 1)

        out1 = self.FClayer_1(cnn_combine)
        out1 = nn.functional.relu(out1)
        out2 = self.dropout2(self.FClayer_2(out1))
        out2 = nn.functional.relu(out2)
        out = self.outlayer(out2)
        out = nn.functional.softmax(out, dim=1)
        return out, out2

    def Conv1d(self, cfg):
        layers = []
        in_channels = 18
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=2, stride=1, padding=2),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def Conv2d(self, cfg, kernel_size):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class K_model(nn.Module):
    def __init__(self):
        super(K_model, self).__init__()
        self.input_layer = nn.Linear(792, 1024)
        self.dense_layer1 = nn.Linear(1024, 256)
        self.dense_layer2 = nn.Linear(256, 32)
        self.binary_layer = nn.Linear(32, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, _input):
        out1 = self.input_layer(_input)
        out1 = nn.functional.relu(out1)
        out2 = self.dropout(self.dense_layer1(out1))
        out2 = nn.functional.relu(out2)
        out3 = self.dropout(self.dense_layer2(out2))
        out3 = nn.functional.relu(out3)
        out = self.binary_layer(out3)
        out = nn.functional.softmax(out, dim=1)
        return out, out3


class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.input_layer = nn.Linear(96, 32)
        self.output_layer = nn.Linear(32, 2)
        self.dropout = nn.Dropout()

    def forward(self, s_input, k_input):
        input_combine = torch.cat((s_input, k_input), 1)
        out1 = self.input_layer(input_combine)
        out1 = nn.functional.relu(out1)
        out = self.output_layer(out1)
        out = nn.functional.softmax(out, dim=1)
        return out


class B_Seq_model(nn.Module):
    def __init__(self):
        super(B_Seq_model, self).__init__()

        self.cnn2d_b_3 = self.Conv2d(CFG['cfg04'], 3)
        self.cnn2d_b_4 = self.Conv2d(CFG['cfg04'], 4)
        self.cnn2d_b_5 = self.Conv2d(CFG['cfg04'], 5)

        self.rnnlayer_b = nn.LSTM(
            input_size=18,  # 17 for one-hot, 13 for pca
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.rnn_dense = nn.Linear(64, 32)

        self.FClayer_1 = nn.Linear(1664, 512)
        self.FClayer_2 = nn.Linear(512, 64)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x_b):
        cnn_x_b = torch.unsqueeze(x_b, 1)
        cnn_b_3 = self.cnn2d_b_3(cnn_x_b)
        cnn_b_3 = cnn_b_3.view(cnn_b_3.size(0), -1)  # 展开多维的卷积图
        cnn_b_4 = self.cnn2d_b_4(cnn_x_b)
        cnn_b_4 = cnn_b_4.view(cnn_b_4.size(0), -1)  # 展开多维的卷积图
        cnn_b_5 = self.cnn2d_b_5(cnn_x_b)
        cnn_b_5 = cnn_b_5.view(cnn_b_5.size(0), -1)  # 展开多维的卷积图

        rnn_out_b, (h_n, h_c) = self.rnnlayer_b(x_b, None)
        out_b1 = self.dropout1(self.rnn_dense(rnn_out_b[:, -1, :]))   # 选取最后一个时间点的rnn_out的输出

        cnn_combine = torch.cat((cnn_b_3, cnn_b_4, cnn_b_5, out_b1), 1)

        out1 = self.FClayer_1(cnn_combine)
        out1 = nn.functional.relu(out1)
        out2 = self.dropout2(self.FClayer_2(out1))
        out2 = nn.functional.relu(out2)
        return out2

    def Conv2d(self, cfg, kernel_size):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class B_K_model(nn.Module):
    def __init__(self):
        super(B_K_model, self).__init__()
        self.input_layer = nn.Linear(792, 1024)
        self.dense_layer1 = nn.Linear(1024, 256)
        self.dense_layer2 = nn.Linear(256, 32)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, _input):
        out1 = self.input_layer(_input)
        out1 = nn.functional.relu(out1)
        out2 = self.dropout(self.dense_layer1(out1))
        out2 = nn.functional.relu(out2)
        out3 = self.dropout(self.dense_layer2(out2))
        out3 = nn.functional.relu(out3)
        return out3


class B_EnsembleModel(nn.Module):
    def __init__(self):
        super(B_EnsembleModel, self).__init__()
        self.input_layer = nn.Linear(96, 32)
        self.output_layer = nn.Linear(32, 2)
        self.dropout = nn.Dropout()

    def forward(self, s_input, k_input):
        input_combine = torch.cat((s_input, k_input), 1)
        out1 = self.input_layer(input_combine)
        out1 = nn.functional.relu(out1)
        out = self.output_layer(out1)
        out = nn.functional.softmax(out, dim=1)
        return out


def covert_para():
    lr, batch_size = 0.001, 64
    old_seq_model = torch.load('./saved_pkl/temp_pkl/bs_{}_{}.pkl'.format(lr, batch_size), map_location='cpu')
    old_k_model = torch.load('./saved_pkl/temp_pkl/bk_{}_{}.pkl'.format(lr, batch_size), map_location='cpu')
    old_e_model = torch.load('./saved_pkl/temp_pkl/be_{}_{}.pkl'.format(lr, batch_size), map_location='cpu')

    new_seq_model = B_Seq_model()
    new_k_model = B_K_model()
    new_e_model = B_EnsembleModel()

    new_seq_model.load_state_dict(old_seq_model.state_dict(), strict=False)
    new_k_model.load_state_dict(old_k_model.state_dict(), strict=False)
    new_e_model.load_state_dict(old_e_model.state_dict(), strict=False)

    torch.save(new_seq_model.state_dict(), './saved_pkl/bs_{}_{}_dict.pkl'.format(lr, batch_size))
    torch.save(new_k_model.state_dict(), './saved_pkl/bk_{}_{}_dict.pkl'.format(lr, batch_size))
    torch.save(new_e_model.state_dict(), './saved_pkl/be_{}_{}_dict.pkl'.format(lr, batch_size))


def bina_training(is_gpu=False):

    if not os.path.exists("./saved_pkl/temp_pkl"):
        os.system('mkdir ' + "./saved_pkl")
        os.system('mkdir ' + "./saved_pkl/temp_pkl/")

    def read_data(path):
        pd_data = pd.read_csv(path)
        aa_seqs = pd_data['AA_seq'].values.tolist()
        labels = pd_data['Label'].values.tolist()

        dataset = []
        for aa_seq in aa_seqs:
            # 防止乱码和错误
            if aa_seq == '':
                continue
            if aa_seq == 'non' or aa_seq == 'NA' or aa_seq == 'Couldn\'t find FGXG':
                continue
            if len(aa_seq) < MIN_AA_LEN or len(aa_seq) > MAX_AA_LEN:
                continue
            b_chain = list(aa_seq)
            used_chain_list = b_chain[4:(len(b_chain) - 1)]
            used_chain = aa_seq[4:(len(b_chain) - 1)]
            aaidx_features = get_aaidx_feature(used_chain_list, k=K)  # 将序列转化为AA_index的PCA 特征
            if len(aaidx_features) == 0:
                continue
            k3_rank_features = get_k_feature(used_chain, k=3)  # 将序列转化为k_mer的丰度特征
            k4_rank_features = get_k_feature(used_chain, k=4)  # 将序列转化为k_mer的丰度特征
            k5_rank_features = get_k_feature(used_chain, k=5)  # 将序列转化为k_mer的丰度特征
            k_rank_features = [k3_rank_features, k4_rank_features, k5_rank_features]
            dataset.append([aaidx_features, k_rank_features])
        return dataset, labels

    def map2features(x):
        x_b, x_k = [], []
        for i in range(len(x)):
            each_x_b = np.array(x[i][0], dtype=float)
            each_k = x[i][1]
            max_seq_shape = MAX_AA_LEN - K + 1 - 5
            max_3_shape = MAX_AA_LEN - 3 + 1 - 5
            max_4_shape = MAX_AA_LEN - 4 + 1 - 5
            max_5_shape = MAX_AA_LEN - 5 + 1 - 5
            if (each_x_b.shape[0]) < max_seq_shape:
                each_x_b = np.pad(each_x_b, ((0, max_seq_shape - (each_x_b.shape[0])), (0, 0)), 'constant',
                                  constant_values=(0, 0))
            assert each_x_b.shape[0] == max_seq_shape
            k3_feature1 = np.pad(np.array(each_k[0][0]), (0, 12 * max_3_shape - len(each_k[0][0])), 'constant')
            k3_feature2 = np.pad(np.array(each_k[0][1]), (0, 12 * max_3_shape - len(each_k[0][1])), 'constant')
            k4_feature1 = np.pad(np.array(each_k[1][0]), (0, 12 * max_4_shape - len(each_k[1][0])), 'constant')
            k4_feature2 = np.pad(np.array(each_k[1][1]), (0, 12 * max_4_shape - len(each_k[1][1])), 'constant')
            k5_feature1 = np.pad(np.array(each_k[2][0]), (0, 12 * max_5_shape - len(each_k[2][0])), 'constant')
            k5_feature2 = np.pad(np.array(each_k[2][1]), (0, 12 * max_5_shape - len(each_k[2][1])), 'constant')
            k_features = np.concatenate((k3_feature1, k3_feature2, k4_feature1, k4_feature2, k5_feature1, k5_feature2))
            k_features = np.array(k_features, dtype=float)
            x_b.append(each_x_b.tolist())
            x_k.append(k_features.tolist())
        if is_gpu:
            x_b = torch.FloatTensor(x_b).cuda()
            x_k = torch.FloatTensor(x_k).cuda()
        else:
            x_b = torch.FloatTensor(x_b)
            x_k = torch.FloatTensor(x_k)
        return x_b, x_k

    if is_gpu:
        seq_model = Seq_model().cuda()
        k_model = K_model().cuda()
        e_model = EnsembleModel().cuda()
    else:
        seq_model = Seq_model()
        k_model = K_model()
        e_model = EnsembleModel()

    lr = 0.001
    batch_size = 64
    epoch = 70
    s_optimizer = torch.optim.Adam(seq_model.parameters(), lr=lr, weight_decay=0.00001)
    k_optimizer = torch.optim.Adam(k_model.parameters(), lr=lr, weight_decay=0.00001)
    e_optimizer = torch.optim.Adam(e_model.parameters(), lr=lr, weight_decay=0.00001)
    binary_loss_func = nn.CrossEntropyLoss()

    training_x, training_y = read_data('./data/bina_training/training_seq_bina.csv')
    x_tra, x_val, y_tra, y_val = split_file(training_x, training_y, test_size=0.1, is_random=True)
    val_x_b, val_x_k = map2features(x_val)
    val_y_binary = y_val
    train_dataset = DataLoad(x_tra, y_tra, is_gpu)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    seq_model.train()
    k_model.train()
    e_model.train()

    # validation
    def validation(test_model, test_input, is_ensemble=False):
        if is_ensemble:
            test_out_bina = test_model(test_input[0], test_input[1])
        else:
            test_out_bina, elseout = test_model(test_input)
        if is_gpu:
            test_out_bina = test_out_bina.cpu()
        test_model.eval()
        y_tensor_bina = torch.LongTensor(val_y_binary)
        test_loss_bina = binary_loss_func(test_out_bina, y_tensor_bina)
        bina_pred_y = torch.max(test_out_bina, 1)[1].data.numpy()
        bina_acc = float((bina_pred_y == np.array(val_y_binary)).astype(int).sum()) / float(len(val_y_binary))
        post_score = torch.index_select(test_out_bina, 1, torch.tensor([1]))
        post_score = post_score.data.numpy()
        fpr, tpr, thresholds = metrics.roc_curve(np.array(val_y_binary), post_score, pos_label=1)
        bina_auc = metrics.auc(fpr, tpr)
        return bina_pred_y, test_loss_bina, bina_acc, bina_auc

    def seq_run():
        best_auc = 0
        for epo in range(epoch):
            seq_model.train()
            for step, (x_b, x_k, binary_y) in enumerate(train_loader):
                bina_out, semi_outs = seq_model(x_b)
                bina_loss = binary_loss_func(bina_out, binary_y)
                s_optimizer.zero_grad()
                bina_loss.backward()
                s_optimizer.step()

            bina_pred_y, test_loss_bina, bina_acc, bina_auc = validation(seq_model, val_x_b)
            if bina_auc > best_auc:
                best_auc = bina_auc
                torch.save(seq_model, './saved_pkl/temp_pkl/bs_{}_{}.pkl'.format(lr, batch_size))
            print('Seq_Epoch: ', epo, '| b_loss : %.2f' % test_loss_bina)

    def k_run():
        best_auc = 0
        for epo in range(epoch):
            k_model.train()
            for step, (x_b, x_k, binary_y) in enumerate(train_loader):
                bina_out, semi_outs = k_model(x_k)
                bina_loss = binary_loss_func(bina_out, binary_y)
                k_optimizer.zero_grad()
                bina_loss.backward()
                k_optimizer.step()

            bina_pred_y, test_loss_bina, bina_acc, bina_auc = validation(k_model, val_x_k)
            if bina_auc > best_auc:
                best_auc = bina_auc
                torch.save(k_model, './saved_pkl/temp_pkl/bk_{}_{}.pkl'.format(lr, batch_size))
            print('K_Epoch: ', epo, '| b_loss : %.2f' % test_loss_bina)

    def ensemble_run():
        best_seq_model = torch.load('./saved_pkl/temp_pkl/bs_{}_{}.pkl'.format(lr, batch_size))
        best_k_model = torch.load('./saved_pkl/temp_pkl/bk_{}_{}.pkl'.format(lr, batch_size))
        best_seq_model.eval()
        best_k_model.eval()
        best_auc = 0
        for epo in range(epoch):
            e_model.train()
            for step, (x_b, x_k, binary_y) in enumerate(train_loader):
                with torch.no_grad():
                    bina_outb, semi_outk = best_k_model(x_k)
                    bina_outs, semi_outs = best_seq_model(x_b)
                bina_out = e_model(semi_outs, semi_outk)
                bina_loss = binary_loss_func(bina_out, binary_y)
                e_optimizer.zero_grad()
                bina_loss.backward()
                e_optimizer.step()

            test_bina_outb, test_semi_outk = best_k_model(val_x_k)
            test_bina_outs, test_semi_outs = best_seq_model(val_x_b)
            bina_pred_y, test_loss_bina, bina_acc, bina_auc = validation(e_model, [test_semi_outs, test_semi_outk], True)
            if bina_auc > best_auc:
                best_auc = bina_auc
                torch.save(e_model, './saved_pkl/temp_pkl/be_{}_{}.pkl'.format(lr, batch_size))
            print('Ensem_Epoch: ', epo, '| b_loss : %.2f' % test_loss_bina)

    print('seq_model:\n')
    seq_run()
    print('k_model:\n')
    k_run()
    print('ensemble_model:\n')
    ensemble_run()
    torch.cuda.empty_cache()

    # testing flow
    testing_x, testing_y = read_data('./data/bina_training/testing_seq_bina.csv')
    test_x_b, test_x_k = map2features(testing_x)
    test_y_binary = testing_y

    def test_out(test_model, test_input, model_name, is_ensemble=False):
        if is_ensemble:
            seq_model, k_model, e_model = test_model[0], test_model[1], test_model[2]
            test_s, test_semi_outs = seq_model(test_input[0])
            test_b, test_semi_outk = k_model(test_input[1])
            test_out_bina = e_model(test_semi_outs, test_semi_outk)
        else:
            test_out_bina, elseout = test_model(test_input)
        if is_gpu:
            test_out_bina = test_out_bina.cpu()
        y_tensor_bina = torch.LongTensor(test_y_binary)
        test_loss_bina = binary_loss_func(test_out_bina, y_tensor_bina)
        bina_pred_y = torch.max(test_out_bina, 1)[1].data.numpy()
        bina_acc = float((bina_pred_y == np.array(test_y_binary)).astype(int).sum()) / float(len(test_y_binary))
        post_score = torch.index_select(test_out_bina, 1, torch.tensor([1]))
        post_score = post_score.data.numpy()
        fpr, tpr, thresholds = metrics.roc_curve(np.array(test_y_binary), post_score, pos_label=1)
        bina_auc = metrics.auc(fpr, tpr)
        # with open('output/test_out/b{}_{}_{}.csv'.format(model_name, lr, batch_size), 'w') as f:
        #     for i in range(len(bina_pred_y)):
        #         score = post_score[i][0]
        #         f.write('{},{},{}\n'.format(str(score), bina_pred_y[i], test_y_binary[i]))
        # f.close()
        print('test_{}: loss_{}, acc_{}, auc_{}\n'.format(model_name, test_loss_bina, bina_acc, bina_auc))

    best_seq_model = torch.load('./saved_pkl/temp_pkl/bs_{}_{}.pkl'.format(lr, batch_size))
    best_k_model = torch.load('./saved_pkl/temp_pkl/bk_{}_{}.pkl'.format(lr, batch_size))
    best_e_model = torch.load('./saved_pkl/temp_pkl/be_{}_{}.pkl'.format(lr, batch_size))
    best_seq_model.eval()
    best_k_model.eval()
    best_e_model.eval()
    torch.no_grad()
    test_out([best_seq_model, best_k_model, best_e_model], [test_x_b, test_x_k], 'e', True)
    covert_para()
    print("binary model done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--D', type=str, default='cpu', help="Device: 'gpu' or 'cpu'")
    args = parser.parse_args()
    is_gpu = False
    if torch.cuda.is_available() and args.D == 'gpu':
        is_gpu = True

    print('start training binary model!\n')
    time_start = time.time()
    bina_training(is_gpu)

