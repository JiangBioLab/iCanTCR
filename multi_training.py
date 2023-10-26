# coding: utf-8
import torch
from torch import nn
import argparse
import numpy as np
import my_utils
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

torch.manual_seed(102)  # reproducible

MIN_AA_LEN = 11
MAX_AA_LEN = 19
RNN_INPUTSIZE = 20
K = 1
PC = 18

CFG = {  # CNN config
    'cfg00': [16, 'M', 16, 'M'],
    'cfg01': [16, 'M', 32, 'M'],
    'cfg02': [8, 'M', 8, 'M'],
    'cfg03': [64, 'M'],
    'cfg04': [8, 'M', 16, 'M'],
}
out_nodes = 7
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
            aa_name = items[0]
            aa_pca = [float(x) for x in items[1:]]
            aa_pc_map[aa_name] = aa_pca[:18]  # 前18个包含信息>99%, 前14个包含信息> 95%
    f.close()
    return aa_pc_map


# 将序列转化为k_mer的丰度特征
def get_k_feature(seq, k):
    cancer_l = ['BRCA', 'NSCLC', 'ESCA', 'GBM', 'LIHC', 'SARC', 'MELA', 'PRC', 'BLCA', 'HNSCC', 'MCC']
    k_feature, k_feature2 = [], []
    t1, t2 = 0, 0
    if k == 3:
        cancer_k_dict = CANCER_3_DICT.tolist()
    elif k == 4:
        cancer_k_dict = CANCER_4_DICT.tolist()
    elif k == 5:
        cancer_k_dict = CANCER_5_DICT.tolist()
    for i in range(len(seq) - k + 1):
        a_kmer = seq[i:i + k]
        one_feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        one_feature2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        n = 0
        for cancer in cancer_l:
            k_mer_d1 = cancer_k_dict[cancer][0]
            k_mer_d2 = cancer_k_dict[cancer][1]
            if a_kmer in k_mer_d1:
                fea1 = float(k_mer_d1[a_kmer])
                if fea1 < t1:
                    fea1 = 0
                one_feature[n] = fea1
            if a_kmer in k_mer_d2:
                fea2 = float(k_mer_d2[a_kmer])
                if fea2 < t2:
                    fea2 = 0
                one_feature2[n] = fea2
            n += 1
        k_feature.extend(one_feature)
        k_feature2.extend(one_feature2)
    assert len(k_feature) == len(k_feature2)
    joint_features = [k_feature, k_feature2]
    return joint_features


# 将序列转化为AA_index的PCA 特征
def get_aaidx_feature(seq, k=1):
    aa_pc_map = load_aa_pc()
    aaidx_features = my_utils.get_aaseq_kmer(seq, aa_pc_map, k=k)
    return aaidx_features


class DataLoad(Dataset):
    def __init__(self, data, data_label, is_gpu=False, transform=None):
        self.b_feature_matrixs = []
        self.k_features = []
        self.cloneFracs = []
        self.mutil_labels = []
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

            k3_feature = np.pad(np.array(k_feature_list[0][0]), (0, 11 * self.max_k3_len - len(k_feature_list[0][0])),
                                'constant')
            k3_feature2 = np.pad(np.array(k_feature_list[0][1]), (0, 11 * self.max_k3_len - len(k_feature_list[0][1])),
                                 'constant')
            k4_feature = np.pad(np.array(k_feature_list[1][0]), (0, 11 * self.max_k4_len - len(k_feature_list[1][0])),
                                'constant')
            k4_feature2 = np.pad(np.array(k_feature_list[1][1]), (0, 11 * self.max_k4_len - len(k_feature_list[1][1])),
                                 'constant')
            k5_feature = np.pad(np.array(k_feature_list[2][0]), (0, 11 * self.max_k5_len - len(k_feature_list[2][0])),
                                'constant')
            k5_feature2 = np.pad(np.array(k_feature_list[2][1]), (0, 11 * self.max_k5_len - len(k_feature_list[2][1])),
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
            mutil_label = data_label[i]  # [0]
            self.mutil_labels.append(mutil_label)

        if is_gpu:
            self.mutil_labels = torch.LongTensor(self.mutil_labels).cuda()
        else:
            self.mutil_labels = torch.LongTensor(self.mutil_labels)

    def __getitem__(self, idx):
        idx = idx % len(self)
        b_feature_matrix = self.b_feature_matrixs[idx]
        k_features = self.k_features[idx]
        mutil_label = self.mutil_labels[idx]
        return b_feature_matrix, k_features, mutil_label  # , binary_label

    def __len__(self):
        return len(self.mutil_labels)


class M_Seq_model(nn.Module):
    def __init__(self):
        super(M_Seq_model, self).__init__()

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

    def forward(self, x_b):
        cnn_x_b = torch.unsqueeze(x_b, 1)
        cnn_b_3 = self.cnn2d_b_3(cnn_x_b)
        cnn_b_3 = cnn_b_3.view(cnn_b_3.size(0), -1)  # 展开多维的卷积图
        cnn_b_4 = self.cnn2d_b_4(cnn_x_b)
        cnn_b_4 = cnn_b_4.view(cnn_b_4.size(0), -1)  # 展开多维的卷积图
        cnn_b_5 = self.cnn2d_b_5(cnn_x_b)
        cnn_b_5 = cnn_b_5.view(cnn_b_5.size(0), -1)  # 展开多维的卷积图

        rnn_out_b, (h_n, h_c) = self.rnnlayer_b(x_b, None)
        out_b1 = nn.functional.dropout(self.rnn_dense(rnn_out_b[:, -1, :]), p=0.3, training=self.training)  # 选取最后一个时间点的rnn_out的输出

        cnn_combine = torch.cat((cnn_b_3, cnn_b_4, cnn_b_5, out_b1), 1)

        out1 = self.FClayer_1(cnn_combine)
        out1 = nn.functional.relu(out1)
        out2 = nn.functional.dropout(self.FClayer_2(out1), p=0.3, training=self.training)
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


class M_K_model(nn.Module):
    def __init__(self):
        super(M_K_model, self).__init__()
        self.input_layer = nn.Linear(726, 1024)
        self.dense_layer1 = nn.Linear(1024, 256)
        self.dense_layer2 = nn.Linear(256, 64)

    def forward(self, _input):
        out1 = self.input_layer(_input)
        out1 = nn.functional.relu(out1)
        out2 = nn.functional.dropout(self.dense_layer1(out1), p=0.5, training=self.training)
        out2 = nn.functional.relu(out2)
        out3 = nn.functional.dropout(self.dense_layer2(out2), p=0.5, training=self.training)
        out3 = nn.functional.relu(out3)
        return out2


class M_EnsembleModel(nn.Module):
    def __init__(self):
        super(M_EnsembleModel, self).__init__()
        self.input_layer = nn.Linear(320, 256)
        self.dense_layer = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, out_nodes)

    def forward(self, s_input, k_input):
        input_combine = torch.cat((s_input, k_input), 1)
        out1 = self.input_layer(input_combine)
        out1 = nn.functional.relu(out1)
        out2 = nn.functional.dropout(self.dense_layer(out1), p=0.5, training=self.training)
        out2 = nn.functional.relu(out2)
        out = self.output_layer(out2)
        out = nn.functional.softmax(out, dim=1)
        return out


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

        self.outlayer = nn.Linear(64, out_nodes)

    def forward(self, x_b):
        cnn_x_b = torch.unsqueeze(x_b, 1)
        cnn_b_3 = self.cnn2d_b_3(cnn_x_b)
        cnn_b_3 = cnn_b_3.view(cnn_b_3.size(0), -1)  # 展开多维的卷积图
        cnn_b_4 = self.cnn2d_b_4(cnn_x_b)
        cnn_b_4 = cnn_b_4.view(cnn_b_4.size(0), -1)  # 展开多维的卷积图
        cnn_b_5 = self.cnn2d_b_5(cnn_x_b)
        cnn_b_5 = cnn_b_5.view(cnn_b_5.size(0), -1)  # 展开多维的卷积图

        rnn_out_b, (h_n, h_c) = self.rnnlayer_b(x_b, None)
        out_b1 = nn.functional.dropout(self.rnn_dense(rnn_out_b[:, -1, :]), p=0.3, training=self.training)  # 选取最后一个时间点的rnn_out的输出

        cnn_combine = torch.cat((cnn_b_3, cnn_b_4, cnn_b_5, out_b1), 1)

        out1 = self.FClayer_1(cnn_combine)
        out1 = nn.functional.relu(out1)
        out2 = nn.functional.dropout(self.FClayer_2(out1), p=0.3, training=self.training)
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
        self.input_layer = nn.Linear(726, 1024)
        self.dense_layer1 = nn.Linear(1024, 256)
        self.dense_layer2 = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, out_nodes)

    def forward(self, _input):
        out1 = self.input_layer(_input)
        out1 = nn.functional.relu(out1)
        out2 = nn.functional.dropout(self.dense_layer1(out1), p=0.5, training=self.training)
        out2 = nn.functional.relu(out2)
        out3 = nn.functional.dropout(self.dense_layer2(out2), p=0.5, training=self.training)
        out3 = nn.functional.relu(out3)
        out = self.output_layer(out3)
        out = nn.functional.softmax(out, dim=1)
        return out, out2


class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.input_layer = nn.Linear(320, 256)
        self.dense_layer = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, out_nodes)

    def forward(self, s_input, k_input):
        input_combine = torch.cat((s_input, k_input), 1)
        out1 = self.input_layer(input_combine)
        out1 = nn.functional.relu(out1)
        out2 = nn.functional.dropout(self.dense_layer(out1), p=0.5, training=self.training)
        out2 = nn.functional.relu(out2)
        out = self.output_layer(out2)
        out = nn.functional.softmax(out, dim=1)
        return out


def split_file(x, y, test_size=0.2, is_random=False):
    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(y) is not np.ndarray:
        y = np.array(y)
    random_state = None
    if is_random:
        random_state = 111
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def covert_para_m():
    lr, batch_size = 0.001, 64
    old_seq_model = torch.load('./saved_pkl/temp_pkl/m3s_{}_{}.pkl'.format(lr, batch_size), map_location='cpu')
    old_k_model = torch.load('./saved_pkl/temp_pkl/m3k_{}_{}.pkl'.format(lr, batch_size), map_location='cpu')
    old_e_model = torch.load('./saved_pkl/temp_pkl/m3e_{}_{}.pkl'.format(lr, batch_size), map_location='cpu')

    new_seq_model = M_Seq_model()
    new_k_model = M_K_model()
    new_e_model = M_EnsembleModel()

    new_seq_model.load_state_dict(old_seq_model.state_dict(), strict=False)
    new_k_model.load_state_dict(old_k_model.state_dict(), strict=False)
    new_e_model.load_state_dict(old_e_model.state_dict(), strict=False)

    torch.save(new_seq_model.state_dict(), './saved_pkl/ms_{}_{}_dict.pkl'.format(lr, batch_size))
    torch.save(new_k_model.state_dict(), './saved_pkl/mk_{}_{}_dict.pkl'.format(lr, batch_size))
    torch.save(new_e_model.state_dict(), './saved_pkl/me_{}_{}_dict.pkl'.format(lr, batch_size))


def multi_run(is_gpu=False):
    cancer_list = ['BRCA', 'BLCA', 'LIHC', 'NSCLC', 'SARC', 'MELA', 'OT']
    CANCER_LABEL = {  # cancer label
        'BRCA': 0,
        'BLCA': 1,
        'LIHC': 2,
        'NSCLC': 3,
        'SARC': 4,
        'MELA': 5,
        'OT': 6,
    }

    def read_data(path):
        pd_data = pd.read_csv(path)
        aa_seqs = pd_data['AA_seq'].values.tolist()
        type = pd_data['Label'].values.tolist()
        labels = []
        for i in type:
            labels.append(CANCER_LABEL[i])

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
            k3_feature1 = np.pad(np.array(each_k[0][0]), (0, 11 * max_3_shape - len(each_k[0][0])), 'constant')
            k3_feature2 = np.pad(np.array(each_k[0][1]), (0, 11 * max_3_shape - len(each_k[0][1])), 'constant')
            k4_feature1 = np.pad(np.array(each_k[1][0]), (0, 11 * max_4_shape - len(each_k[1][0])), 'constant')
            k4_feature2 = np.pad(np.array(each_k[1][1]), (0, 11 * max_4_shape - len(each_k[1][1])), 'constant')
            k5_feature1 = np.pad(np.array(each_k[2][0]), (0, 11 * max_5_shape - len(each_k[2][0])), 'constant')
            k5_feature2 = np.pad(np.array(each_k[2][1]), (0, 11 * max_5_shape - len(each_k[2][1])), 'constant')
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
    loss_func = nn.CrossEntropyLoss()

    training_x, training_y = [], []
    for cancer in cancer_list:
        one_training_x, one_training_y = read_data('./data/multi_training/{}_training.csv'.format(cancer))
        training_x.extend(one_training_x)
        training_y.extend(one_training_y)

    x_tra, x_val, y_tra, y_val = split_file(training_x, training_y, test_size=0.1, is_random=True)
    val_x_b, val_x_k = map2features(x_val)
    val_y_mutil = y_val
    train_dataset = DataLoad(x_tra, y_tra, is_gpu)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    seq_model.train()
    k_model.train()
    e_model.train()

    def validation(test_model, test_input, is_ensemble=False):
        if is_ensemble:
            test_out_mutil = test_model(test_input[0], test_input[1])
        else:
            test_out_mutil, elseout = test_model(test_input)
        if is_gpu:
            test_out_mutil = test_out_mutil.cpu()
        test_model.eval()
        y_tensor_mutil = torch.LongTensor(val_y_mutil)
        test_loss_mutil = loss_func(test_out_mutil, y_tensor_mutil)
        pred_y = torch.max(test_out_mutil, 1)[1].data.numpy()
        mutil_acc = float((pred_y == np.array(val_y_mutil)).astype(int).sum()) / float(len(val_y_mutil))
        return pred_y, test_loss_mutil, mutil_acc

    def seq_run():
        best_acc = 0
        for epo in range(epoch):
            seq_model.train()
            for step, (x_b, x_k, mutil_y) in enumerate(train_loader):
                mutil_out, elseout = seq_model(x_b)
                mutil_loss = loss_func(mutil_out, mutil_y)
                s_optimizer.zero_grad()
                mutil_loss.backward()
                s_optimizer.step()

            mutil_pred_y, test_loss_mutil, mutil_acc = validation(seq_model, val_x_b)
            if mutil_acc > best_acc:
                best_acc = mutil_acc
                torch.save(seq_model, './saved_pkl/temp_pkl/m3s_{}_{}.pkl'.format(lr, batch_size))
            print('Epoch: ', epo, '| m_loss : %.2f' % test_loss_mutil)

    def k_run():
        best_acc = 0
        for epo in range(epoch):
            k_model.train()
            for step, (x_b, x_k, mutil_y) in enumerate(train_loader):
                mutil_out, elseout = k_model(x_k)
                mutil_loss = loss_func(mutil_out, mutil_y)
                k_optimizer.zero_grad()
                mutil_loss.backward()
                k_optimizer.step()

            mutil_pred_y, test_loss_mutil, mutil_acc = validation(k_model, val_x_k)
            if mutil_acc > best_acc:
                best_acc = mutil_acc
                torch.save(k_model, './saved_pkl/temp_pkl/m3k_{}_{}.pkl'.format(lr, batch_size))
            print('Epoch: ', epo, '| m_loss : %.2f' % test_loss_mutil)

    def ensemble_run():
        best_seq_model = torch.load('./saved_pkl/temp_pkl/m3s_{}_{}.pkl'.format(lr, batch_size))
        best_k_model = torch.load('./saved_pkl/temp_pkl/m3k_{}_{}.pkl'.format(lr, batch_size))
        best_seq_model.eval()
        best_k_model.eval()
        best_acc = 0
        for epo in range(epoch):
            e_model.train()
            for step, (x_b, x_k, mutil_y) in enumerate(train_loader):
                with torch.no_grad():
                    b, semi_outk = best_k_model(x_k)
                    s, semi_outs = best_seq_model(x_b)
                mutil_out = e_model(semi_outs, semi_outk)
                mutil_loss = loss_func(mutil_out, mutil_y)
                e_optimizer.zero_grad()
                mutil_loss.backward()
                e_optimizer.step()

            test_b, test_semi_outk = best_k_model(val_x_k)
            test_s, test_semi_outs = best_seq_model(val_x_b)
            mutil_pred_y, test_loss_mutil, mutil_acc = validation(e_model, [test_semi_outs, test_semi_outk], True)
            if mutil_acc > best_acc:
                best_acc = mutil_acc
                torch.save(e_model, './saved_pkl/temp_pkl/m3e_{}_{}.pkl'.format(lr, batch_size))
            print('Epoch: ', epo, '| m_loss : %.2f' % test_loss_mutil)

    print('seq_model:\n')
    seq_run()
    print('k_model:\n')
    k_run()
    print('ensemble_model:\n')
    ensemble_run()

    torch.cuda.empty_cache()

    # testing flow
    testing_x, testing_y = [], []
    for cancer in cancer_list:
        one_testing_x, one_testing_y = read_data('./data/multi_training/{}_testing.csv'.format(cancer))
        testing_x.extend(one_testing_x)
        testing_y.extend(one_testing_y)
    test_x_b, test_x_k = map2features(testing_x)
    test_y_mutil = testing_y

    def test_out(test_model, test_input, model_name, is_ensemble=False):
        if is_ensemble:
            seq_model, k_model, e_model = test_model[0], test_model[1], test_model[2]
            test_s, test_semi_outs = seq_model(test_input[0])
            test_b, test_semi_outk = k_model(test_input[1])
            test_out_mutil = e_model(test_semi_outs, test_semi_outk)
        else:
            test_out_mutil, elseout = test_model(test_input)
        if is_gpu:
            test_out_mutil = test_out_mutil.cpu()
        y_tensor_mutil = torch.LongTensor(test_y_mutil)
        test_loss_mutil = loss_func(test_out_mutil, y_tensor_mutil)
        pred_y = torch.max(test_out_mutil, 1)[1].data.numpy()
        # f1_macro = f1_score(test_y_mutil, pred_y, average='macro')
        mutil_acc = float((pred_y == np.array(y_tensor_mutil)).astype(int).sum()) / float(len(y_tensor_mutil))
        print('test_{}: loss_{}, acc_{}\n'.format(model_name, test_loss_mutil, mutil_acc))

    best_seq_model = torch.load('./saved_pkl/temp_pkl/m3s_{}_{}.pkl'.format(lr, batch_size))
    best_k_model = torch.load('./saved_pkl/temp_pkl/m3k_{}_{}.pkl'.format(lr, batch_size))
    best_e_model = torch.load('./saved_pkl/temp_pkl/m3e_{}_{}.pkl'.format(lr, batch_size))
    best_seq_model.eval()
    best_k_model.eval()
    best_e_model.eval()
    torch.no_grad()

    test_out([best_seq_model, best_k_model, best_e_model], [test_x_b, test_x_k], 'e', True)
    covert_para_m()
    print("binary model done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--D', type=str, default='cpu', help="Device: 'gpu' or 'cpu'")
    args = parser.parse_args()
    is_gpu = False
    if torch.cuda.is_available() and args.D == 'gpu':
        is_gpu = True

    print('start training multi model!\n')
    time_start = time.time()
    multi_run(is_gpu)


