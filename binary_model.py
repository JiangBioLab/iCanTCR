# coding: utf-8
import pandas
import torch
from torch import nn
import my_utils
import numpy as np
import math
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics


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


def run_one_sample(seq_list, freq_list, seq_model, k_model, e_model, is_gpu):
    top_n = 50

    def seq2features(all_seqs):
        test_x_b, test_x_k = [], []
        for seq in all_seqs:
            used_chain = seq[4:(len(seq) - 1)]
            used_chain_list = list(used_chain)
            each_x_b = get_aaidx_feature(used_chain_list, k=K)  # 将序列转化为AA_index的PCA 特征
            each_x_b = np.array(each_x_b, dtype=float)
            k_3_features = get_k_feature(used_chain, k=3)  # 将序列转化为k_mer的丰度特征
            k_4_features = get_k_feature(used_chain, k=4)  # 将序列转化为k_mer的丰度特征
            k_5_features = get_k_feature(used_chain, k=5)  # 将序列转化为k_mer的丰度特征
            max_seq_shape = MAX_AA_LEN - K + 1 - 5
            max_3_shape = MAX_AA_LEN - 3 + 1 - 5
            max_4_shape = MAX_AA_LEN - 4 + 1 - 5
            max_5_shape = MAX_AA_LEN - 5 + 1 - 5
            if (each_x_b.shape[0]) < max_seq_shape:
                each_x_b = np.pad(each_x_b, ((0, max_seq_shape - (each_x_b.shape[0])), (0, 0)), 'constant',
                                  constant_values=(0, 0))
            assert each_x_b.shape[0] == max_seq_shape
            k3_feature1 = np.pad(np.array(k_3_features[0]), (0, 12 * max_3_shape - len(k_3_features[0])), 'constant')
            k3_feature2 = np.pad(np.array(k_3_features[1]), (0, 12 * max_3_shape - len(k_3_features[1])), 'constant')
            k4_feature1 = np.pad(np.array(k_4_features[0]), (0, 12 * max_4_shape - len(k_4_features[0])), 'constant')
            k4_feature2 = np.pad(np.array(k_4_features[1]), (0, 12 * max_4_shape - len(k_4_features[1])), 'constant')
            k5_feature1 = np.pad(np.array(k_5_features[0]), (0, 12 * max_5_shape - len(k_5_features[0])), 'constant')
            k5_feature2 = np.pad(np.array(k_5_features[1]), (0, 12 * max_5_shape - len(k_5_features[1])), 'constant')
            k_features = np.concatenate((k3_feature1, k3_feature2, k4_feature1, k4_feature2, k5_feature1, k5_feature2))
            k_features = np.array(k_features, dtype=float)
            test_x_b.append(each_x_b.tolist())
            test_x_k.append(k_features.tolist())
        return test_x_b, test_x_k

    test_x_b, test_x_k = seq2features(seq_list)

    if is_gpu:
        test_x_b = torch.FloatTensor(test_x_b).cuda()
        test_x_k = torch.FloatTensor(test_x_k).cuda()
    else:
        test_x_b = torch.FloatTensor(test_x_b)
        test_x_k = torch.FloatTensor(test_x_k)
    semi_outs = seq_model(test_x_b)
    semi_outk = k_model(test_x_k)
    test_out_bina = e_model(semi_outs, semi_outk)
    if is_gpu:
        test_out_bina = test_out_bina.cpu()

    post_score = torch.index_select(test_out_bina, 1, torch.tensor([1]))
    pro_vec = list(post_score.data.numpy())

    high_idx = []
    for idx in range(len(pro_vec)):
        if pro_vec[idx] > 0.6:
            high_idx.append(idx)

    if len(pro_vec) < top_n:
        top_n = len(pro_vec)
    sum_value = 0
    for i in range(top_n):
        sum_value += pro_vec[i]*freq_list[i]
    temp_value = 1 - math.exp(-sum_value)
    weight_score = math.sqrt(temp_value)

    return weight_score, high_idx


def bina_run(out_dir, out_name, sample_ids, sample_dict, is_gpu, is_only):
    lr, batch_size = 0.001, 64
    if is_gpu:
        best_seq_model = B_Seq_model().cuda()
        best_k_model = B_K_model().cuda()
        best_e_model = B_EnsembleModel().cuda()
        seq_state_dict = torch.load('./trained_model/bs_{}_{}_dict.pkl'.format(lr, batch_size))
        k_state_dict = torch.load('./trained_model/bk_{}_{}_dict.pkl'.format(lr, batch_size))
        e_state_dict = torch.load('./trained_model/be_{}_{}_dict.pkl'.format(lr, batch_size))
    else:
        best_seq_model = B_Seq_model()
        best_k_model = B_K_model()
        best_e_model = B_EnsembleModel()
        seq_state_dict = torch.load('./trained_model/bs_{}_{}_dict.pkl'.format(lr, batch_size), map_location='cpu')
        k_state_dict = torch.load('./trained_model/bk_{}_{}_dict.pkl'.format(lr, batch_size), map_location='cpu')
        e_state_dict = torch.load('./trained_model/be_{}_{}_dict.pkl'.format(lr, batch_size), map_location='cpu')

    best_seq_model.load_state_dict(seq_state_dict)
    best_k_model.load_state_dict(k_state_dict)
    best_e_model.load_state_dict(e_state_dict)
    best_seq_model.eval()
    best_k_model.eval()
    best_e_model.eval()
    torch.no_grad()

    bina_result = {}
    for k, v in sample_dict.items():
        s_id = k
        seq_list, freq_list = v[0], v[1]
        try:
            weight_score, high_idx = run_one_sample(seq_list, freq_list, best_seq_model, best_k_model, best_e_model, is_gpu)
            bina_result[s_id] = [weight_score, high_idx]
        except RuntimeError as e:
            print(e)
        except:
            bina_result[s_id] = ['nan', -1]

    if is_only:
        csv_file = 'iCanTCR_binary_{}.csv'.format(out_name)
        csv_path = str(out_dir) + '/' + str(csv_file)
        with open(csv_path, 'w') as f:
            f.write('{}\n'.format('Binary results:'))
            f.write('{},{}\n'.format('Sample_id', 'Score'))
            for s_id in sample_ids:
                try:
                    cancer_score = '%.4f' % bina_result[s_id][0]
                except:
                    cancer_score = bina_result[s_id][0]
                f.write('{},{}\n'.format(s_id, str(cancer_score)))
        f.close()

    print('binary task done!')
    return bina_result
