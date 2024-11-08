# coding: utf-8
import torch
from torch import nn
import numpy as np
import my_utils

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


def run_one_sample(seq_list, seq_model, k_model, e_model, is_gpu):

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
            k3_feature1 = np.pad(np.array(k_3_features[0]), (0, 11 * max_3_shape - len(k_3_features[0])), 'constant')
            k3_feature2 = np.pad(np.array(k_3_features[1]), (0, 11 * max_3_shape - len(k_3_features[1])), 'constant')
            k4_feature1 = np.pad(np.array(k_4_features[0]), (0, 11 * max_4_shape - len(k_4_features[0])), 'constant')
            k4_feature2 = np.pad(np.array(k_4_features[1]), (0, 11 * max_4_shape - len(k_4_features[1])), 'constant')
            k5_feature1 = np.pad(np.array(k_5_features[0]), (0, 11 * max_5_shape - len(k_5_features[0])), 'constant')
            k5_feature2 = np.pad(np.array(k_5_features[1]), (0, 11 * max_5_shape - len(k_5_features[1])), 'constant')
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
    test_out_multi = e_model(semi_outs, semi_outk)
    if is_gpu:
        test_out_multi = test_out_multi.cpu()

    pro_vec = test_out_multi.data.numpy()

    return pro_vec


def multi_run(out_dir, out_name, sample_ids, sample_dict, bina_result, is_gpu, is_only=False):

    def soft_max(inList):
        list_sum = 0
        for e in inList:
            list_sum += e
        outList = []
        for i in inList:
            i_soft = i/list_sum
            outList.append(i_soft)
        return outList

    label_dict = {
        0: 'BRCA',
        1: 'BLCA',
        2: 'LIHC',
        3: 'NSCLC',
        4: 'SARC',
        5: 'MELA',
        6: 'Other Types',
    }

    lr, batch_size = 0.001, 64
    if is_gpu:
        best_seq_model = M_Seq_model().cuda()
        best_k_model = M_K_model().cuda()
        best_e_model = M_EnsembleModel().cuda()
        seq_state_dict = torch.load('./saved_model/ms_{}_{}_dict.pkl'.format(lr, batch_size))
        k_state_dict = torch.load('./saved_model/mk_{}_{}_dict.pkl'.format(lr, batch_size))
        e_state_dict = torch.load('./saved_model/me_{}_{}_dict.pkl'.format(lr, batch_size))
    else:
        best_seq_model = M_Seq_model()
        best_k_model = M_K_model()
        best_e_model = M_EnsembleModel()
        seq_state_dict = torch.load('./saved_model/ms_{}_{}_dict.pkl'.format(lr, batch_size), map_location='cpu')
        k_state_dict = torch.load('./saved_model/mk_{}_{}_dict.pkl'.format(lr, batch_size), map_location='cpu')
        e_state_dict = torch.load('./saved_model/me_{}_{}_dict.pkl'.format(lr, batch_size), map_location='cpu')


    best_seq_model.load_state_dict(seq_state_dict)
    best_k_model.load_state_dict(k_state_dict)
    best_e_model.load_state_dict(e_state_dict)
    best_seq_model.eval()
    best_k_model.eval()
    best_e_model.eval()
    torch.no_grad()

    multi_result = {}
    for k, v in sample_dict.items():
        s_id = k
        seq_list, freq_list = v[0], v[1]

        try:
            pro_vec = run_one_sample(seq_list, best_seq_model, best_k_model, best_e_model, is_gpu)

            score_prob = [0 for x in range(7)]
            cancer_score = bina_result[s_id][0]
            high_prob_idx = bina_result[s_id][1]
            if high_prob_idx == -1:
                multi_result[s_id] = 'Unknown'
                continue

            for idx in high_prob_idx:
                weight_prob_vec = freq_list[idx] * pro_vec[idx]
                score_prob += weight_prob_vec

            sm_prob_v = soft_max(list(score_prob))
            pred_label = np.array(sm_prob_v).argmax()
            pred_class = label_dict[pred_label]
            if is_only == False:
                if cancer_score < 0.26:
                    pred_class = 'Other Types'
            multi_result[s_id] = pred_class
        except RuntimeError as e:
            print(e)
        except:
            multi_result[s_id] = 'Unknown'

    if is_only:
        csv_file = 'iCanTCR_{}_multi.csv'.format(out_name)
        csv_path = str(out_dir) + '/' + str(csv_file)
        with open(csv_path, 'w') as f:
            f.write('{}\n'.format('Multiple results:'))
            f.write('{},{}\n'.format('Sample_id', 'Pred_class'))
            for s_id in sample_ids:
                cancer_type = multi_result[s_id]
                f.write('{},{}\n'.format(s_id, cancer_type))
        f.close()

    else:
        csv_file = 'iCanTCR_{}.csv'.format(out_name)
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

            f.write('\n')
            f.write('{}\n'.format('Multiple results:'))
            f.write('{},{}\n'.format('Sample_id', 'Pred_class'))
            for s_id in sample_ids:
                cancer_type = multi_result[s_id]
                f.write('{},{}\n'.format(s_id, cancer_type))
        f.close()
    print('multi task done!')
