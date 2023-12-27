# coding:utf-8
import argparse
import torch
import os, time, re
from binary_model import B_Seq_model
from binary_model import B_K_model
from binary_model import B_EnsembleModel
from multi_model import M_Seq_model
from multi_model import M_K_model
from multi_model import M_EnsembleModel
import binary_model as bm
import multi_model as mm


MIN_AA_LEN = 11
MAX_AA_LEN = 19


def read_file(file_path):
    seq_list, freq_list = [], []
    pat = re.compile('[\\*_~XB]')
    with open(file_path, 'r') as f:
        head_line = True
        n = 0
        for line in f:
            if head_line:
                head_line = False
                continue
            items = line.strip().split('\t')
            aa_seq, freq = items[0], items[1]
            try:
                freq = float(items[1])
            except Exception as e:
                print("Clonal frequency is not a numeric type, please check your input file")
                continue
            if aa_seq[0] != 'C' or aa_seq[-1] != 'F':
                continue
            if len(aa_seq) < MIN_AA_LEN or len(aa_seq) > MAX_AA_LEN:
                continue
            if len(pat.findall(aa_seq)) > 0:
                continue
            seq_list.append(aa_seq)
            freq_list.append(freq)
            n += 1
            if n >= 50:
                break
    f.close()
    return seq_list, freq_list


def run_pred(dir_path, out_dir, out_name, is_gpu):
    pat = re.compile('[\\*_~XB]')   # 正则表达式检测特殊字符

    file_list = [f for f in os.listdir(dir_path) if f.endswith('.tsv')]
    file_list = sorted(file_list)
    sample_dict = {}
    sample_ids = []
    for file in file_list:
        sample_id = file.split('.')[0]
        sample_ids.append(sample_id)
        file_path = dir_path + '/' + file
        seq_list, freq_list = read_file(file_path)
        sample_dict[sample_id] = {}
        sample_dict[sample_id][0] = seq_list
        sample_dict[sample_id][1] = freq_list

    bina_result = bm.bina_run(out_dir, out_name, sample_ids, sample_dict, is_gpu, is_only=False)
    mm.multi_run(out_dir, out_name, sample_ids, sample_dict, bina_result, is_gpu, is_only=False)


def run_bina_only(dir_path, out_dir, out_name, is_gpu):
    pat = re.compile('[\\*_~XB]')   # 正则表达式检测特殊字符

    file_list = [f for f in os.listdir(dir_path) if f.endswith('.tsv')]
    file_list = sorted(file_list)
    sample_dict = {}
    sample_ids = []
    for file in file_list:
        sample_id = file.split('.')[0]
        sample_ids.append(sample_id)
        file_path = dir_path + '/' + file
        seq_list, freq_list = read_file(file_path)
        sample_dict[sample_id] = {}
        sample_dict[sample_id][0] = seq_list
        sample_dict[sample_id][1] = freq_list

    bina_result = bm.bina_run(out_dir, out_name, sample_ids, sample_dict, is_gpu, is_only=True)


def run_multi_only(dir_path, out_dir, out_name, is_gpu):
    pat = re.compile('[\\*_~XB]')   # 正则表达式检测特殊字符

    file_list = [f for f in os.listdir(dir_path) if f.endswith('.tsv')]
    file_list = sorted(file_list)
    sample_dict = {}
    sample_ids = []
    for file in file_list:
        sample_id = file.split('.')[0]
        sample_ids.append(sample_id)
        file_path = dir_path + '/' + file
        seq_list, freq_list = read_file(file_path)
        sample_dict[sample_id] = {}
        sample_dict[sample_id][0] = seq_list
        sample_dict[sample_id][1] = freq_list

    bina_result = bm.bina_run(out_dir, out_name, sample_ids, sample_dict, is_gpu, is_only=False)
    mm.multi_run(out_dir, out_name, sample_ids, sample_dict, bina_result, is_gpu, is_only=True)


if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--I', type=str, default='', help="Input folder")
    parser.add_argument('--O', type=str, default='output', help="output folder")
    parser.add_argument('--D', type=str, default='cpu', help="Device: 'gpu' or 'cpu'")
    parser.add_argument('--T', type=str, default='general', help="Class Type: 'binary' or 'multi' or 'general'")

    args = parser.parse_args()

    if args.I == '':
        print('No Input!')
        exit()

    is_gpu = False
    if torch.cuda.is_available() and args.D == 'gpu':
        is_gpu = True

    in_dir = args.I
    out_dir = args.O
    if not os.path.exists(in_dir):
        print('The input path does not exist!')
        exit()
    if in_dir.endswith('/'):
        in_dir = in_dir[:-1]
    if out_dir == '':
        out_dir = in_dir.split('/')[-1] + '_result'

    if not os.path.exists(out_dir):
        os.system('mkdir ' + out_dir)
    out_name = in_dir.split('/')[-1] + '_result'

    class_type = args.T
    if class_type == 'general':
        run_pred(in_dir, out_dir, out_name, is_gpu)
    elif class_type == 'binary':
        run_bina_only(in_dir, out_dir, out_name, is_gpu)
    elif class_type == 'multi':
        run_multi_only(in_dir, out_dir, out_name, is_gpu)
    else:
        print('Class Type Error')
        exit()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

