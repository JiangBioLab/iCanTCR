# coding: utf-8

import numpy as np
# from allennlp.commands.elmo import ElmoEmbedder
# from pathlib import Path


def get_aaseq_pc(sequence_list, aa_pc_map):
    value_matrix = []
    for i in range(len(sequence_list)):
        value_matrix.append(aa_pc_map[sequence_list[i]])
    return value_matrix


# 通过序列，获取该序列的k_mers矩阵的向量表示
def get_aaseq_kmer(sequence_list, aa_map, k=3):
    value_matrix = []
    zero_list = np.zeros(18).tolist()
    for i in range(len(sequence_list)):
        t = sequence_list[i:(i+k)]
        values = []
        if len(t) == k:     # 去掉最后 k-1 个长度小于k 的 mer
            for j in range(k):
                if t[j] not in aa_map:  # 遇到乱码用零代替
                    values.append(zero_list)
                else:
                    values.append(aa_map[t[j]])
            values = np.array(values).flatten()
            values = [float(x) for x in values]
            value_matrix.append(np.array(values))
    return value_matrix

#
# def get_seqVec(sequence_list):
#     model_dir = Path('data/seqveq/uniref50_v2')
#     weights = model_dir / 'weights.hdf5'
#     options = model_dir / 'options.json'
#     embedder = ElmoEmbedder(options, weights, cuda_device = -1)  # cuda_device=-1 for CPU
#     # seq = 'SEQWENCE'  # your amino acid sequence
#     embedding = embedder.embed_sentence(sequence_list)  # List-of-Lists with shape [3,L,1024]
#     return embedding


# seq = 'SEQWENCE'
# embed = get_seqVec(list(seq))
# print(embed)