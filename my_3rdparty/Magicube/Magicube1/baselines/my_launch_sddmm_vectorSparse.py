import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
import numpy as np



def get_vec_len_from_filename(line):
    if 'pruned_bert' in line:
        pos = line[::-1].find('_')
        line = line[-(pos+1):]
    vec_lens = [2**i for i in range(5, 0, -1)]
    for vec_len in vec_lens:
        if str(vec_len) in line:
            return vec_len
    assert False, "Fail to extract vec_len from file name."




# Args
parser = argparse.ArgumentParser(description='lauch the spmm benchmarks')

#parser.add_argument('--dimK', type=int, default=256, help="the dimension N of the benchmark")
#parser.add_argument('--dimV', type=int, default=8, help="vector length")
#parser.add_argument('--sparsity', choices=['50', '70', '80', '90', '95', '98'], default='70', help='sparsity of the matrix')
#parser.add_argument('--preA', type=int, default=8, help="number of bits for A")
#parser.add_argument('--preB', type=int, default=8, help="number of bits for B")
args = parser.parse_args()

dataset_dir = '/homes/jfangak/sparsetir-artifact/spmm' # os.environ.get('dataset_dir')
# sparsities = ['50', '70', '80', '90', '95', '98']
dimKs = [32, 64, 128, 256, 512]
dimKs = [512]
dimKs = [32, 64, 128, 256, 512]
# vec_lens = [2, 4, 8]

for dimK in dimKs:
    # for vec_len in vec_lens:

    # matrix_list = open('./eval_matrices/my_spmm.txt', 'r')
    # matrix_list = open('./eval_matrices/my_spmm_pruned_bert.txt', 'r')
    matrix_list = open('./eval_matrices/my_spmm_extra_datasets.txt', 'r')
    lines = matrix_list.readlines()
    for i in range(len(lines)):
    #for i in range(1):
        vec_len = get_vec_len_from_filename(lines[i])
        if vec_len not in [2, 4, 8]:
            continue
        # print("dimK: ", dimK, "vec_len: ", vec_len)
        matrix = '%s/%s' % (dataset_dir, lines[i][:-1])
        cmd = './sddmm_benchmark %s %d %d 0 2 1 0 1 1' % (matrix, dimK, vec_len)
        os.system(cmd)

