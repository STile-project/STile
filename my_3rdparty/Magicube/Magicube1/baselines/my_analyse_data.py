# 分析profile的结果
'''
# sddmm data
mma + warp shfl
Sparse matrix: /homes/jfangak/sparsetir-artifact/spmm/data_for_Magicube/citeseer2.txt
Problem size: M: 3328, N: 3327, nnz: 18178, K: 32, Vec_length: 2
vectorSparse SDDMM runtime 0.148068 ms

'''
'''
# spmm data
Sparse matrix: /homes/jfangak/sparsetir-artifact/spmm/data_for_Magicube/citeseer2.txt
vec_len: 2 m_vec: 1664 m: 3328 n: 32 k: 3327
Using WMMA 
vectorSparse SpMM runtime 0.0286741 ms

'''

def get_data_name_from_filename(line):
	names = ['citeseer', 'cora', 'ppi', 'pubmed',  'arxiv', 'proteins', 'reddit']
	for name in names:
		if name in line:
			return name
	assert False, "Fail to extract dataset name from file name."



def get_vec_len_from_filename(line):
    vec_lens = [2**i for i in range(5, 0, -1)]
    for vec_len in vec_lens:
        if str(vec_len) in line:
            return vec_len
    assert False, "Fail to extract vec_len from file name."



sddmm_file = 'my_sddmm_vectorSparse2.txt'
spmm_file = 'my_spmm_vectorSparse.txt'


cost_spmm = dict()
with open(spmm_file, 'r') as f:
	lines = f.readlines()
	name, n, vec_len, cost = None, None, None, None
	for line in lines:
		if 'Sparse matrix:' in line:
			# get the name of the dataset
			name = get_data_name_from_filename(line)
			vec_len = get_vec_len_from_filename(line)
		elif 'vec_len' in line:
			# get the n value of the test op
			tmp = line.split()
			n = tmp[-3]
		elif 'runtime' in line:
			# get the measurement latency
			pos0 = line.find('runtime') + len('runtime')
			pos1 = line.find('ms')
			cost = float(line[pos0:pos1])
			assert name!=None
			if (name, n) not in cost_spmm:
				cost_spmm[(name, n)] = list()
			cost_spmm[(name, n)].append(cost)
			name, n, vec_len, cost = None, None, None, None



# get cost dict for SDDMM
cost_sddmm = dict()
with open(sddmm_file, 'r') as f:
	lines = f.readlines()
	name, k, vec_len, cost = None, None, None, None
	for line in lines:
		if 'Sparse matrix:' in line:
			# get the name of the dataset
			name = get_data_name_from_filename(line)
			vec_len = get_vec_len_from_filename(line)
		elif 'Vec_length' in line:
			# get the n value of the test op
			tmp = line.split()
			k = tmp[-3]
		elif 'runtime' in line:
			# get the measurement latency
			pos0 = line.find('runtime') + len('runtime')
			pos1 = line.find('ms')
			cost = float(line[pos0:pos1])
			assert name!=None
			if (name, k) not in cost_sddmm:
				cost_sddmm[(name, k)] = list()
			cost_sddmm[(name, k)].append(cost)
			name, k, vec_len, cost = None, None, None, None


# select the best result from different vector lengths as the final cost
for k, v in cost_spmm.items():
	cost_spmm[k] = min(v)


for k, v in cost_sddmm.items():
	cost_sddmm[k] = min(v)


# 这个是我们画在图里的
from scipy.stats import gmean
import numpy as np

labels = ['citeseer', 'cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit']
feat_sizes = [32, 64, 128, 256, 512]
# 我们还需要获得cusparse的cost才能求normalized cost。
cusparse = np.asarray([0.04018176, 0.042670082, 0.047431685, 0.051589128, 0.055726081, 0.041451521, 0.04263936, 0.042946558, 0.045844484, 0.049367037, 0.206694394, 0.245503992, 0.318740457, 0.635535359, 1.324963808, 0.246937603, 0.319815665, 0.463267863, 0.86581248, 1.697669268, 47.5530777, 54.74970627, 62.60726166, 113.7134094, 203.0335999, 0.046469122, 0.051148802, 0.065966077, 0.101765119, 0.188784644, 141.9566345, 127.0233612, 127.1853027, 244.9338989, 408.3432007])
spmm_data = np.asarray([ cost_spmm[(label, str(feat))] for label in labels for feat in feat_sizes])
# 求geometric mean
norm_spmm = cusparse/spmm_data
geo_spmm = gmean(norm_spmm.reshape((-1, 5)), axis=1)
geo_spmm
# [1.425189976994329, 0.9104822139586618, 0.8222391141566255, 0.9594177916537703, 3.356692416173335, 0.9500742873446871, 5.562293435532563]


# 现在开始画SDDMM的数据
# 尴尬，现在还没有SDDMM的完整数据，需要等结果。