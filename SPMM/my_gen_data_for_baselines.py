# 我们在这个文件里生成Magicube需要的数据文件。
# 这个文件是在sparsetir的spmm目录里面运行的

import my_search_formats
from importlib import reload
my_search_formats = reload(my_search_formats)

from my_search_formats import *


def get_pruned_bert_graphNum_and_getOpFunc(name):
	tot_num = None
	get_op = None
	# 
	if name == 'pruned_bert':
		# measure the latency on pruned-bert sparse matrices
		_, tot_num = my_search_formats.test_real_op_pruned_bert('sddmm', float('inf'), feat_size = 32, print_infor=True)
		get_op = my_search_formats.test_real_op_pruned_bert
	elif name == 'pruned_bert_unstructured':
		_, tot_num = my_search_formats.test_real_op_pruned_bert_unstructured('sddmm', float('inf'), feat_size = 32, print_infor=True)
		get_op = my_search_formats.test_real_op_pruned_bert_unstructured
	# 
	return tot_num, get_op




def get_op(op_type, data_i, feat_size, name, m=4096, patch_size = 2, mask_ratio = 0.75):
    op = None
    if name == 'pruned_bert':
        op, _ = test_real_op_pruned_bert(op_type, data_i, feat_size = feat_size, print_infor=False)
    elif name == 'pruned_bert_unstructured':
        op, _ = test_real_op_pruned_bert_unstructured(op_type, data_i, feat_size = feat_size, print_infor=False)
    elif name == 'logsparse':
        op = test_LogSparse(op_type, m, m, feat_size = feat_size)
    elif name == 'strided':
        op = test_Strided(op_type, m, m, feat_size = feat_size)
    elif name == 'random':
        op = test_random_sample(op_type, m, m, feat_size = feat_size, patch_size = patch_size, mask_ratio = mask_ratio)
    elif ('.txt' in name) or ('.csv' in name) or ('.' in name):
        op = load_snap_graph(op_type, name = name, feat_size = feat_size, pad=False)
    else:
        op = test_real_op(op_type, name = name, feat_size = feat_size, pad=False)
    return op





# names = ['citeseer', 'cora', 'ppi', 'pubmed',  'arxiv', 'proteins', 'reddit']
names = ['pruned_bert', 'pruned_bert_unstructured']
names = ['out.web-NotreDame', 'strided', 'logsparse']
names = ['citeseer']

vector_lens = [2**i for i in range(1, 6)]
op_type = 'spmm'

for name in names:
	# tot_num = 1
	# get_op = test_real_op
	# if name in ['pruned_bert', 'pruned_bert_unstructured']:
	# 	tot_num, get_op = get_pruned_bert_graphNum_and_getOpFunc(name)
	# for data_i in range(tot_num):
	# 	op = None
	# 	if name in ['pruned_bert', 'pruned_bert_unstructured']:
	# 		op, _ = get_op(op_type, data_i, feat_size = 32, print_infor=False)
	# 	else:
	# 		op = test_real_op(op_type, name = name, feat_size = 32, pad=False)
	tot_num = 1
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):
		feat_size = 32
		op = get_op(op_type, data_i, feat_size, name, m=4096, patch_size = 2, mask_ratio = 0.75)
	# 
		# op = test_real_op(op_type, name = name, feat_size = 32, pad=False)
		A = op.inps[0]
		for vector_len in vector_lens:
			# name = "citeseer" # "proteins" # "arxiv" "pubmed" "citeseer"
			# vector_len = 2
			m = math.ceil(A.shape[0]/vector_len)
			n = A.shape[1]
			# the number of vectors in each row window
			vector_nums = [np.count_nonzero(A[ i*vector_len:(i+1)*vector_len ].getnnz(axis=0)) for i in range(m)]
			vector_ptr = np.cumsum([0] + vector_nums)
			vector_indices = np.concatenate([np.nonzero(A[ i*vector_len:(i+1)*vector_len ].getnnz(axis=0))[0] for i in range(m)])
			vector_nnz = sum(vector_nums)
			# 
			with open(f'data_for_Magicube/{name}{data_i}_{vector_len}.txt', 'w') as f:
				f.write(f'{m}, {n}, {vector_nnz}\n')
				f.write(' '.join([str(i) for i in vector_ptr])+'\n')
				f.write(' '.join([str(i) for i in vector_indices])+'\n')





for name in names:
	tot_num = 1
	get_op = test_real_op
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, get_op = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):
		for vector_len in vector_lens:
			# with open(f'data_for_Magicube/my_spmm_pruned_bert.txt', 'a') as f:
			# 	f.write(f'data_for_Magicube/{name}{data_i}_{vector_len}.txt\n')
			with open(f'data_for_Magicube/my_spmm_extra_datasets.txt', 'a') as f:
				f.write(f'data_for_Magicube/{name}{data_i}_{vector_len}.txt\n')

