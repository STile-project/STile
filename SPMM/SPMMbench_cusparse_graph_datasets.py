import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['single_level'] = 'False'


import my_search_formats
from importlib import reload
my_search_formats = reload(my_search_formats)

from my_search_formats import *




def get_pruned_bert_graphNum_and_getOpFunc(name):
	tot_num = None
	get_op = None

	if name == 'pruned_bert':
		# measure the latency on pruned-bert sparse matrices
		_, tot_num = test_real_op_pruned_bert('sddmm', float('inf'), feat_size = 32, print_infor=True)
		get_op = test_real_op_pruned_bert
	elif name == 'pruned_bert_unstructured':
		_, tot_num = test_real_op_pruned_bert_unstructured('sddmm', float('inf'), feat_size = 32, print_infor=True)
		get_op = test_real_op_pruned_bert_unstructured

	return tot_num, get_op




def prepare_fake_graph(idx, indptr, indices):
	# idx is the index of the 1D tile
	# first get the row ids of this tile
	row1 = np.searchsorted(indptr, idx*32, side='right') - 1
	row2 = min(np.searchsorted(indptr, (idx+1)*32-1, side='right'), len(indptr)-1) # exclusive
	# get the nnzs per row
	# nnzs = np.asarray([ min(indptr[i+1], (idx+1)*32) - indptr[i] for i in range(row1, row2)], dtype='int32')
	# 
	nnzs = indptr[row1:row2+1]
	nnzs[-1] = min(nnzs[-1], (idx+1)*32)
	nnzs[0] = max(nnzs[0], idx*32)
	nnzs = np.diff(nnzs)
	print(row1, row2, nnzs, sum(nnzs))
	# 
	tmp_indices = indices[ idx*32 : (idx+1)*32 ]
	# tmp_indices的数据有问题，我们也应该对其平移，但是首先应该对其压缩
	_, tmp_indices = np.unique(tmp_indices, return_inverse=True)
	# 
	fake_indptr = np.tile(nnzs, 108*100)
	fake_indptr = np.concatenate([[0], np.cumsum(fake_indptr)])
	# 
	node_num = max(len(tmp_indices)*108*100, len(fake_indptr)-1)
	fake_indptr = np.concatenate([fake_indptr, [fake_indptr[-1] for i in range( node_num+1-len(fake_indptr) )]])
	# fake_indices = np.tile(tmp_indices, 108*100)
	step = len(tmp_indices)
	fake_indices = np.concatenate([tmp_indices+step*i for i in range(108*100)])
	# print(fake_indptr, len(fake_indptr), fake_indices, len(fake_indices))
	assert fake_indptr[-1] == len(fake_indices)
	g = dgl.graph(('csr', (fake_indptr, fake_indices, [])), idtype=torch.int32, num_nodes=node_num)
	return g



def measure_for_cost_model(selected_tiles, filename):
	'''
	Sample from the selected tiles and measure the time costs to validate the cost model performance.
	'''
	import bench_sddmm_for_test
	import sys

	del os.environ['MyFileID']
	dsize = 16

	tiles_1d = [t for t in selected_tiles if get_template_str(t.op) == '1D_sddmm']
	if len(tiles_1d) == 0:
		return
	# sample from tiles_1d according to their pred_cost
	pred_costs = [t.pred_cost for t in tiles_1d]
	uni_pred_costs, counts = np.unique(pred_costs, return_counts=True)
	# 从selected tiles中sample 400个 1D tile
	_, indices = np.unique(pred_costs, return_inverse = True)
	weights = counts[indices]
	# weights = np.array([i for i in weights])
	print(weights[:10], len(weights))
	weights = weights/sum(weights)
	sampled = np.random.choice(np.arange(len(tiles_1d)), size=min(400, len(tiles_1d)), replace=False, p=weights)

	for tile_i in sampled:
		t = tiles_1d[tile_i]
		row_num = len(set(t.op.ori_row_ids_1d[ t.tile_pos[0]*t.tile_sizes[0] : (t.tile_pos[0]+1)*t.tile_sizes[0] ]))
		col_num = len(set(t.op.ori_col_ids_1d[ t.tile_pos[0]*t.tile_sizes[0] : (t.tile_pos[0]+1)*t.tile_sizes[0] ]))
		g = prepare_fake_graph(t.tile_pos[0], t.op.position_space[0].indptr, t.op.position_space[0].indices)
		for feat_size in [32, 64, 128, 256, 512]:
			# if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size!=512):
			# 	continue
			try:
				pred_c = my_cost_model._cost_tb_latency_1D_tiles_given_features(row_num, col_num, feat_size, t.tile_sizes, dsize)
				print(f"pred costs: {pred_c, t.pred_cost}", flush=True)
				bench_sddmm_for_test.bench_sddmm(filename, g, feat_size, t.tile_pos[0], pred_c, given_params=None)
					# (t.params['tx'], t.params['ty'], t.params['vec_size'], t.params['group_size']))
			except Exception as e:
				print("OOM")
				print(e, file=sys.stderr)



# NOTE: should not pad here
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












def get_memory_usage(args_nd):
	# return how many MB the given args_nd takes
	return sum([arg.numpy().nbytes for arg in args_nd])/(1024**2)




def do_profile(op_type, filename, name, data_i, feat_size, m=4096):
	cuda_i = 0
	dtype = "float16"
	op = get_op(op_type, data_i, feat_size, name, m=m)
	latency, memory_usage = bench_cusparse(op.inps[0], op.inps[1].data, dtype, cuda_i)
	with open(filename, 'a') as file:
		json.dump(['cusparse', op_type, (name, data_i), feat_size, m, ('time', latency), ('mem', memory_usage)], file)
		file.write('\n')






# reddit
op_type = 'spmm'
names = ['cora', 'citeseer', 'ppi', 'pubmed', 'arxiv', 'proteins', 'reddit', 'out.web-NotreDame'] + ['pruned_bert', 'pruned_bert_unstructured']

names = ['out.web-NotreDame', 'strided', 'logsparse']
names = ['cora', 'citeseer', 'ppi', 'pubmed', 'arxiv', 'proteins', 'reddit', 'out.web-NotreDame']

feat_sizes = [32, 64, 128, 256, 512]

filename = "Mem_CuSparse_graphs_fp16.json"
with open(filename, 'a') as file:
	file.write(f"\n\n\n\nNew Round---------\n")

m=4096
for name in names:
	tot_num = 1
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):
		for feat_size in feat_sizes:
			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
				continue
			# if (name not in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 32):
			# 	continue
			# 
			do_profile(op_type, filename, name, data_i, feat_size, m=m)



