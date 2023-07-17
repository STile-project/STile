import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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




def do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False):
	# name = "reddit" # "proteins" # "arxiv" "pubmed" "citeseer"
	op = None
	if name == 'pruned_bert':
		op, _ = test_real_op_pruned_bert(op_type, data_i, feat_size = feat_size, print_infor=False)
	elif name == 'pruned_bert_unstructured':
		op, _ = test_real_op_pruned_bert_unstructured(op_type, data_i, feat_size = feat_size, print_infor=False)
	else:
		op = test_real_op(op_type, name = name, feat_size = feat_size)
	# 
	os.environ['op_type'] = 'sddmm'


	os.environ['single_level'] = 'False' # 'True'
	TC_k_notsorted = False
	os.environ['TC_k_notsorted'] = 'False'
	os.environ['no_withdraw'] = 'False'
	os.environ['REMAP'] = 'False' # 'True'


	import time
	start_time = time.time()
	run_params = {"summarize_nnz_iter_space":(True, True)} # use_file, do_parallel
	cuda_i = 0
	dtype = "float16"
	dtype_str = '"float16"'
	zerotype = "T.float16(0)"
	# dtype = "float32"
	# dtype_str = '"float32"'
	# zerotype = "T.float32(0)"
	# bench_cusparse(op.inps[0], op.inps[1].data, dtype, cuda_i)
	# bench_cublas(op.inps[0], op.inps[1].data, dtype, cuda_i)
	# bench_dgl(op.inps[0], op.inps[1].data, dtype, cuda_i)


	max_bucket_size = 32 # 256 # 32
	use_faster_tuner = True
	cache_set = ['A'] # [] # ['A']
	dsize = 16 # 32 # 16
	kernel_tile_size_options = ([1], [1], [1]) # ([1], [1], [8]) # ([1], [1], [1])
	TC_tile_sizes = (32, 32, 768) # (16, 16, 16)
	max_avg_cost_diff = 0.2 # 0.1 # float('inf') 0.2 
	reorder_k_by_nnz = True
	only_TC, only_ELL = False, False

	# for pure TC format or pure ELL format
	only_ELL = True
	max_avg_cost_diff = float('inf')

	if bench_cost_model:
		only_TC, only_ELL = False, False
		max_avg_cost_diff = 0.2


	log_file="log_hub/CostModel_pbert_lower_bound_0320_512_1.py"
	# max_bucket_size = estimate_max_bucket_size(op)
	penalty = 1

	# op = test_real_op(name = name, feat_size = 32)
	selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row = greedy_search_use_cost_model_lower_bound(
		op, dsize, run_params, max_bucket_size, cache_set, 
		kernel_tile_size_options,
		# TC_tile_sizes, 
		only_TC, only_ELL, penalty, TC_k_notsorted,
		reorder_k_by_nnz = reorder_k_by_nnz,
		max_avg_cost_diff = max_avg_cost_diff, use_faster_tuner=use_faster_tuner, 
		log_file=log_file, cuda_i = cuda_i, 
		dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)



	end_time = time.time()
	print("search time: ", end_time - start_time, flush=True)


	# my_branch_and_bound = reload(my_branch_and_bound)

	old_selected_tiles_0 = selected_tiles
	selected_tiles = my_branch_and_bound.post_process_for_withdraw(op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)

	print("TC blk row num: ", len(set([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu'])))
	print("TC blk num: ", len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu']))
	print("1D blk num: ", len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] != 'uuu']))
	print("tot cost: ", sum([t.pred_cost for t in selected_tiles]))
	# len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu' and t.is_atomic_tile])
	print("TC cost: ", sum([t.pred_cost for t in selected_tiles if t.op.loop_protocals[0] == 'uuu']))
	print("1D cost: ", sum([t.pred_cost for t in selected_tiles if t.op.loop_protocals[0] != 'uuu']))
	# print("blk shapes: ", sorted(set([t.tile_sizes for t in selected_tiles if len(t.tile_sizes) > 1])), set([t.tile_sizes for t in selected_tiles if len(t.tile_sizes) == 1])
	# )
	print("blk shapes: ", np.unique([t.tile_sizes[2][1] for t in selected_tiles if len(t.tile_sizes) > 1], return_counts=True), set([t.tile_sizes for t in selected_tiles if len(t.tile_sizes) == 1])
	)


	if bench_cost_model:
		measure_for_cost_model(selected_tiles, filename)
		return


	TC_row_num = len(set([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu']))
	TC_num = len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu'])
	num_1D = len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] != 'uuu'])
	pred_tot = sum([t.pred_cost for t in selected_tiles])
	pred_TC = sum([t.pred_cost for t in selected_tiles if t.op.loop_protocals[0] == 'uuu'])
	pred_1D = sum([t.pred_cost for t in selected_tiles if t.op.loop_protocals[0] != 'uuu'])
	blk_shapes = [sorted(set([t.tile_sizes for t in selected_tiles if len(t.tile_sizes) > 1])), list(set([t.tile_sizes for t in selected_tiles if len(t.tile_sizes) == 1]))]





	best_cost = float('inf')
	best_config = None
	gened_inputs = list()
	# start_measure = False
	for TC_vec in [1, 2, 4, 8]:
		for tx in [8, 16, 32]:
			for ty in [32//tx, 64//tx, 128//tx, 256//tx]: # allow more than 32 threads for only_ELL formats
				for vec_1d in [4, 8]:
					for group_size in [1, 2, 4, 8]:

						# For Measure Pure TC formats
						# if (tx, ty, vec_1d, group_size) != (8, 4, 4, 1):
						# 	continue
						# For Measure Pure ELL formats
						if TC_vec != 1:
							continue



						if tx*vec_1d > feat_size:
							continue
						for t in selected_tiles:
							if t.op.loop_protocals[0] == 'uuu':
								t.params = {'mma_shape_str': "m16n16k16", 'warp_num': 1, 'vec_size': TC_vec}
								t.best_params = t.params
							else:
								t.params = {'tx': tx, 'ty': ty, 'vec_size': vec_1d, 'group_size': group_size, 'max_bucket_size': 32}
								t.best_params = t.params
						# 

						cost = None
						try:
							res1 = my_fuse_formats.measure_seleted_formats(op, 'sddmm', 1, selected_tiles, cuda_i, cache_set, dsize, 
								gened_inputs = gened_inputs,
								dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
							cost = my_fuse_formats.measure_latency(*res1)
						except Exception as e:
							print(e)
							cost = float('inf')
						
						print(f"measure result: {(name, data_i), feat_size, (TC_vec, tx, ty, vec_1d, group_size), cost, end_time - start_time}", flush=True)

						with open(filename, 'a') as file:
							file.write(json.dumps([(name, data_i), feat_size, (TC_vec, tx, ty, vec_1d, group_size), cost, end_time - start_time, best_cost, best_config, '||', TC_row_num, TC_num, num_1D, pred_tot, pred_TC, pred_1D, blk_shapes]))
							file.write('\n')						

						if cost < best_cost:
							best_cost = cost
							best_config = (TC_vec, tx, ty, vec_1d, group_size)


	with open(filename, 'a') as file:
		file.write(json.dumps(["BEST", (name, data_i), feat_size, best_config, best_cost]))
		file.write('\n')

	return






# reddit
op_type = 'sddmm'


# # names = ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer'][::-1]
# names = ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit'][::-1]
# feat_sizes = [32, 64, 128, 256, 512]


# names = ['pruned_bert', 'pruned_bert_unstructured']
# feat_sizes = [512]

# names = ['pruned_bert', 'pruned_bert_unstructured'][::-1] + ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer'][::-1]
# feat_sizes = [32, 64, 128, 256, 512]




# filename = 'my_sddmm_fp16.json'
# with open(filename, 'a') as file:
# 	file.write("New Round---------\n")




# for name in names:
# 	tot_num = 1
# 	if name in ['pruned_bert', 'pruned_bert_unstructured']:
# 		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
# 	for data_i in range(tot_num):
# 		for feat_size in feat_sizes:
# 			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
# 				continue
# 			do_profile(op_type, name, feat_size, filename, data_i)




names = ['pruned_bert', 'pruned_bert_unstructured'] + ['cora', 'citeseer', 'ppi', 'pubmed', 'arxiv', 'proteins', 'reddit']
names = ['pruned_bert', 'pruned_bert_unstructured'] + ['cora', 'citeseer', 'ppi', 'pubmed']
names = ['arxiv', 'proteins', 'reddit']

feat_sizes = [32, 512]
# names = ['arxiv']


filename = "cost_model_PERFORMANCE_fp16_part2.json"
with open(filename, 'a') as file:
	file.write(f"\n\n\n\nNew Round---------\n")


for name in names:
	tot_num = 1
	for data_i in range(tot_num):
		for feat_size in feat_sizes:
			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
				continue
			if (name not in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 32):
				continue
			with open(filename, 'a') as file:
				file.write(f"{op_type, name, feat_size, data_i}\n")
			# 
			do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=True)

