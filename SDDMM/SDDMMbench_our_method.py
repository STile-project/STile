import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
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
		op = load_snap_graph(op_type, name = name, feat_size = feat_size)
	else:
		op = test_real_op(op_type, name = name, feat_size = feat_size)
	return op






def get_mem_usage(selected_tiles):
	# compute the memory usage of the given set of selected_tiles
	assert selected_tiles[0].op.op_type == 'sddmm', 'Only support SDDMM now.'
	tiles_1D = [t for t in selected_tiles if t.op.loop_protocals[0] != 'uuu']
	tiles_TC = [t for t in selected_tiles if t.op.loop_protocals[0] == 'uuu']
	memory_usage = 0
	if len(tiles_1D) > 0:
		print(f"1D tile tile sizes: {tiles_1D[0].best_tile_sizes[0]}")
		memory_usage = memory_usage+len(tiles_1D)*tiles_1D[0].best_tile_sizes[0]*(2*4+2)
	if len(tiles_TC) > 0:
		memory_usage = memory_usage+4*(len(tiles_TC)*(tiles_TC[0].tile_i_rng[1]+1-tiles_TC[0].tile_i_rng[0])+len(tiles_TC)*(tiles_TC[0].tile_k_rng[1]+1-tiles_TC[0].tile_k_rng[0]))
		memory_usage = memory_usage+2*(len(tiles_TC)*tiles_TC[0].best_tile_sizes[0][1]*tiles_TC[0].best_tile_sizes[2][1])
	print(selected_tiles[0].op.inps[1].data, selected_tiles[0].op.inps[1].data.size)
	memory_usage = memory_usage + sum([inp.data.size*2 for inp in selected_tiles[0].op.inps[1:]])
	return memory_usage/(1024*1024)





def get_mem_usage_from_args(args_nd):
    # return how many MB the given args_nd takes
    return sum([arg.numpy().nbytes for arg in args_nd])/(1024**2)


def do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=4096, 
	max_level_num=float('inf'), 
	only_TC = False, only_ELL=False,
	):
	# name = "reddit" # "proteins" # "arxiv" "pubmed" "citeseer"
	op = get_op(op_type, data_i, feat_size, name, m=m)
	# 
	os.environ['op_type'] = op_type
	# op = test_multi_level(1, 1, 512, 128, 0.7, feat_size = 32)
	# data_i = 0
	# op, _ = test_real_op_pruned_bert(op_type, data_i, feat_size = 32)
	# print(op.idx_lens)
	# op = simple_test()

	use_single_level = 'False' if max_level_num > 1 else 'True'
	os.environ['single_level'] = use_single_level # 'False' # 'True'
	TC_k_notsorted = False
	os.environ['TC_k_notsorted'] = 'False'
	os.environ['no_withdraw'] = 'False'
	os.environ['REMAP'] = 'False' # 'True'

	# pattern = "longformer" # "longformer" # "pixelfly"  "pubmed"
	# op = test_real_op_sparse_attention_bsr(pattern, feat_size = 64)
	# TC_k_notsorted = True
	# os.environ['TC_k_notsorted'] = 'True'
	# # 不做condense是因为我们希望k轴的顺序不变，如果condense了，我们tile_pos_k可能出现分数
	# os.environ['single_level'] = 'True' # 我们默认对于block wise structured pruned的sparse pattern，不reorder TC op k轴，不做condense。


	# op = test_real_op_big_bird(feat_size = 64)

	# pad_irregular_idx(op,0,2)
	# greedy_search_use_cost_model(op, max_bucket_size = 256, log_file="log_hub/Approx_arxiv_0104_1.py", cuda_i = 3)

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
	# only_TC, only_ELL = False, False

	# for pure TC format or pure ELL format
	# only_ELL = True
	# max_avg_cost_diff = float('inf')
	if only_ELL or only_TC:
		max_avg_cost_diff = float('inf')

	if bench_cost_model:
		only_TC, only_ELL = False, False
		max_avg_cost_diff = 0.2


	log_file="log_hub/CostModel_pbert_lower_bound_0320_512_1.py"
	# max_bucket_size = estimate_max_bucket_size(op)
	penalty = 1

	# op = test_real_op(name = name, feat_size = 32)
	selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row, level_num = greedy_search_use_cost_model_lower_bound(
		op, dsize, run_params, max_bucket_size, cache_set, 
		kernel_tile_size_options,
		# TC_tile_sizes, 
		only_TC, only_ELL, penalty, TC_k_notsorted,
		reorder_k_by_nnz = reorder_k_by_nnz,
		max_level_num=max_level_num,
		max_avg_cost_diff = max_avg_cost_diff, use_faster_tuner=use_faster_tuner, 
		log_file=log_file, cuda_i = cuda_i, 
		dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)

	# selected_tiles = greedy_search_use_cost_model_lower_bound(op, 16, run_params, max_bucket_size = 256, 
	# 	log_file="log_hub/CostModel_realop_lower_bound_0221_2.py", cuda_i = 3, 
	# 	dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)")

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



	memory_usage = get_mem_usage(selected_tiles)


	# 决定在最后枚举多种parameter setting，总而找到最好的implementation
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
						if only_TC:
							if (tx, ty, vec_1d, group_size) != (8, 4, 4, 1):
								continue
						# For Measure Pure ELL formats
						if only_ELL:
							if (TC_vec != 1): # or (tx*ty > 32):
								continue

						if (not only_TC) and (not only_ELL):
							if (tx*ty > 32):
								continue

						# if ((name == 'reddit') and ((4, 16, 2, 8, 8) == (TC_vec, tx, ty, vec_1d, group_size))) or (name != 'reddit'):
						# 	start_measure=True
						# if not start_measure:
						# 	continue

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
						# my_fuse_formats = reload(my_fuse_formats)
						# cost = my_fuse_formats.measure_seleted_formats(op, 'sddmm', 1, selected_tiles, cuda_i, cache_set, dsize,
						# 	dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
						cost = None
						try:
							res1 = my_fuse_formats.measure_seleted_formats(op, 'sddmm', 1, selected_tiles, cuda_i, cache_set, dsize, 
								gened_inputs = gened_inputs,
								dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
							cost = my_fuse_formats.measure_latency(*res1)

							# assert memory_usage == get_mem_usage_from_args(res1[4]), \
							# 	f"Something wrong with memory usage computation {memory_usage, get_mem_usage_from_args(res1[4])}"

						except Exception as e:
							print(e)
							cost = float('inf')
						
						print(f"measure result: {(name, data_i), ('m', m), feat_size, ('single_level', use_single_level, max_level_num), ('only_TC', only_TC, 'only_ELL', only_ELL), (TC_vec, tx, ty, vec_1d, group_size), cost, end_time - start_time, ('mem', memory_usage), ('level_num', level_num)}", flush=True)

						with open(filename, 'a') as file:
							file.write(json.dumps([(name, data_i), ('m', m), feat_size, ('single_level', use_single_level, max_level_num), ('only_TC', only_TC, 'only_ELL', only_ELL), (TC_vec, tx, ty, vec_1d, group_size), cost, end_time - start_time, ('mem', memory_usage), ('level_num', level_num), best_cost, best_config, '||', TC_row_num, TC_num, num_1D, pred_tot, pred_TC, pred_1D, blk_shapes]))
							file.write('\n')						

						if cost < best_cost:
							best_cost = cost
							best_config = (TC_vec, tx, ty, vec_1d, group_size)


	with open(filename, 'a') as file:
		file.write(json.dumps(["BEST", (name, data_i), ('m', m), feat_size, ('single_level', use_single_level, max_level_num), ('only_TC', only_TC, 'only_ELL', only_ELL), best_config, best_cost]))
		file.write('\n')

	return
	# ==============================================================================================================
	# 以下是之前默认parameter setting的时候的写法。


	for t in selected_tiles:
		if t.op.loop_protocals[0] == 'uuu':
			t.params = {'mma_shape_str': "m16n16k16", 'warp_num': 1, 'vec_size': 8}
			t.best_params = {'mma_shape_str': "m16n16k16", 'warp_num': 1, 'vec_size': 8}
		else:
			t.params = {'tx': 8, 'ty': 4, 'vec_size': 4, 'group_size': 8, 'max_bucket_size': 32}
			t.best_params = {'tx': 8, 'ty': 4, 'vec_size': 4, 'group_size': 8, 'max_bucket_size': 32}



	# my_fuse_formats = reload(my_fuse_formats)
	# cost = my_fuse_formats.measure_seleted_formats(op, 'sddmm', 1, selected_tiles, cuda_i, cache_set, dsize,
	# 	dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
	res1 = my_fuse_formats.measure_seleted_formats(op, 'sddmm', 1, selected_tiles, cuda_i, cache_set, dsize,
		dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
	cost = my_fuse_formats.measure_latency(*res1)

	print(f"measure result: {name, feat_size, cost, end_time - start_time}", flush=True)

	with open(filename, 'a') as file:
		file.write(json.dumps([name, feat_size, cost, end_time - start_time, TC_row_num, TC_num, num_1D, pred_tot, pred_TC, pred_1D, blk_shapes]))
		file.write('\n')








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
# 	file.write("Change TC tile candidate tile_sizes; Change 1D tile cost model coefficients; Make the measure faster by reusing inputs\n")
# 	file.write("Pure ELL formats; allow more than 32 threads for 1D tiles\n")



# for name in names:
# 	tot_num = 1
# 	if name in ['pruned_bert', 'pruned_bert_unstructured']:
# 		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
# 	for data_i in range(tot_num):
# 		for feat_size in feat_sizes:
# 			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
# 				continue
# 			do_profile(op_type, name, feat_size, filename, data_i)




# ---------------------------------------------------------------------------------------------------------------------------
# names = ['pruned_bert', 'pruned_bert_unstructured'] + ['cora', 'citeseer', 'ppi', 'pubmed', 'arxiv', 'proteins', 'reddit']
# names = ['pruned_bert', 'pruned_bert_unstructured'] + ['cora', 'citeseer', 'ppi', 'pubmed']
# names = ['arxiv', 'proteins', 'reddit']

# feat_sizes = [32, 512]
# # names = ['arxiv']


# filename = "cost_model_PERFORMANCE_fp16_part2.json"
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for name in names:
# 	tot_num = 1
# 	for data_i in range(tot_num):
# 		for feat_size in feat_sizes:
# 			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
# 				continue
# 			if (name not in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 32):
# 				continue
# 			with open(filename, 'a') as file:
# 				file.write(f"{op_type, name, feat_size, data_i}\n")
# 			# 
# 			do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=True)

# ---------------------------------------------------------------------------------------------------------------------------

# experiments for comparing multi-level and single-level setting on real datasets

# names = ['logsparse', 'strided']
# feat_sizes = [32, 64, 128, 256, 512]

# filename = "case_study/multi_level_vs_single_level_real_datasets_fp16.json" # run in case_study folder
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for name in names:
# 	for m in [512, 1024, 2048, 4096][::-1]:
# 		tot_num = 1
# 		for data_i in range(tot_num):
# 			for feat_size in feat_sizes:
# 				for use_single_level in ['False', 'True']:
# 					with open(filename, 'a') as file:
# 						file.write(f"{op_type, name, m, feat_size, data_i, use_single_level}\n")
# 					# 
# 					do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, use_single_level=use_single_level)



# ---------------------------------------------------------------------------------------------------------------------------

# experiments for studying influence of the depth and width of search space on the final optimization effect.


# names = ['logsparse', 'strided']
# feat_sizes = [32, 64, 128, 256, 512][::-1]

# filename = "search_space_depth_influence_real_datasets_fp16.json" # run in case_study folder
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for m in [4096][::-1]: #[512, 1024, 2048, 4096][::-1]:
# 	tot_num = 1
# 	for data_i in range(tot_num):
# 		for name in names:
# 			for feat_size in feat_sizes:
# 				for max_level_num in list(range(2, 10)):
# 					with open(filename, 'a') as file:
# 						file.write(f"{op_type, name, m, feat_size, data_i, ('max_level_num', max_level_num), ('only_TC', False), ('only_ELL', False)}\n")
# 					# 
# 					do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, max_level_num=max_level_num, only_TC = False, only_ELL=False)
# 				# for only_TC, only_ELL in [(True, False), (False, True)]:
# 				# 	with open(filename, 'a') as file:
# 				# 		file.write(f"{op_type, name, m, feat_size, data_i, ('max_level_num', float('inf')), ('only_TC', only_TC), ('only_ELL', only_ELL)}\n")
# 				# 	# 
# 				# 	do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, max_level_num=float('inf'), only_TC = only_TC, only_ELL=only_ELL)


# ---------------------------------------------------------------------------------------------------------------------------

# experiments for studying influence of the depth and width of search space on the final optimization effect.


names = ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer'][::-1] + ['pruned_bert', 'pruned_bert_unstructured'][::-1]
names = names + ['Amazon0302.txt']

names = ['reddit', 'out.web-NotreDame', 'logsparse', 'strided']
names = ['reddit']
names = ['logsparse', 'strided']

feat_sizes = [32, 64, 128, 256, 512][::-1]
feat_sizes = [512]

filename = "Var_Depth_real_datasets_fp16.json" # run in case_study folder
with open(filename, 'a') as file:
	file.write(f"\n\n\n\nNew Round---------\n")


m = 4096
for name in names:
	tot_num = 1
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):
		for feat_size in feat_sizes:
			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
				continue
			for max_level_num in [float('inf'), 1, 2, 10, 100, 1000]: # [2, 10, 100, 1000]:
				# if (name in ['reddit', 'out.web-NotreDame']) and (max_level_num in [2, 10, 100, 1000]):
				# 	continue

				with open(filename, 'a') as file:
					file.write(f"{op_type, name, m, feat_size, data_i, ('max_level_num', max_level_num), ('only_TC', False), ('only_ELL', False)}\n")
				# 
				# do_profile(op_type, name, feat_size, filename, data_i)
				do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, max_level_num=max_level_num, only_TC = False, only_ELL=False)



