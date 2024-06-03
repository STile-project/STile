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






def profile_with_tensorboard(f, args_nd, name, feat_size):
	with torch.profiler.profile(
	        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
	        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{name}_{feat_size}'),
	        record_shapes=True,
	        profile_memory=True,
	        with_stack=True
	) as prof:
		f(*args_nd)
		prof.step()
	    # for step, batch_data in enumerate(train_loader):
	    #     if step >= (1 + 1 + 3) * 2:
	    #         break
	    #     train(batch_data)
	    #     prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.








def get_memory_usage(args_nd):
	# return how many MB the given args_nd takes
	return sum([arg.numpy().nbytes for arg in args_nd])/(1024**2)




def do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=4096, patch_size = 2, mask_ratio = 0.75, 
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
	os.environ['REMAP'] = 'False' # 'True' # 'REMAP' is only used for SDDMM operators


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
	# NOTE: for spmm, only_TC cannot use inf max_avg_cost_diff
	if only_ELL:
		max_avg_cost_diff = float('inf')

	if bench_cost_model:
		only_TC, only_ELL = False, False
		max_avg_cost_diff = 0.2


	log_file="log_hub/CostModel_pbert_lower_bound_0320_512_1.py"
	max_bucket_size = estimate_max_bucket_size(op)
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



	# my_branch_and_bound = reload(my_branch_and_bound)

	old_selected_tiles_0 = selected_tiles
	selected_tiles = my_branch_and_bound.post_process_for_withdraw(op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)

	if bench_cost_model:
		measure_for_cost_model(selected_tiles, filename)
		return


	TC_row_num = len(set([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu']))
	TC_num = len(([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu']))
	print(TC_row_num)
	ELL_num = len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] != 'uuu'])
	print(ELL_num)
	tot_sum = sum([t.pred_cost for t in selected_tiles])
	print(tot_sum)
	atomic_TC_num = len([t.tile_pos[0] for t in selected_tiles if t.op.loop_protocals[0] == 'uuu' and t.is_atomic_tile])
	print(atomic_TC_num)
	TC_sum = sum([t.pred_cost for t in selected_tiles if t.op.loop_protocals[0] == 'uuu'])
	print(TC_sum)
	ELL_sum = sum([t.pred_cost for t in selected_tiles if t.op.loop_protocals[0] != 'uuu'])
	print(ELL_sum)


	end_time = time.time()
	search_time = end_time - start_time
	print(f"total search time: {search_time}", flush=True)


	results = list()
	for tmp_feat_size in [32, 64, 128, 256, 512]:

		# stores the memory costs of the selected hybrid format
		memory_costs = list()

		# For Measure Pure TC formats
		if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (tmp_feat_size!=512):
			continue


		# 直接把找到的结果用在不同feature size的数据上。
		# op = test_real_op(op_type, name = name, feat_size = tmp_feat_size)
		op = get_op(op_type, data_i, tmp_feat_size, name, m=m)
		for t in selected_tiles:
			t.op.idx_lens[1] = tmp_feat_size


		# 需要把线程量还原成32个
		for i in selected_tiles:
			if len(i.best_tile_sizes[0]) == 3:
				i.best_tile_sizes = ([None, i.tile_sizes[0][1]//1, 1], i.best_tile_sizes[1], i.best_tile_sizes[2])					

		print("tmp_feat_size: ", tmp_feat_size)
		res1 = my_fuse_formats.measure_seleted_formats(op, 'spmm', 1, selected_tiles, cuda_i, cache_set, dsize, gened_inputs=list(),
			dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)

		cost1 = my_fuse_formats.measure_latency(*res1)

		# append memory information
		memory_costs.append(get_memory_usage(res1[4])) # res1[4] is args_nd

		print(f"cost 32 thread: {cost1}")

		cost2 = float('inf')
		if True: #ELL_num > 0:
			for i in selected_tiles:
				if len(i.best_tile_sizes[0]) == 3:
					i.best_tile_sizes = ([None, i.tile_sizes[0][1]//8, 8], i.best_tile_sizes[1], i.best_tile_sizes[2])

			res2 = my_fuse_formats.measure_seleted_formats(op, 'spmm', 1, selected_tiles, cuda_i, cache_set, dsize, gened_inputs=list(),
				dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)

			cost2 = my_fuse_formats.measure_latency(*res2)

			# append memory information
			memory_costs.append(get_memory_usage(res2[4])) # res1[4] is args_nd

		print(f"cost 256 thread: {cost2}")


		results.append( ((name, data_i), ('m', m), tmp_feat_size, max_avg_cost_diff, 
			('single_level', use_single_level, max_level_num), ('only_TC', only_TC, 'only_ELL', only_ELL),
			os.environ['single_level'], os.environ['TC_k_notsorted'], os.environ['no_withdraw'],
			TC_row_num, TC_num, ELL_num, 
			tot_sum, atomic_TC_num, TC_sum, ELL_sum, 
			'||', search_time, cost1, cost2, ('mem', memory_costs), ('level_num', level_num)) )


		print("summary:", results[-1], flush=True )

		# with open('cost_ablation_localSearch.csv', 'a') as f:
		# with open('cost_added_dataset.csv', 'a') as f:
		# with open('cost_pureTC_dataset.csv', 'a') as f:
		# with open('cost_pureELL_dataset.csv', 'a') as f:
		# with open('cost_dataset_updateTCELLModel.csv', 'a') as f:
		with open(filename, 'a') as f:
			json.dump(results[-1], f )
			f.write('\n')

	return  # ----------------------------------------------------------------------------------------------------------------






# reddit
op_type = 'spmm'


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
# # feat_sizes = [32, 64, 128, 256, 512]
# # for spmm, the hyrid format for different feat_sizes will be the same
# feat_sizes = [32]

# filename = "multi_level_vs_single_level_real_datasets_fp16.json" # run in case_study folder
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for name in names:
# 	for m in [512, 1024, 2048, 4096][::-1]:
# 		tot_num = 1
# 		for data_i in range(tot_num):
# 			for feat_size in feat_sizes:
# 				for max_level_num in [1, float('inf')]:
# 					with open(filename, 'a') as file:
# 						file.write(f"{op_type, name, m, feat_size, data_i, max_level_num}\n")
# 					# 
# 					do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, max_level_num=max_level_num)


# names = ['random']
# # for spmm, the hyrid format for different feat_sizes will be the same
# feat_sizes = [32]

# filename = "multi_level_vs_single_level_real_datasets_fp16.json" # run in case_study folder
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for name in names:
# 	for m in [224, 256, 512, 1024, 2048, 4096]:
# 		tot_num = 1
# 		for data_i in range(tot_num):
# 			for patch_size in [1, 2, 14, 16]:
# 				if m%patch_size!=0:
# 					continue
# 				for mask_ratio in [0.75, 0.5]:
# 					for feat_size in feat_sizes:
# 						for max_level_num in [1, float('inf')]:
# 							with open(filename, 'a') as file:
# 								file.write(f"{op_type, name, m, feat_size, patch_size, mask_ratio, data_i, max_level_num}\n")
# 							# 
# 							do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, 
# 								patch_size = patch_size, mask_ratio = mask_ratio, max_level_num=max_level_num)





# names = ['pruned_bert', 'pruned_bert_unstructured'] + ['cora', 'citeseer', 'ppi', 'pubmed', 'arxiv', 'proteins', 'reddit']

# feat_sizes = [32, 512]
# # names = ['arxiv']


# filename = "multi_level_vs_single_level_main_result_datasets_fp16_2.json"
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------[ Reduce TC tile K size options to [16, 16*5] ]\n")


# for name in names:
# 	tot_num = 1
# 	if name in ['pruned_bert', 'pruned_bert_unstructured']:
# 		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
# 	for data_i in range(tot_num):
# 		for feat_size in feat_sizes:
# 			for max_level_num in [1, float('inf')]:
# 				if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
# 					continue
# 				if (name not in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 32):
# 					continue
# 				with open(filename, 'a') as file:
# 					file.write(f"{op_type, name, feat_size, data_i, max_level_num}\n")
# 				# 
# 				do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, 
# 					max_level_num=max_level_num)






# # names = ['wikics', 'cobuy']
# # names = ['bgs']
# names = ['collab']
# names = ['twitter_combined.txt']
# names = ['large_twitch_edges.csv']
# names = ['web-Stanford.txt', 'Amazon0302.txt']
# names = ['Amazon0302.txt']
# names = ['ppi']
# names = ['reddit']
# names = ['out.web-NotreDame']


# feat_sizes = [32]


# filename = "VarDepth_fp16.json"
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for name in names:
# 	tot_num = 1
# 	for data_i in range(tot_num):
# 		for feat_size in feat_sizes:
# 			for max_level_num in [2, 10, 100, 1000]:
# 				with open(filename, 'a') as file:
# 					file.write(f"{op_type, name, feat_size, data_i, max_level_num}\n")
# 				# 
# 				do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, 
# 					max_level_num=max_level_num, 
# 					only_TC = False, only_ELL=False)
# 			# do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False,  
# 			# 	only_TC = True, only_ELL=False)
# 			# do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, 
# 			# 	only_TC = False, only_ELL=True)



# ---------------------------------------------------------------------------------------------------------------------------
# experiments for profiling the memory usage of our method

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser("hybrid format spmm in STile")
# 	parser.add_argument("--name", "-n", type=str, default="arxiv", help="dataset name")
# 	parser.add_argument("--datai", "-i", type=int, default=0, help="dataset index")
# 	parser.add_argument("--featsize", "-f", type=int, default=32, help="feature size")
# 	args = parser.parse_args()
# 	name = args.name
# 	data_i = args.datai
# 	feat_size = args.featsize
# 	filename = "Mem_fp16.json"
# 	do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, 
# 		max_level_num=float('inf'), 
# 		only_TC = False, only_ELL=False)

names = ['cora', 'citeseer', 'ppi', 'pubmed', 'arxiv', 'proteins', 'reddit', 'out.web-NotreDame'] + ['pruned_bert', 'pruned_bert_unstructured']
names = ['pruned_bert_unstructured']
names = ['pruned_bert_unstructured', 'logsparse', 'strided']

feat_sizes = [32, 512]

filename = "Mem_LevelNum_fp16.json"
with open(filename, 'a') as file:
	file.write(f"\n\n\n\nNew Round---------\n")


for name in names:
	tot_num = 1
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):
		for feat_size in feat_sizes:
			if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
				continue
			if (name not in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 32):
				continue
			# if data_i < 6:
			# 	continue
			with open(filename, 'a') as file:
				file.write(f"{op_type, name, feat_size, data_i}\n")
			# 
			do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False)
			# do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, only_TC = True, only_ELL=False)
			# do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, only_TC = False, only_ELL=True)



# ---------------------------------------------------------------------------------------------------------------------------

# experiments for studying influence of the depth and width of search space on the final optimization effect.


# names = ['logsparse', 'strided']
# feat_sizes = [32, 64, 128, 256, 512][::-1]

# filename = "case_study/search_space_influence_real_datasets_fp16.json" # run in case_study folder
# with open(filename, 'a') as file:
# 	file.write(f"\n\n\n\nNew Round---------\n")


# for m in [512, 1024, 2048, 4096][::-1]:
# 	tot_num = 1
# 	for data_i in range(tot_num):
# 		for name in names:
# 			for feat_size in feat_sizes:
# 				for max_level_num in [1, 10, 100, 1000, float('inf')]:
# 					with open(filename, 'a') as file:
# 						file.write(f"{op_type, name, m, feat_size, data_i, ('max_level_num', max_level_num), ('only_TC', False), ('only_ELL', False)}\n")
# 					# 
# 					do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, max_level_num=max_level_num, only_TC = False, only_ELL=False)
# 				for only_TC, only_ELL in [(True, False), (False, True)]:
# 					with open(filename, 'a') as file:
# 						file.write(f"{op_type, name, m, feat_size, data_i, ('max_level_num', float('inf')), ('only_TC', only_TC), ('only_ELL', only_ELL)}\n")
# 					# 
# 					do_profile(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=m, max_level_num=float('inf'), only_TC = only_TC, only_ELL=only_ELL)


