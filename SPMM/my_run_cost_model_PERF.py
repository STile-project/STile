import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


import my_search_formats
from importlib import reload
my_search_formats = reload(my_search_formats)

from my_search_formats import *




def get_pruned_bert_graphNum_and_getOpFunc(name):
	tot_num = None
	get_op = None

	if name == 'pruned_bert':
		# measure the latency on pruned-bert sparse matrices
		_, tot_num = test_real_op_pruned_bert('spmm', float('inf'), feat_size = 32, print_infor=True)
		get_op = test_real_op_pruned_bert
	elif name == 'pruned_bert_unstructured':
		_, tot_num = test_real_op_pruned_bert_unstructured('spmm', float('inf'), feat_size = 32, print_infor=True)
		get_op = test_real_op_pruned_bert_unstructured

	return tot_num, get_op





def measure_for_cost_model(selected_tiles, filename, dsize, name, only_ELL):

	with open(filename, 'a') as f:
		f.write(f'\n\n\nNext Measure Set  {name}, {only_ELL}\n\n')


	SM_num = 108
	costs = list()
	a = [t for t in selected_tiles if t.op.loop_protocals[0] != 'uuu']
	print("num of ELL tiles:", len(a))

	if (len(a) > 0) and only_ELL:
		pred_costs = [t.pred_cost for t in a]
		uni_pred_costs, counts = np.unique(pred_costs, return_counts=True)

		_, indices = np.unique(pred_costs, return_inverse = True)
		weights = counts[indices]

		print(weights[:10], len(weights))
		weights = weights/sum(weights)
		sampled = np.random.choice(np.arange(len(a)), size=min(1000, len(a)), replace=False, p=weights)
		a = [a[i] for i in sampled]

	for selected_tile in a:
		set_atomic = selected_tile.is_atomic_tile
		for set_atomic in [True, False]:
		# selected_tile = tiles[0]
			c = my_fuse_formats.measure_seleted_tile_more_accurate(selected_tile.op, selected_tile, cuda_i, cache_set, dsize, set_atomic,
				dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
			costs.append((name, selected_tile.tile_sizes, selected_tile.pred_cost, int(selected_tile.nnz_when_selected), 
				my_cost_model.cost_tb_latency_given_ELL_tiles(
					selected_tile.op, selected_tile.tile_sizes, np.array([selected_tile.tile_pos[0]]), dsize, selected_tile.op.inps[0].nnz, SM_num, set_atomic)[0],
				# my_cost_model.compute_cost_tb_impl_given_tile(selected_tile),
				# my_cost_model.memory_cost_tb_given_tile(selected_tile, dsize)[set_atomic],
				c, set_atomic))
			print(len(costs), " : ", costs[-1])

			with open(filename, 'a') as f:
				# (((None, 128), (None, 32), (1, 2)), 13696.000000000002, 8192, 16384, 214, 0.19583951000000002, False)
				json.dump(costs[-1] + (selected_tile.is_atomic_tile, ), f )
				f.write('\n')








# names = ['citeseer', 'cora', 'ppi', 'pubmed',  'arxiv', 'proteins', 'reddit']
# names = ['citeseer', 'amazon', 'coauthor', 'GDELT']

names = ['citeseer', 'cora', 'ppi', 'proteins', 'reddit', 'pubmed',  'arxiv'] # + ['pruned_bert', 'pruned_bert_unstructured']
# names = ['pubmed',  'arxiv'] # + ['pruned_bert', 'pruned_bert_unstructured']
names = ['pubmed']

results = list()

op_type = 'spmm'

for name in names:
	tot_num = 1
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):		
		for feat_size in [32]: #[32, 64, 128, 256, 512]:
			for max_avg_cost_diff in [0.2]: #[float('inf')]: # [0.2]: #[0, 0.2, 0.4, 0.6]:

				os.environ['op_type'] = 'spmm'

				os.environ['single_level'] = 'False' # 'True'
				TC_k_notsorted = False
				os.environ['TC_k_notsorted'] = 'False'
				os.environ['no_withdraw'] = 'False'


				print(f"name: {(name, data_i)}  feat_size: {feat_size}")

				# reddit
				# name = "arxiv" # "proteins" # "arxiv" "pubmed" "citeseer"
				op = test_real_op(op_type, name = name, feat_size = feat_size)

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

				max_bucket_size = 32 # 256 # 32
				use_faster_tuner = True
				cache_set = ['A'] # [] # ['A']
				dsize = 16 # 32 # 16
				kernel_tile_size_options = ([1], [1], [1]) # ([1], [1], [8]) # ([1], [1], [1])
				TC_tile_sizes = (32, 32, 768) # (16, 16, 16)
				# max_avg_cost_diff = 0.2 # 0.1 # float('inf') 0.2 
				reorder_k_by_nnz = True
				only_TC, only_ELL = False, False
				log_file=f"log_final/{name}{data_i}_{feat_size}fp16.py"
				max_bucket_size = estimate_max_bucket_size(op)
				penalty = 1

				# FOR pure TC formats--------------------
				# only_ELL = True

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




				# my_branch_and_bound = reload(my_branch_and_bound)

				old_selected_tiles_0 = selected_tiles
				selected_tiles = my_branch_and_bound.post_process_for_withdraw(op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)

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


				# sampel the selected tiles and measure their latency.
				measure_for_cost_model(selected_tiles, "costModelPerf_fp16_updateTCELLModel2.csv", dsize, name, only_ELL)
				continue





for i in results:
	print(i)

