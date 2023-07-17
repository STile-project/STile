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




# names = ['citeseer', 'cora', 'ppi', 'pubmed',  'arxiv', 'proteins', 'reddit']
# names = ['citeseer', 'amazon', 'coauthor', 'GDELT'] 

names = ['cora', 'ppi', 'pubmed',  'arxiv', 'proteins', 'reddit', 'citeseer'][::-1] # + ['pruned_bert', 'pruned_bert_unstructured']
names = ['cora', 'ppi', 'pubmed',  'arxiv', 'proteins'][::-1] # + ['pruned_bert', 'pruned_bert_unstructured']
# names = ['pruned_bert', 'pruned_bert_unstructured']

results = list()

op_type = 'spmm'

for name in names:
	tot_num = 1
	get_op = None
	if name in ['pruned_bert', 'pruned_bert_unstructured']:
		tot_num, get_op = get_pruned_bert_graphNum_and_getOpFunc(name)
	for data_i in range(tot_num):		
		for feat_size in [32]: #[32, 64, 128, 256, 512]:
			for max_avg_cost_diff in [0.2]: #[float('inf')]: # [0.2]: #[0, 0.2, 0.4, 0.6]:

				os.environ['op_type'] = 'spmm'

				os.environ['single_level'] = 'False' # 'True'
				TC_k_notsorted = False
				os.environ['TC_k_notsorted'] = 'False'
				os.environ['no_withdraw'] = 'True' # 'False'


				print(f"name: {(name, data_i)}  feat_size: {feat_size}")

				# reddit
				# name = "arxiv" # "proteins" # "arxiv" "pubmed" "citeseer"
				op = None
				if name in ['pruned_bert', 'pruned_bert_unstructured']:
					op, _ = get_op('spmm', data_i, feat_size = feat_size, print_infor=False)
				else:
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


				for tmp_feat_size in [32, 64, 128, 256, 512]:


					if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (tmp_feat_size!=512):
						continue

					if tmp_feat_size!=32: 
						continue



					for i in selected_tiles:
						if len(i.best_tile_sizes[0]) == 3:
							i.best_tile_sizes = ([None, i.tile_sizes[0][1]//1, 1], i.best_tile_sizes[1], i.best_tile_sizes[2])					

					print("tmp_feat_size: ", tmp_feat_size)
					res1 = my_fuse_formats.measure_seleted_formats(op, 'spmm', 1, selected_tiles, cuda_i, cache_set, dsize, gened_inputs=list(),
						dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)

					cost1 = my_fuse_formats.measure_latency(*res1)

					print(f"cost 32 thread: {cost1}")

					cost2 = float('inf')
					if True: #ELL_num > 0:
						for i in selected_tiles:
							if len(i.best_tile_sizes[0]) == 3:
								i.best_tile_sizes = ([None, i.tile_sizes[0][1]//8, 8], i.best_tile_sizes[1], i.best_tile_sizes[2])

						res2 = my_fuse_formats.measure_seleted_formats(op, 'spmm', 1, selected_tiles, cuda_i, cache_set, dsize, gened_inputs=list(),
							dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)

						cost2 = my_fuse_formats.measure_latency(*res2)

					print(f"cost 256 thread: {cost2}")


					results.append( ((name, data_i), tmp_feat_size, max_avg_cost_diff, 
						os.environ['single_level'], os.environ['TC_k_notsorted'], os.environ['no_withdraw'],
						TC_row_num, TC_num, ELL_num, 
						tot_sum, atomic_TC_num, TC_sum, ELL_sum, 
						search_time, cost1, cost2) )
					print("summary:", results[-1], flush=True )


					with open('cost_dataset_updateTCELLModel.csv', 'a') as f:
						json.dump(results[-1], f )
						f.write('\n')

				continue  # ----------------------------------------------------------------------------------------------------------------


for i in results:
	print(i)

