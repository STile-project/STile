# train the cost model
# try MLP first, if have time, try XGBoost




import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


import my_search_formats
from importlib import reload
my_search_formats = reload(my_search_formats)

from my_search_formats import *

from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import mean_squared_error



def get_features(tile):
	features = list()
	row_num = tile.tile_sizes[0][1]
	col_num = tile.op.idx_lens[2]
	csr = tile.position_space_when_selected
	density = None
	if csr.nnz < (row_num * tile.tile_sizes[2][1]):
		density = (csr.nnz+1)/row_num/col_num
	else:
		density = csr.nnz/row_num/col_num
	nnzs_by_rows = csr.getnnz(axis = 1)
	nnzs_by_rows_feats = [np.mean(nnzs_by_rows), max(nnzs_by_rows), min(nnzs_by_rows)]
	# 
	same_cache_lines = csr.indices
	if csr.nnz < (row_num * tile.tile_sizes[2][1]):
		same_cache_lines = np.concatenate([csr.indices, [0]])
	same_cache_lines, counts = np.unique(same_cache_lines, return_counts=True)
	same_cache_lines_feats = [np.mean(counts), max(counts), min(counts)]
	# 
	distinct_cache_lines_perwarp = len(same_cache_lines)
	total_cache_lines_perwarp = sum(counts)
	adjacent_vector_distances = list()
	for i in range(csr.shape[0]-1):
		extra = 0
		if (len(csr[i].indices)<tile.tile_sizes[2][1]) ^ (len(csr[i+1].indices)<tile.tile_sizes[2][1]):
			extra = 0
		else:
			extra = 1
		adjacent_vector_distances.append(extra + len(csr[i].indices)+len(csr[i+1].indices)-2*len(np.intersect1d(csr[i].indices, csr[i+1].indices)))
	adjacent_vector_distances_feats = [np.mean(adjacent_vector_distances), max(adjacent_vector_distances), min(adjacent_vector_distances)]
	# 
	# add feature about the unique output row number and total output row number
	indices_reordered_i = None
	if tile.tile_i_rng==None:
		indices_reordered_i = tile.i_vals
		if csr.shape[0] < row_num:
			indices_reordered_i = np.concatenate([indices_reordered_i, [indices_reordered_i[-1]]])
	else:
		indices_reordered_i = tile.op.hyb_new_rows[0][tile.tile_i_rng[0]: tile.tile_i_rng[1]+1]
		if csr.shape[0] < row_num:
			indices_reordered_i = np.concatenate([indices_reordered_i, [tile.op.hyb_new_rows[0][-1]]])
	indices_reordered_i, counts = np.unique(indices_reordered_i, return_counts=True)
	disctict_row_num = len(indices_reordered_i)
	same_row_nums_feat = [np.mean(counts), max(counts), min(counts)]
	# 
	features = [row_num, tile.tile_sizes[2][1], density, *nnzs_by_rows_feats, *same_cache_lines_feats, distinct_cache_lines_perwarp, 
		total_cache_lines_perwarp, *adjacent_vector_distances_feats, 
		disctict_row_num, *same_row_nums_feat]
	features = [float(i) for i in features]
	return features






# def get_MLP_features_in_batch(op, tile_sizes, tile_pos_is):
def get_MLP_features_in_batch(tile_sizes, tile_pos, start, end, indptr, indices, csr_shape, idx_i_for_pad, col_num, indices_reordered_i):

	# print("MLP: run in batches")
	# import time
	# time1 = time.time()

	features = list()
	row_num = tile_sizes[0][1]
	# col_num = op.idx_lens[2]

	# tile_i_rngs = [tile_sizes[0][1] * tile_pos_is, np.minimum(tile_sizes[0][1] * (tile_pos_is+1), op.position_space[0].shape[0]) - 1]
	# csr = op.position_space[0][range(tile_rngs[0][0], min(tile_rngs[0][1], len(op.hyb_new_rows[0]))),:]
	

	# densitys = (op.position_space[0].indptr[tile_i_rngs[1] + 1] - op.position_space[0].indptr[tile_i_rngs[0]])/row_num/col_num
	densitys = indptr[-1] / row_num/col_num

	# nnzs = op.position_space[0].getnnz(axis = 1)
	# valid_row_nums = tile_i_rngs[1]+1 - tile_i_rngs[0]

	nnzs = np.diff(indptr)
	valid_row_nums = sum(nnzs>0)

	# nnzs_by_rows = np.zeros((len(tile_pos_is), row_num))
	# for i in range(len(tile_pos_is)):
	# 	nnzs_by_rows[i][:valid_row_nums[i]] = nnzs[ tile_i_rngs[0][i]:tile_i_rngs[1][i]+1 ]


	nnzs_by_rows = nnzs

	# nnzs_by_rows_feats = np.mean(nnzs_by_rows, axis=1), np.max(nnzs_by_rows, axis=1), np.min(nnzs_by_rows, axis=1)
	
	nnzs_by_rows_feats = np.mean(nnzs_by_rows), np.max(nnzs_by_rows), np.min(nnzs_by_rows)
	# 
	# same_cache_lines = np.zeros( (len(tile_pos_is), row_num, tile_sizes[2][1]) )
	
	# indices = op.position_space[0].indices
	# indptr = op.position_space[0].indptr

	# for i in range(len(tile_pos_is)):
	# 	for j in range(valid_row_nums[i]):		
	# 		same_cache_lines[i][j][ :nnzs[ tile_i_rngs[0][i]+j ] ] = indices[ indptr[ tile_i_rngs[0][i]+j ]:indptr[ tile_i_rngs[0][i]+j+1 ] ]


	same_cache_lines = np.zeros( (row_num, tile_sizes[2][1]) )
	for j in range(valid_row_nums):
		same_cache_lines[j][ :nnzs[j] ] = indices[ indptr[ j ]:indptr[ j+1 ] ]


	# same_cache_lines = same_cache_lines.reshape( (len(tile_pos_is), -1) )
	same_cache_lines = same_cache_lines.flatten()



	# distinct_cache_lines_perwarps = np.zeros(len(tile_pos_is))
	# same_cache_lines_feats = [np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is))]
	# total_cache_lines_perwarps = np.zeros(len(tile_pos_is))
	# for i, cache_lines in enumerate(same_cache_lines):
	# 	tmp, counts = np.unique(cache_lines, return_counts=True)
	# 	same_cache_lines_feats[0][i] = np.mean(counts)
	# 	same_cache_lines_feats[1][i] = np.max(counts)
	# 	same_cache_lines_feats[2][i] = np.min(counts)
	# 	distinct_cache_lines_perwarps[i] = len(tmp)
	# 	total_cache_lines_perwarps[i] = sum(counts) # should be the same for all ELL blocks, may delete it
	# 


	tmp, counts = np.unique(same_cache_lines, return_counts=True)
	same_cache_lines_feats = np.mean(counts), np.max(counts), np.min(counts)
	distinct_cache_lines_perwarps = len(tmp)
	total_cache_lines_perwarps = sum(counts) # should be the same for all ELL blocks, may delete it


	# adjacent_vector_distances = np.zeros((len(tile_pos_is), row_num-1))
	# for i, cache_lines in enumerate(same_cache_lines):
	# 	tmp = cache_lines.reshape((row_num, tile_sizes[2][1]))
	# 	adjacent_vector_distances[i] = np.asarray([ len(np.union1d(tmp[j], tmp[j+1])) - len(np.intersect1d(tmp[j], tmp[j+1])) for j in range(row_num-1) ])
	# adjacent_vector_distances_feats = np.mean(adjacent_vector_distances, axis=1), np.max(adjacent_vector_distances, axis=1), np.min(adjacent_vector_distances, axis=1)


	tmp = same_cache_lines.reshape((row_num, tile_sizes[2][1]))
	adjacent_vector_distances = np.asarray([ len(np.union1d(tmp[j], tmp[j+1])) - len(np.intersect1d(tmp[j], tmp[j+1])) for j in range(row_num-1) ])
	adjacent_vector_distances_feats = np.mean(adjacent_vector_distances), np.max(adjacent_vector_distances), np.min(adjacent_vector_distances)



	# 
	# # add feature about the unique output row number and total output row number
	# disctict_row_nums = np.zeros(len(tile_pos_is))
	# same_row_nums_feats = [np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is))]
	# for i in range(len(tile_pos_is)):
	# 	indices_reordered_i = np.full(row_num, op.hyb_new_rows[0][-1])
	# 	start, end = tile_i_rngs[0][i], tile_i_rngs[1][i]
	# 	indices_reordered_i[:end+1-start] = op.hyb_new_rows[0][start: end+1]
	# 	tmp, counts = np.unique(indices_reordered_i, return_counts=True)
	# 	disctict_row_nums[i] = len(tmp)
	# 	same_row_nums_feats[0][i] = np.mean(counts)
	# 	same_row_nums_feats[1][i] = np.max(counts)
	# 	same_row_nums_feats[2][i] = np.min(counts)



	tmp, counts = np.unique(indices_reordered_i, return_counts=True)
	disctict_row_nums = len(tmp)
	same_row_nums_feats = np.mean(counts), np.max(counts), np.min(counts)

	# 
	# features = [np.full(len(tile_pos_is), row_num), np.full(len(tile_pos_is), tile_sizes[2][1]), densitys, *nnzs_by_rows_feats, *same_cache_lines_feats, 
	# 	distinct_cache_lines_perwarps, total_cache_lines_perwarps, *adjacent_vector_distances_feats, 
	# 	disctict_row_nums, *same_row_nums_feats]

	features = [row_num, tile_sizes[2][1], densitys, *nnzs_by_rows_feats, *same_cache_lines_feats, 
		distinct_cache_lines_perwarps, total_cache_lines_perwarps, *adjacent_vector_distances_feats, 
		disctict_row_nums, *same_row_nums_feats]

	# features = np.asarray(features).astype('float').T

	features = np.asarray(features).astype('float')


	# time2 = time.time()
	# print(f"time2-time1: {time2-time1}, num of tiles: {len(tile_pos_is)}, avg compute time: {(time2-time1)/len(tile_pos_is)}")

	return features














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





def get_selected_tiles(op_type, name, feat_size, filename, data_i, bench_cost_model=False, m=4096, patch_size = 2, mask_ratio = 0.75, 
	max_level_num=float('inf'), 
	only_TC = False, only_ELL=False):

	op = get_op(op_type, data_i, feat_size, name, m=m)

	os.environ['op_type'] = op_type

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

	max_bucket_size = 32 # 256 # 32
	use_faster_tuner = True
	cache_set = ['A'] # [] # ['A']
	dsize = 16 # 32 # 16
	kernel_tile_size_options = ([1], [1], [1]) # ([1], [1], [8]) # ([1], [1], [1])
	TC_tile_sizes = (32, 32, 768) # (16, 16, 16)
	max_avg_cost_diff = 0.2 # 0.1 # float('inf') 0.2 
	reorder_k_by_nnz = True
	# only_TC, only_ELL = False, False

	if only_ELL:
		max_avg_cost_diff = float('inf')

	if bench_cost_model:
		only_TC, only_ELL = False, False
		max_avg_cost_diff = 0.2


	log_file=f"log_final/{name}{data_i}_{feat_size}fp16.py"
	max_bucket_size = estimate_max_bucket_size(op)
	penalty = 1

	# FOR pure TC formats--------------------
	# only_ELL = True

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


	# sample the selected tiles and measure their latency.
	return selected_tiles




def get_tile(selected_tiles, tile_pos, tile_sizes):
	print(tile_pos, tile_sizes)
	ret = list()
	for t in selected_tiles:
		if (t.tile_pos == tuple(tile_pos)) and (t.tile_sizes[2][1] == tile_sizes[2][1]):
			# return t
			ret.append(t)
	assert len(ret) == 1, f"len(ret): {len(ret)}"
	return ret[0]


def get_dataset(tiles_dict, data, data_atomic, filename):
	# it is not sufficient to use the tile_pos to identify tiles, as we will generate new ops during search
	# need to divide the selected tiles by tile_K, and use tile_pos for each K as the ID.
	# In other words, should use tileK and tile_pos together as the ID.
	with open(filename, 'r') as file:
		lines = file.readlines()
		selected_tiles = None
		for line in lines:
			if (line == '\n') or ('tile_K' in line):
				continue
			if 'Next Measure Set' in line:
				pos = line.find(", ('only_ELL', True)")
				name = line[len('Next Measure Set  '):pos]
				selected_tiles = tiles_dict[name]
				assert len(set([(t.tile_pos, t.tile_sizes[2][1]) for t in selected_tiles])) == len(selected_tiles)
			else:
				res = json.loads(line)
				name, tile_pos, tile_sizes, _, _, _, cost, set_atomic, is_atomic = res
				tile = get_tile(selected_tiles, tile_pos, tile_sizes)
				features = get_features(tile)
				if set_atomic:
					data_atomic.append((features, cost))
				else:
					data.append((features, cost))







def get_dataset_from_csrs(data, data_atomic, filename):
	with open(filename, 'r') as file:
		lines = file.readlines()
		for line in lines:
			info, cost, set_atomic = json.loads(line)
			info = json.loads(info)
			tile_sizes = info['tile_sizes']
			tile_pos = info['tile_pos']
			start, end = info['start, end']
			indptr = info['indptr']
			indices = info['indices']
			csr_shape = info['csr_shape']
			idx_i_for_pad = info['idx_i_for_pad']
			col_num = info['col_num']
			indices_reordered_i = info['indices_reordered_i']
			# 
			features = get_MLP_features_in_batch(tile_sizes, tile_pos, start, end, indptr, indices, csr_shape, idx_i_for_pad, col_num, indices_reordered_i)
			if set_atomic:
				data_atomic.append((features, cost))
			else:
				data.append((features, cost))







def get_tile_dict():
	FraudAmazon_num_edges={('user', 'net_upu', 'user'): 351216, ('user', 'net_usu', 'user'): 7132958, ('user', 'net_uvu', 'user'): 2073474}
	names = [('FraudAmazon', k) for k in FraudAmazon_num_edges]

	mag_num_edges={('author', 'affiliated_with', 'institution'): 1043998, ('author', 'writes', 'paper'): 7145660}
	names = names + ['coauthor'] + [('mag', k) for k in mag_num_edges]

	mag_num_edges={('paper', 'cites', 'paper'): 5416271, ('paper', 'has_topic', 'field_of_study'): 7505078}
	names = names + [('mag', k) for k in mag_num_edges]

	# # FOR DEBUG
	# names = ['coauthor']

	op_type = 'spmm'

	filename = "costModelPerf_fp16_with_tilePosInfor.csv"
	tiles_dict = dict()

	for name in names:
		tot_num = 1
		if name in ['pruned_bert', 'pruned_bert_unstructured']:
			tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
		for data_i in range(tot_num):		
			for feat_size in [32]: #[32, 64, 128, 256, 512]:
				for max_avg_cost_diff in [float('inf')]: # [0.2]: #[0, 0.2, 0.4, 0.6]:
					print(f"name: {(name, data_i)}  feat_size: {feat_size}")
					selected_tiles = get_selected_tiles(op_type, name, feat_size, filename, data_i, bench_cost_model=False, 
						max_level_num=float('inf'), only_TC = False, only_ELL=True)
					tiles_dict[str(name)] = selected_tiles

	return tiles_dict





# define the dataset
class MyDataset(Dataset):
	def __init__(self, X, Y):
		self.features = X
		self.labels = Y

	def __getitem__(self, index):
		x = self.features[index]
		y = self.labels[index]        
		return x, y

	def __len__(self):
		return self.labels.shape[0]





def standardize_feature(features, param_path='preprcess_param.pt'):
	means = features.mean(dim=0, keepdim=True)
	stds = features.std(dim=0, keepdim=True) + 1e-6
	print(means)
	print(stds)
	torch.save({'num_features': len(features[0]), 'means': means, 'stds': stds}, param_path)
	normalized_data = (features - means) / stds
	return normalized_data




def standardize_feature_infer(features, means, stds):
	normalized_data = (features - means) / stds
	return normalized_data


def get_features_infer(tile, means, stds):
	features = get_features(tile)
	return standardize_feature_infer(features, means, stds)



def get_train_test_data(filename='train_dataset.csv', param_path='preprcess_param.pt', get_final_model=False):
	from sklearn.model_selection import train_test_split
	features = list()
	labels = list()
	with open(filename, 'r') as file:
		lines = file.readlines()
		for line in lines:
			a, b = json.loads(line)
			features.append(a)
			labels.append([b])
	# need to scale the features
	features = torch.as_tensor(features, dtype=torch.float)
	labels = torch.as_tensor(labels, dtype=torch.float)
	features = standardize_feature(features, param_path=param_path)
	X, Y = features, labels
	if get_final_model:
		X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=1)
		return X_train, X_val, None, Y_train, Y_val, None
	else:		
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
		X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)
		return X_train, X_val, X_test, Y_train, Y_val, Y_test






def get_data_loader(filename='train_dataset.csv', param_path='preprcess_param.pt', get_final_model=False):
	X_train, X_val, X_test, Y_train, Y_val, Y_test = get_train_test_data(filename, param_path=param_path, get_final_model=get_final_model)

	train_ds = MyDataset(X_train, Y_train)
	val_ds = MyDataset(X_val, Y_val)
	test_ds = None
	if not get_final_model:
		test_ds = MyDataset(X_test, Y_test)

	train_loader = DataLoader(
		dataset=train_ds,
		batch_size=32,
		shuffle=True,
	)

	val_loader = DataLoader(
		dataset=val_ds,
		batch_size=32,
		shuffle=False,
	)

	test_loader = None
	if not get_final_model:
		test_loader = DataLoader(
			dataset=test_ds,
			batch_size=32,
			shuffle=False,
		)

	return X_train.shape[1], train_loader, val_loader, test_loader






# define the model
class PyTorchMLP(torch.nn.Module):
	def __init__(self, num_features):
		super().__init__()

		self.all_layers = torch.nn.Sequential(
			# 1st hidden layer
			torch.nn.Linear(num_features, 64),
			torch.nn.ReLU(),

			# 2nd hidden layer
			torch.nn.Linear(64, 32),
			torch.nn.ReLU(),

			# # 2nd hidden layer
			# torch.nn.Linear(64, 32),
			# torch.nn.ReLU(),

			# output layer
			torch.nn.Linear(32, 1),
		)

	def forward(self, x):
		logits = self.all_layers(x)
		return logits








def compute_accuracy(model, dataloader):
	model = model.eval()
	correct = 0.0
	total_examples = 0
	preds = list()
	reals = list()
	# for idx, (features, labels) in enumerate(dataloader):
	# 	# compute in batches
		# 
	# 	with torch.inference_mode(): # basically the same as torch.no_grad
	# 		logits = model(features)
	# 		preds.append(logits)
	# 		reals.append(labels)
	with torch.no_grad():
		for idx, (features, labels) in enumerate(dataloader):
			logits = model(features)
			preds.append(logits)
			reals.append(labels)
	# compute the RMSE
	return mean_squared_error(np.concatenate(reals), np.concatenate(preds))








# train the model

def train_model(num_features, train_loader, val_loader, test_loader, num_epochs = 100, lr=1e-3, model_path=f'trained_model', get_final_model=False):
	loss_func = torch.nn.MSELoss()
	# 
	torch.manual_seed(1)
	model = PyTorchMLP(num_features)
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # Stochastic gradient descent
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
	# 
	# num_epochs = 100
	# 
	best_val_acc = float('inf')
	best_epoch = None
	# 
	for epoch in range(num_epochs):
		model = model.train()
		for batch_idx, (features, labels) in enumerate(train_loader):
			# 
			logits = model(features)
			# 
			# loss = F.cross_entropy(logits, labels) # Loss function
			# print(logits.shape, labels.shape)
			loss = loss_func(logits, labels)
			# 
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# 
			### LOGGING
			# print(f"Epoch: {epoch+1}/{num_epochs}"
			# 	f" | Batch {batch_idx}/{len(train_loader)}"
			# 	f" | Train/Val Loss: {loss}")
			# 
		model = model.eval()
		train_acc = compute_accuracy(model, train_loader)
		val_acc = compute_accuracy(model, val_loader)
		# test_acc = compute_accuracy(model, test_loader)
		# print(f"Epoch: {epoch+1}/{num_epochs} | Train RMSE {train_acc} | Val RMSE {val_acc} | Test RMSE {test_acc}")
		print(f"Epoch: {epoch+1}/{num_epochs} | Train RMSE {train_acc} | Val RMSE {val_acc}")
		# print(f"Epoch: {epoch+1}/{num_epochs} | Val RMSE {val_acc}")
		# 
		if val_acc < best_val_acc:
			best_val_acc = val_acc
			# model_path = model_path
			torch.save(model.state_dict(), model_path)
			best_epoch = epoch+1
	# 
	model = model.eval()
	train_acc = compute_accuracy(model, train_loader)
	val_acc = compute_accuracy(model, val_loader)
	test_acc = None
	if not get_final_model:
		test_acc = compute_accuracy(model, test_loader)
	# 
	print(f"Train RMSE {train_acc}")
	print(f"Val RMSE {val_acc}")
	print(f"Test RMSE {test_acc}")
	print(f"BEST epoch: {best_epoch}")
	return model






def load_model(PATH = 'trained_model', param_path = 'preprcess_param.pt'):
	loaded = torch.load(param_path)
	num_features = loaded['num_features']
	means = loaded['means']
	stds = loaded['stds']
	# PATH = 'trained_model'
	saved_model = PyTorchMLP(num_features)
	saved_model.load_state_dict(torch.load(PATH))
	return saved_model, means, stds







def store_train_data_0():
	tiles_dict = get_tile_dict()
	filename = "costModelPerf_fp16_with_tilePosInfor.csv"
	prestrs = ['./cuda1/','./cuda2/','./cuda3/']
	data = list()
	data_atomic = list()
	for prestr in prestrs:
		get_dataset(tiles_dict, data, data_atomic, prestr+filename)


	# store the data back to file
	with open('train_dataset.csv', 'w') as file:
		for a, b in data:
			file.write(f"{json.dumps((a, b))}\n")


	# store the data back to file
	with open('train_dataset_atomic.csv', 'w') as file:
		for a, b in data_atomic:
			file.write(f"{json.dumps((a, b))}\n")






def store_train_data():
	filename = "Ours_tile_infor.csv"
	prestrs = ['./Run_script/cuda1/','./Run_script/cuda3/']
	data = list()
	data_atomic = list()
	for prestr in prestrs:
		get_dataset_from_csrs(data, data_atomic, prestr+filename)
		print(len(data))
		print(data[0])
		print(len(data_atomic))
		print(data_atomic[0])


	# store the data back to file
	with open('train_dataset.csv', 'w') as file:
		for a, b in data:
			file.write(f"{json.dumps((a.tolist(), b))}\n")


	# store the data back to file
	with open('train_dataset_atomic.csv', 'w') as file:
		for a, b in data_atomic:
			file.write(f"{json.dumps((a.tolist(), b))}\n")



'''
# train the model
from train_cost_model import *
# store_train_data()

# torch.backends.cudnn.deterministic = True
import random
random.seed(1)
torch.manual_seed(1)
# # torch.cuda.manual_seed(1)
np.random.seed(1)

get_final_model = True
num_features, train_loader, val_loader, test_loader = get_data_loader(filename='train_dataset.csv', param_path='preprcess_param.pt', get_final_model=get_final_model)
model = train_model(num_features, train_loader, val_loader, test_loader, num_epochs = 500, lr=1e-3, model_path=f'trained_model', get_final_model=get_final_model)

num_features, train_loader, val_loader, test_loader = get_data_loader(filename='train_dataset_atomic.csv', param_path='preprcess_param_atomic.pt', get_final_model=get_final_model)
model = train_model(num_features, train_loader, val_loader, test_loader, num_epochs = 500, lr=1e-3, model_path=f'trained_model_atomic', get_final_model=get_final_model)

'''
