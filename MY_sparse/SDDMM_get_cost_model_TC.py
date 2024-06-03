# Get the benchmark latency of TC tiles

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['single_level'] = 'False'


import my_search_formats
from importlib import reload
my_search_formats = reload(my_search_formats)

from my_search_formats import *


row_window_num = 108*96


def prepare_fake_graph(tile_sizes):
	# tile_sizes is of the TC tile to be benchmarked
	node_num = max(tile_sizes[0][1]*row_window_num, tile_sizes[2][1]*row_window_num)
	fake_indptr = np.arange(0, row_window_num*tile_sizes[0][1]*tile_sizes[2][1]+1, tile_sizes[2][1])
	fake_indptr = np.concatenate([fake_indptr, np.full(node_num+1 - len(fake_indptr), fake_indptr[-1])])
	fake_indices = np.concatenate([np.arange(blk_i*tile_sizes[2][1], (blk_i+1)*tile_sizes[2][1]) for blk_i in range(row_window_num) for i in range(tile_sizes[0][1])])
	g = dgl.graph(('csr', (fake_indptr, fake_indices, [])), idtype=torch.int32, num_nodes=node_num)
	return g





def get_fake_op(op_type, tile_sizes, feat_size = 32, pad=True):
	'''
	if pad is True, we will pad the shape of the sparse matrix to multiples of 128.
	'''
	from utils import get_dataset
	# 
	# op_type = 'spmm'
	# name = "arxiv"
	g = prepare_fake_graph(tile_sizes)
	# 
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = None
	if pad:
		if op_type == 'spmm':
			idx_lens = [math.ceil(g.num_dst_nodes()/128)*128, feat_size, math.ceil(g.num_src_nodes()/128)*128]
		elif op_type == 'sddmm':
			idx_lens = [math.ceil(g.num_dst_nodes()/128)*128, feat_size, math.ceil(g.num_src_nodes()/160)*160]
	else:
		idx_lens = [g.num_dst_nodes(), feat_size, g.num_src_nodes()]
	idx_graph = None
	# 
	# x = th.rand((g.num_src_nodes(), feat_size))
	rng = np.random.default_rng(seed=0)
	# B = np.random.rand(k,n).astype("float32")
	Bs = list()
	if op_type == 'spmm':
		Bs = [rng.random((idx_lens[2],feat_size)).astype("float32")]
	elif op_type == 'sddmm':
		Bs = [rng.random((idx_lens[0],feat_size)).astype("float32"), rng.random((idx_lens[2],feat_size)).astype("float32")]
	# B = rng.random((idx_lens[2],feat_size)).astype("float32")
	# 
	# indptr, indices, _ = g.adj_sparse("csr")
	indptr, indices, _ = g.adj_tensors("csr") # 更新了dgl版本之后使用这个函数名。
	# new_row_is, new_ptr = csr_to_dcsr(op_type, indptr)
	# 
	i_pad_num = 0
	if pad:
		i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 - g.num_dst_nodes()
	# indptr = list(indptr) + [indptr[-1] for tmp in range(i_pad_num)]
	indptr = np.concatenate( [indptr, np.full(i_pad_num, indptr[-1])] )
	# 
	# A_val = tuple( np.array([1]*len(indices)).astype("float32") )
	A_val = np.full(len(indices), 1, dtype="float32")
	A_csr = scipy.sparse.csr_matrix((A_val, indices, indptr), shape=( idx_lens[0], idx_lens[2] ))
	# return A_data, A_raw_poses
	# inps = [A_csr, DenseTensor(B)]
	inps = [A_csr] + [DenseTensor(B) for B in Bs]
	sparse_inps_transposed = [ A_csr.tocsc() ]
	# 
	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)



with open("cost_model_TC_fp16_4.json", 'a') as f:
	f.write('New Round\n')
	# f.write('The same large sparse tensor for different TC tiles, the same feat_size.--------------------------\n')
	f.write('The same large sparse tensor for different TC tiles, the same feat_size, change the TC schedule so that read_A and wmma_A are read every time.--------------------------\n')

tile_sizes_list = [((None, 16), (None, 16*(2**j)), (None, 16*i)) for i in range(1, 9) for j in range(1, 3)]
for tile_sizes in tile_sizes_list:
	# if tile_sizes == ((None, 16), (None, 16), (None, 16)):
	# 	continue
	op_type = 'sddmm'
	standard_tile_sizes = ((None, 16), (None, 16*(2**2)), (None, 16*8))
	# 
	if standard_tile_sizes[2][1] % tile_sizes[2][1]!=0:
		continue
	# 
	row_window_width = standard_tile_sizes[2][1] // tile_sizes[2][1]
	op = get_fake_op(op_type, standard_tile_sizes, feat_size = 512) #tile_sizes[1][1])	
	# 
	with open("cost_model_TC_fp16_4.json", 'a') as f:
		f.write(f'operator idx_lens: {op.idx_lens}, standard_tile_sizes: {standard_tile_sizes}\n')	
	# 
	# 
	os.environ['op_type'] = 'sddmm'
	os.environ['single_level'] = 'False' # 'True'
	TC_k_notsorted = True
	os.environ['TC_k_notsorted'] = 'True'
	os.environ['no_withdraw'] = 'False'
	os.environ['REMAP'] = 'False' # 'True'
	# 
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
	# 
	max_bucket_size = 32 # 256 # 32
	use_faster_tuner = True
	cache_set = ['A'] # [] # ['A']
	dsize = 16 # 32 # 16
	kernel_tile_size_options = ([1], [1], [1]) # ([1], [1], [8]) # ([1], [1], [1])
	TC_tile_sizes = (32, 32, 768) # (16, 16, 16)
	max_avg_cost_diff = float('inf') # 0.1 # float('inf') 0.2 
	reorder_k_by_nnz = True
	only_TC, only_ELL = True, False
	log_file="log_hub/CostModel_pbert_lower_bound_0320_512_1.py"
	# max_bucket_size = estimate_max_bucket_size(op)
	penalty = 1
	# 
	# 还可以直接找pure TC formation，用下面这个函数可能更快一点~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 
	# 为了benchmark设置一下TC tile sizes
	os.environ['TC_tile_sizes'] = json.dumps(tile_sizes)
	# 
	selected_tiles = None
	selected_tiles = get_TC_tile_for_benchmark(tile_sizes, op,
		cuda_i, dtype, 
		max_bucket_size, kernel_tile_size_options, TC_k_notsorted, reorder_k_by_nnz,
		)
	# 
	end_time = time.time()
	print(end_time - start_time, flush=True)
	# 
	my_branch_and_bound = reload(my_branch_and_bound)
	old_selected_tiles_0 = selected_tiles
	selected_tiles = my_branch_and_bound.post_process_for_withdraw(op, selected_tiles, None, None, None)
	# 
	# 为每个tile sizes，我们都要tune一下各个可能得parameter
	cand_warp_nums = [1] # get_factors(tile_sizes[2][1]//16)
	cand_vecsizes = [2**i for i in range(1, 4)] # range(5)]
	print("cand_warp_nums: ", cand_warp_nums, flush=True)
	print("cand_vecsizes: ", cand_vecsizes, flush=True)
	best_params = None
	best_cost = float('inf')
	for warp_num in cand_warp_nums:
		for vec_size in cand_vecsizes:
			print(warp_num, vec_size, flush=True)
			params = {'mma_shape_str': "m16n16k16", 'warp_num': warp_num, 'vec_size': vec_size}
			for t in selected_tiles:
				if t.op.loop_protocals[0] == 'uuu':
					t.params = params
					t.best_params = params
			# 
			try:
				my_fuse_formats = reload(my_fuse_formats)
				cost = None
				cost = my_fuse_formats.Benchmark_TC_BSDDMM(op, 'sddmm', 1, selected_tiles, cuda_i, cache_set, dsize, row_window_num, row_window_width,
					dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
				# 
				with open("cost_model_TC_fp16_4.json", 'a') as f:
					json.dump( (tile_sizes, params, cost, best_cost) , f)
					f.write('\n')
				# 
				if cost < best_cost:
					best_cost = cost
					best_params = params
			except Exception as e:
				print(e)
	# 
	with open("cost_model_TC_fp16_4.json", 'a') as f:
		json.dump( (tile_sizes, best_params, best_cost, "best") , f)
		f.write('\n')

