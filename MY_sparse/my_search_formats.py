# This file is about searching the best format combination for a given sparse operator. 
# import gen_formats
# from importlib import reload
# gen_formats = reload(gen_formats)

from gen_formats_v2 import *

import my_cost_model

import my_branch_and_bound

import my_fuse_formats

import os
# my_cost_model = reload(my_cost_model)





global_op_id = 0






def merge_csrs_TC_tile(sorted_tiles):
	'''
	Generate the merged covered_position_space of the TC tiles sorted by their position.
	The csr returned is already transformed to the original position space.
	Assume: we assume the op of any TC tile is not kernel-tiled, and the k axis is not reordered.
	Assume: all sorted tiles are from the same op.
	'''
	op = sorted_tiles[0].op
	# (_, I), (_, J), (_, K) = sorted_tiles[0].tile_sizes
	indptr = np.zeros( op.position_space_update.shape[0] + 1 )
	nnz = sum([t.nnz_when_selected // t.j_num for t in sorted_tiles])
	indices = np.zeros(nnz)
	data = np.zeros(nnz)
	poses = [t.tile_pos for t in sorted_tiles]
	i_poses, counts = np.unique([t.tile_pos[0] for t in sorted_tiles], return_counts=True)
	splits = np.concatenate([[0], np.cumsum(counts)])
	
	# print(splits)

	tot = 0
	last_row = 0
	for cnt, i in enumerate(i_poses):
		# print(i)
		# 因为我们在local search的时候允许每行的withdraw选项有不同的tile shape，所以此处要为每一行分别确定tile sizes
		(_, I), (_, J), (_, K) = sorted_tiles[ splits[cnt] ].tile_sizes
		ks = np.concatenate([range(int(K*pos[2]), min(int(K*(pos[2]+1)), len(op.k_vals[i]) )) for pos in poses[ splits[cnt]:splits[cnt+1] ] ])
		csr = op.position_space_update[ I*i:I*(i+1), :][:, op.k_vals[i][ks] ]
		indptr[last_row:I*i] = tot
		indptr[I*i:I*(i+1)+1] = tot + csr.indptr
		# print(sum([t.position_space_when_selected.nnz for t in sorted_tiles[ splits[cnt]:splits[cnt+1] ]]))
		# print(f"{cnt}/{len(i_poses)}, {i} || csr.nnz:{csr.nnz} tot_indices:{len(indices)} len(indices):{len(indices[tot:tot+csr.nnz])} [{tot, tot+csr.nnz}] len(right):{len(op.idx_values_list[0][2][ op.k_vals[i][ks][csr.indices] ])}")
		indices[tot:tot+csr.nnz] =  op.idx_values_list[0][2][ op.k_vals[i][ks][csr.indices] ]
		data[tot:tot+csr.nnz] = csr.data
		last_row = I*(i+1)
		tot = tot+csr.nnz
	indptr[last_row:] = tot

	csr = scipy.sparse.csr_matrix(
				(data, indices, indptr), 
				shape=op.position_space_update.shape)

	ind_i = np.argsort(op.idx_values_list[0][0])

	# 把unfold_covered_csr还原回原始op的in1的矩阵坐标
	return csr[ind_i,:]










def merge_csrs_1D_SDDMM_tile(sorted_tiles):
	'''
	Generate the merged covered_position_space of the TC tiles sorted by their position.
	The csr returned is already transformed to the original position space.
	Assume: we assume the op of any TC tile is not kernel-tiled, and the k axis is not reordered.
	Assume: all sorted tiles are from the same op.
	'''
	op = sorted_tiles[0].op
	data = np.zeros( op.position_space[0].nnz )
	for t in sorted_tiles:
		data[t.tile_pos[0]*t.tile_sizes[0] : (t.tile_pos[0]+1)*t.tile_sizes[0]] = t.position_space_when_selected

	csr = scipy.sparse.csr_matrix(
				(data, op.position_space[0].indices, op.position_space[0].indptr), 
				shape=op.position_space[0].shape)

	ind_i = np.argsort(op.idx_values_list[0][0])
	ind_j = np.argsort(op.idx_values_list[0][2])

	# 把unfold_covered_csr还原回原始op的in1的矩阵坐标
	return csr[ind_i,:][:,ind_j]








def update_position_space_for_ops_given_tiles(sub_ops, selected_tiles):
	'''
		根据一组select tile的position space when selected来更新该op的position space。
		这里我们采用的是scipy sparse csr matrix 的数据结构。
	'''
	if len(selected_tiles) == 0:
		return

	op = selected_tiles[0].op
	if op.op_type in ['spmm', 'sddmm']:
		# =======================================================================================
		# 换一个更快的写法求unfold_covered_csr们的和
		# 计算dummy_tile的position_space_when_selected
		unfold_covered_csr = None
		if get_template_str(op) in ['TensorCore_template', 'TC_sddmm']:
			sorted_tiles = sorted(selected_tiles, key=lambda t: t.tile_pos) # 把tile按i轴位置排序 
			unfold_covered_csr = merge_csrs_TC_tile(sorted_tiles)
			print("dummy_csr.nnz: ", unfold_covered_csr.nnz)
		elif get_template_str(op) == 'sparse_template_ell' :
			dummy_tile = ComputeTile(op, 
				((None, op.position_space_update.shape[0]), 
					(None, selected_tiles[0].tile_sizes[1][1]), 
					(None, op.position_space_update.shape[1])), (0, 0, 0), selected_tiles[0].params)
			dummy_indptr = np.zeros( op.position_space_update.shape[0] + 1 )
			tot, last_row = 0, 0
			sorted_tiles = sorted(selected_tiles, key=lambda t: t.tile_pos) # 把tile按i轴位置排序 
			for t in sorted_tiles:
				dummy_indptr[ last_row:t.tile_i_rng[0]+1 ] = tot
				dummy_indptr[ t.tile_i_rng[0]:t.tile_i_rng[1]+2 ] = t.position_space_when_selected.indptr + tot
				tot = tot + t.position_space_when_selected.indptr[-1]
				last_row = t.tile_i_rng[1]+1
			dummy_indptr[last_row:] = tot
			dummy_indices = np.concatenate([t.position_space_when_selected.indices for t in sorted_tiles])
			dummy_data = np.concatenate([t.position_space_when_selected.data for t in sorted_tiles])
			dummy_csr = scipy.sparse.csr_matrix(
					(dummy_data, dummy_indices, dummy_indptr), 
					shape=op.position_space_update.shape)
			dummy_tile.position_space_when_selected = dummy_csr
			unfold_covered_csr = transform_covered_position_space_to_ori_space(dummy_tile)
		elif get_template_str(op) == '1D_sddmm':
			sorted_tiles = sorted(selected_tiles, key=lambda t: t.tile_pos) # 把tile按i轴位置排序 
			unfold_covered_csr = merge_csrs_1D_SDDMM_tile(sorted_tiles)
			print("dummy_csr.nnz: ", unfold_covered_csr.nnz)

		# 考虑到withdraw的情况，有的时候仅仅是减少了某行的cost，而没有新增cover的点
		assert (unfold_covered_csr.nnz == 0) or (max(unfold_covered_csr.data) == 1)
		print(f"unfold_covered_csr: {unfold_covered_csr.nnz}")
		# 
		# NOTE: 我们这里有一个假设，即我们所处理的所有position space里面的元素的可能值都只有0和1，不包含该矩阵原本的元素值。
		for sub_op in sub_ops:
			# 需要再把unfold_covered_csr转化成和sub_op的position space相一致的坐标的矩阵
			tmp_csr = unfold_covered_csr[sub_op.idx_values_list[0][0],:][:,sub_op.idx_values_list[0][2]]
			if get_template_str(sub_op) == 'sparse_template_ell':
				# tmp_csr = tmp_csr[sub_op.hyb_new_rows[0],:]
				indptr = np.cumsum(np.concatenate( ([0], tmp_csr.getnnz(axis=1)[sub_op.hyb_new_rows[0]]) ))
				indices = np.concatenate([ tmp_csr.indices[ tmp_csr.indptr[i]:tmp_csr.indptr[i+1] ]  for i in sub_op.hyb_new_rows[0] ])
				data = np.ones(len(indices))
				tmp_csr = scipy.sparse.csr_matrix(
						(data, indices, indptr), 
						shape=sub_op.position_space_update.shape)

			# elif get_template_str(sub_op) == '1D_sddmm':
			# 	# 这个template也涉及到折叠，所以也需要还原一下折叠后的tmp_csr
			# 	# 但是好像很难做到啊应该怎么还原折叠？感觉需要一个矩阵来标记原有的每个nonzero对应的行号，这样可以把折叠后的还原，但是原始的还是不能得到还原的
			# 	# easy可以先还原成未折叠的，然后令tmpcsr中的元素对应位置值为-1，然后还原成折叠后的矩阵，然后所有元素加上2，再整除2，即可得到更新之后的矩阵。
			# 	data = (sub_op.position_space[0]+sub_op.position_space_update).data//2
			# 	csr = sub_op.inps[0][sub_op.idx_values_list[0][0],:][:,sub_op.idx_values_list[0][2]]
			# 	csr0 = scipy.sparse.csr_matrix((data, sub_op.position_space[0].indices, csr.indptr), shape=csr.shape)
			# 	csr1 = scipy.sparse.csr_matrix((sub_op.position_space[0].data, sub_op.position_space[0].indices, csr.indptr), shape=csr.shape)
				
			# 	data = (tmp_csr.multiply(csr0) + csr1).data//2
			# 	tmp_csr = scipy.sparse.csr_matrix(
			# 		(data, sub_op.position_space[0].indices, sub_op.position_space[0].indptr), 
			# 		shape=sub_op.position_space[0].shape)


			print(f"before update: {sub_op.position_space_update.nnz}")
			sub_op.position_space_update = sub_op.position_space_update - tmp_csr.multiply(sub_op.position_space_update)
			print(sub_op.position_space_update.nnz)
			# 似乎更新之后，sub_op.position_space_update里面的0也会被同时清除，无所谓了
			# sub_op.position_space_update = sub_op.position_space_update - tmp_csr.multiply(sub_op.position_space_update)

			sub_op.position_space_update = sub_op.position_space_update.tocsr()
			# 更新每个sub_op的position_space_update，之后还需要把里面新增的zeros全部都eliminate掉
			sub_op.position_space_update.eliminate_zeros()

			if get_template_str(sub_op) == '1D_sddmm':
				# 需要更新nnz_update_1D，nnz_data_1D
				# 其实并不需要保证position space update的indices是sorted的
				# sub_op.position_space_update.has_sorted_indices = False
				# sub_op.position_space_update.sort_indices()
				csr = (sub_op.position_space[0]+sub_op.position_space_update).tocsr()
				csr.eliminate_zeros()
				csr.has_sorted_indices = False
				csr.sort_indices()
				sub_op.nnz_data_1D = csr.data//2
				sub_op.nnz_update_1D = np.add.reduceat(
					sub_op.nnz_data_1D, 
					np.arange(0, len(sub_op.nnz_data_1D), sub_op.max_bucket_size)
					)









def update_op_after_tile_selection(op):
	'''
		Require the position_space_update of op has been updated.
	'''
	if op.op_type in ['spmm', 'sddmm']:
		inps = [op.position_space_update ] + op.inps[1:]
		sparse_inps_transposed = [ inps[0].tocsc() ]

		new_op = Op(op.op_type, op.sidxs, op.ridxs, op.idx_lens, op.idx_graph, inps, sparse_inps_transposed)

		new_op.position_space = dict()
		gen_position_space_for_area(new_op, 0)

		return new_op

		




def update_tile_after_selection(selected_tile):
	'''
	Update the information of a tile after it is selected.
	'''
	gen_updated_position_space_for_tile(selected_tile)
	selected_tile.position_space_when_selected = selected_tile.uncovered_position_space
	selected_tile.update_nnz_uncovered()
	selected_tile.nnz_when_selected = selected_tile.nnz_uncovered
	# print("THE NNZ UNCOVERED: ", selected_tile.nnz_when_selected)
	
	# directly update the uncovered_position_space of selected_tile
	# 这两行的操作其实可以不要
	selected_tile.uncovered_position_space = None
	selected_tile.nnz_uncovered = 0	

	# update the pred_avg_cost and the avg_cost
	selected_tile.set_avg_cost()
	selected_tile.set_pred_avg_cost()




def greedy_search_use_cost_model_lower_bound(op, dsize, run_params, max_bucket_size, cache_set, kernel_tile_size_options,
	only_TC, only_ELL, 
	penalty, TC_k_notsorted,
	reorder_k_by_nnz = True,
	max_level_num = float('inf'),
	max_avg_cost_diff = float('inf'), use_faster_tuner = False, log_file="log_hub/log.py", cuda_i = 3, 
	dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
	'''
	Select the computation tiles greedily.
	Input: 
		the original op to be optimized.
	Output:
		the selected tiles.
	'''
	# 需要注意的一点是，我们的greedy是对单个tile中被新cover的element的平均cost而言的，所以每选择完一个tile之后，其他所有tile对应的单element cost都需要更新。

	# 这个dict里的值需要提前设置好。但是里面应该填什么呢？填GFLOPS吗，还是真实的cost，还是暂时不考虑tensor core？感觉得直接比cost，因为tensor core的operation density和其余template的不是相同量级（或者说单位）的
	# tensor_core_cost_dict = dict()

	no_selected_tile_round = 0

	os.environ['MyFileID'] = str(cuda_i)
	if dtype == "float16":
		os.environ['MydtypeIn'] = 'half'
		os.environ['MydtypeOut'] = 'half'
	elif dtype == "float32":
		os.environ['MydtypeIn'] = 'float'
		os.environ['MydtypeOut'] = 'float'

	is_tb_tile = True

	tuning_hub = dict() # 存储tuning的结果，当遇到tile_size 和 sub_op都一样的tile的时候，可以不必tuning，而直接复用已有结果

	# 1. get level 1 possible sub_ops
	global global_op_id
	cand_ops, global_op_id = gen_candidate_formats(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
		reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = 0, gen_TC_formats = (not only_ELL))
	
	# TODO: 暂时先不管TC tiles，对于TC tiles，我们需要为其建立一个以nnz为key的index，方便我们快速搜索nnz最多的TC tile


	# 2. greedily select tiles
	ori_tot_nnz, SM_num = op.inps[0].getnnz(), 108

	tot_nnz = -1
	if op.op_type in ['spmm', 'sddmm']:
		tot_nnz = op.inps[0].getnnz() * op.idx_lens[1]


	new_op = op
	level_num = 1
	round_i = 0

	selected_tiles = list()
	tile_dict = dict()
	nnzs_dict, pos_is_dict, costs_dict = dict(), dict(), dict()

	# 我们暂时不考虑Tensor core tiles  TODO！！！！
	all_subops, next_level_subops, all_TC_ops = list(), list(), list()
	for sub_op in cand_ops:
		gen_position_space_for_area(sub_op, 0)
		print(f"sub_op {sub_op.op_id}  initial nnz: {sub_op.position_space_update.nnz}")
		if (get_template_str(sub_op) in ['TensorCore_template', 'TC_sddmm']):
			all_TC_ops.append(sub_op)
			continue

		# ====================
		# FOR DEBUG: 暂时先舍弃CSR format
		# continue
		if (sub_op.op_type == 'spmm') and (get_template_str(sub_op) != 'sparse_template_ell'):
			continue
		# ====================

		# gen_position_space_for_area(sub_op, 0)
		all_subops.append(sub_op)


	# 也需要为new_op准备position space
	new_op.position_space = dict()
	gen_position_space_for_area(new_op, 0)
	
	if not only_ELL:
		reverse_idx_TC_op_row = np.argsort(all_TC_ops[0].idx_values_list[0][0])
	else:
		reverse_idx_TC_op_row = np.argsort(np.arange(op.idx_lens[0]))

	# FOR DEBUG===============
	# all_TC_ops = list()
	# all_subops = list()
	if only_TC:
		all_subops = list()
	if only_ELL:
		all_TC_ops = list()
	# ========================

	best_possible_ELL = float('inf')
	if len(all_subops) > 0:
		best_possible_ELL = my_cost_model.lower_bound_ELL_cost(all_subops[0], max_bucket_size, dsize)
		print(f"BEST POSSIBLE ELL: {best_possible_ELL}")


	# 不管有没有TC op，ELL做local search都需要有reverse_idx_TC_op_row，所以放到可能设置all_TC_ops之前计算改变量
	# 为了支持withdraw做准备
	# reverse_idx_TC_op_row = None
	# if len(all_TC_ops) > 0:
	# 	reverse_idx_TC_op_row = np.argsort(all_TC_ops[0].idx_values_list[0][0])


	# 需要一个数据结构来记录每个TC row都涉及到哪些被选择的tile，方便进行withdraw
	# 但是我们不会实时把selected tiles列表里的tile进行清除，而是最后再这样做？也不好做啊。
	# 我们可以只记录tile在selected_tiles里面的编号，然后在最后根据这些编号进行删减。
	# 这个数据结构里面只存完全在某个TC row里的tile，覆盖范围只有一部分交叉的ELL tile不被算在里面。
	TC_row_num = math.ceil(op.inps[0].shape[0] / 16)
	TC_row_tile_dict = {i:list() for i in range(TC_row_num)}
	TC_row_costs = np.full(TC_row_num, 0.0) # 此处写成0而非0.0的话，np会自动生成正数array，导致结果错误。
	TC_is = np.full(op.inps[0].shape[0], False)

	# 先在可以不考虑ELL tile的时候单独搜一遍TC tile，默认只会有一个TC op
	# if len(all_TC_ops) > 0:
	# <jingzhi>@revision: speed up pure TC search: only do the following only TC search when there are both TC and ELL ops. ->performance is bad, delete it
	if (len(all_TC_ops) > 0) and ((not only_TC) and (not only_ELL)):
		selected_tiles = my_branch_and_bound.search_TC_tiles_only_by_rows(
			all_TC_ops[0], dsize, is_tb_tile, best_possible_ELL, ori_tot_nnz, all_TC_ops, all_subops, new_op, max_avg_cost_diff, 
			TC_row_tile_dict, TC_row_costs, penalty, max_bucket_size, TC_is)

		if op.op_type in ['spmm', 'sddmm']:
			tot_nnz = all_TC_ops[0].position_space_update.nnz * op.idx_lens[1]

		# 在生成完TC tile之后也需要generate next level tiles
		if (os.environ['single_level']!='True'):
			new_op = update_op_after_tile_selection(new_op)

			print(f"new op nnz: {new_op.position_space_update.getnnz()}")

			# =============================
			next_level_subops, global_op_id = my_branch_and_bound.gen_full_next_level_subops(
				TC_is,
				global_op_id, new_op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
				reorder_k_by_nnz = reorder_k_by_nnz, 
				is_tb_tile = is_tb_tile)
			# 尝试一下每次都只从next level tiles中找tile，而不保留之前的tile
			all_subops = next_level_subops
			next_level_subops = list()
			tile_dict = dict()
			nnzs_dict, pos_is_dict, costs_dict = dict(), dict(), dict()
		# 
	# <jingzhi>@revision: speed up the pure TC search for SDDMM by directly generating all the sparse tiles
	elif only_TC and (op.op_type == 'sddmm'):
		tile_sizes = list(all_TC_ops[0].blk_nums.keys())[0]
		params = {"mma_shape_str":"m16n16k16"}
		for blk_row_i in range(TC_row_num):
			non_empty_tile_nums = all_TC_ops[0].blk_nums[tile_sizes][blk_row_i]
			good_poses = [(int(blk_row_i), 0, i) for i in range(int(non_empty_tile_nums))]
			if all_TC_ops[0].TC_k_notsorted:
				good_poses = [(int(blk_row_i), 0, i) for i in all_TC_ops[0].blk_ks[tile_sizes][blk_row_i] ]

			good_tiles = [ComputeTile(all_TC_ops[0], tile_sizes, tile_pos, params, not_init_nnz=True) for tile_pos in good_poses]
			for t in good_tiles:
				t.is_atomic_tile = False
				t.pred_cost = 1
				t.cost = 1
				t.best_tile_sizes = t.tile_sizes
				t.best_params = t.params
				# 此时的uncovered_position_space应该是正确的。	We do not need position_space_when_selected
				# t.position_space_when_selected = t.uncovered_position_space
				t.nnz_when_selected = 0
			selected_tiles = selected_tiles + good_tiles
		selected_tiles[0].nnz_when_selected = op.inps[0].getnnz() * selected_tiles[0].j_num
		selected_tiles[0].nnz = op.inps[0].getnnz()
		tot_nnz = 0


	while True:
		if tot_nnz <= 0:
			break

		# select best
		selected_tile = None
		good_tiles = list()

		# 打算舍弃这个变量，在做local search的时候只根据预先设定的比例来做
		second_best_pred_avg_cost = None

		# consider TC tiles========================
		# we check TC tiles first before we check ELL tiles, so that we can the avg cost of TC tiles as the start point for ELLs
		if len(all_TC_ops)!=0:
			best_tile_key_TC, best_cost_TC = my_branch_and_bound.search_best_TC_tile(all_TC_ops, dsize, 
				TC_row_tile_dict, TC_row_costs, penalty, max_bucket_size)

			# if best_tile_key_TC[2][2] == 1:
			# 	return all_TC_ops

			# 此处做一个简化，即我们暂时只考虑只有TC tile的情况。
			params = {"mma_shape_str":"m16n16k16"}
			# if best_tile_key_TC[1] == (16, 32, 32):
			# 	params = {"mma_shape_str":"m8n32k16"}
			is_atomic_tile = best_tile_key_TC[3]
			# 我们可能返回同一行的多个TC tile
			for tile_pos in best_tile_key_TC[2]:
				selected_tile = ComputeTile(cand_ops[best_tile_key_TC[0]], best_tile_key_TC[1], tile_pos, params)
				if is_atomic_tile or (op.op_type=='sddmm'):
					# 因为我们支持整行withdraw地选择整行TC tile，所以只对atomic tile用最新position space来确定其覆盖范围
					selected_tile.update_nnz_uncovered()
				selected_tile.pred_cost = best_cost_TC * selected_tile.nnz_uncovered  # 在返回多个tile时，这个数据已经不准了
				selected_tile.cost = 1
				selected_tile.best_tile_sizes = selected_tile.tile_sizes
				selected_tile.best_params = selected_tile.params

				selected_tile.set_pred_avg_cost()

				selected_tile.is_atomic_tile = is_atomic_tile
				good_tiles.append(selected_tile)

			# print(f"TC avg cost: {selected_tile.pred_avg_cost}, TC nnz: {selected_tile.nnz_uncovered}")
			print(f"TC avg cost: {best_cost_TC}, TC nnz: {sum([selected_tile.nnz_uncovered for selected_tile in good_tiles])}")

		# selected_tile.set_pred_avg_cost()
		# ==========================================

		# init_avg_cost 用来加速ELL tile的搜索
		init_avg_cost = float('inf')
		# if selected_tile != None:
		# 	init_avg_cost = selected_tile.pred_avg_cost

		if len(good_tiles) > 0:
			init_avg_cost = good_tiles[0].pred_avg_cost

		# 暂时把selected tile存在selected TC tile里，因为ELL tile选出来的时候可能measure不通过，不过一般不太可能这样，之后也可以考虑去掉
		# selected_TC_tile = selected_tile
		# 直接把good tiles全存在selected_tile 和 selected_TC_tile 里面，从此以后，这两个变量就是列表而非单个tile了。
		selected_tile = good_tiles
		selected_TC_tile = good_tiles


		print(f"selected TC tiles: {selected_tile}")

		while True:	

			# 首先判断之前找到的TC tile（如果有）的avg cost是不是比ELL的最好的情况还要好，如果是，直接跳出循环
			if init_avg_cost < best_possible_ELL:
				break

			# 一直找，直到找到一个cost是有效值的tile 
			if len(all_subops) == 0:
				selected_tile = selected_TC_tile
				break

			best_tile_key, second_best_pred_avg_cost = my_branch_and_bound.search_best_tile(
				op,
				all_subops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size, 
				nnzs_dict, pos_is_dict, costs_dict,
				best_pred_avg_cost = init_avg_cost)
			
			# if best_tile_key == None:
			# 	return all_subops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size
			# 此处其实有可能找不到best_tile_key，因为我们设置了initial avg cost
			if best_tile_key != None:
				selected_tile = tile_dict[best_tile_key]
				init_avg_cost = selected_tile.pred_avg_cost

				print("find a best tile key")

			print("before find next level op", selected_tile, selected_TC_tile)

			if len(next_level_subops) > 0:
				tile_dict_next_level = dict()
				best_tile_key, second_best_pred_avg_cost_next_level = my_branch_and_bound.search_best_tile(
					op,
					next_level_subops, tile_dict_next_level, dsize, ori_tot_nnz, SM_num, max_bucket_size, 
					nnzs_dict, pos_is_dict, costs_dict,
					best_pred_avg_cost = init_avg_cost)
				if best_tile_key != None:
					# 说明一定知道找到了一个avg cost更小的next level的tile
					# update second_best_pred_avg_cost
					# 其实此处更新second_best_pred_avg_cost是有误的，但是因为我们不用second_best_pred_avg_cost了，所以无所谓
					second_best_pred_avg_cost = min(init_avg_cost, \
						second_best_pred_avg_cost, second_best_pred_avg_cost_next_level)
					# 
					selected_tile = tile_dict_next_level[best_tile_key]

					# <jingzhi>@response: 在response阶段修改。因为我们已经不再保留之前的level的tile，所以level_num修改为在生成新tiles的时候更新，而不是在这里更新。
					# level_num += 1

					all_subops = all_subops + next_level_subops
					next_level_subops = list()					

				# selected_tile_next_level = tile_dict[best_tile_key]
				# if selected_tile_next_level.pred_avg_cost < selected_tile.pred_avg_cost:
				# 	# update second_best_pred_avg_cost
				# 	second_best_pred_avg_cost = min(selected_tile.pred_avg_cost, \
				# 		second_best_pred_avg_cost, second_best_pred_avg_cost_next_level)
				# 	# 
				# 	selected_tile = selected_tile_next_level
				# 	level_num += 1
				# 	all_subops = all_subops + next_level_subops
				# 	next_level_subops = list()	

			# =============For DEBUG
			# selected_tile.cost = 1
			# break
			# return [selected_tile]
			# =============

			# 如果selected tile不是ELL tile，直接停止循环
			if selected_tile == selected_TC_tile:
				break


			# compare best other tile and best TC tile
			error_tile = my_fuse_formats.measure_tiles(op, [selected_tile], tuning_hub, dsize, cache_set,
				max_bucket_size, use_faster_tuner = use_faster_tuner,
				log_file=log_file, cuda_i = cuda_i, 
				dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
			
			# 我们已经取消返回error tile了，所以下面的这个判断已经失效-----------
			if error_tile != None:
				return error_tile, tile_dict
			# return error_tile, tile_dict

			if selected_tile.cost != float('inf'):
				break
			else:
				selected_tile = selected_TC_tile


		# consider TC tiles========================
		# if len(all_TC_ops)!=0:
		# 	best_tile_key_TC, best_nnz_TC = my_branch_and_bound.search_best_TC_tile(all_TC_ops, dsize)
		# 	# 此处做一个简化，即我们暂时只考虑只有TC tile的情况。
		# 	params = {"mma_shape_str":"m16n16k16"}
		# 	if TC_tile_sizes == (8, 32, 16):
		# 		params = {"mma_shape_str":"m8n32k16"}
		# 	selected_tile = ComputeTile(cand_ops[best_tile_key_TC[0]], best_tile_key_TC[1], best_tile_key_TC[2], params)
		# 	selected_tile.update_nnz_uncovered()
		# 	selected_tile.pred_cost = 1
		# 	selected_tile.cost = 1
		# 	selected_tile.best_tile_sizes = selected_tile.tile_sizes
		# 	selected_tile.best_params = selected_tile.params

		# # selected_tile.set_pred_avg_cost()
		# ==========================================

		if selected_tile != selected_TC_tile:
			selected_tile = [selected_tile]


		if len(selected_tile) == 0:
			no_selected_tile_round+=1
			if no_selected_tile_round == 2:
				# there should not be two continuous rounds where no tiles are selected.
				return op, all_subops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size, init_avg_cost, next_level_subops
			# 
			# return op, all_subops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size, init_avg_cost, next_level_subops
			new_op = update_op_after_tile_selection(new_op)

			print(f"new op nnz: {new_op.position_space_update.getnnz()}")

			# =============================
			next_level_subops, global_op_id = my_branch_and_bound.gen_full_next_level_subops(
				TC_is,
				global_op_id, new_op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
				reorder_k_by_nnz = reorder_k_by_nnz, 
				is_tb_tile = is_tb_tile)
			# 尝试一下每次都只从next level tiles中找tile，而不保留之前的tile
			all_subops = next_level_subops
			next_level_subops = list()
			tile_dict = dict()
			nnzs_dict, pos_is_dict, costs_dict = dict(), dict(), dict()
			continue

		no_selected_tile_round = 0

		# print("selected tile avg_cost and pred_avg_cost: ", selected_tile.avg_cost, selected_tile.pred_avg_cost)
		print("selected tile avg_cost and pred_avg_cost: ", selected_tile[0].avg_cost, selected_tile[0].pred_avg_cost)

		# if selected_tile.tile_sizes == ((None, 128), (None, 128), (1, 128)):
		# 	print(selected_tile.get_key())
		# 	print(selected_tile.nnz_when_selected, selected_tile.cost, selected_tile.pred_cost)


		# ---------------------
		# best_pred_avg_cost = selected_tile.pred_avg_cost

		# # update selected_tile infor
		# update_tile_after_selection(selected_tile)
		# selected_tiles.append(selected_tile)
		# ----------------------


		best_pred_avg_cost = selected_tile[0].pred_avg_cost

		# 等到做完了local search之后一起更新，其实只需要更新good tiles即可=====================
		# update selected_tile infor
		# for tile in selected_tile:
		# 	update_tile_after_selection(tile)
		# selected_tiles = selected_tiles + selected_tile
		# =====================================================

		# =============For DEBUG
		# if len(selected_tiles)>3:
		# 	return selected_tiles
		# =============

		# # 更新selected_position_space 
		# my_branch_and_bound.update_covered_position_space(selected_tile, selected_position_space)

		# update the sub_ops' position_space_update
		# update_position_space_for_ops(all_subops, selected_tile)
		# update_position_space_for_ops(all_TC_ops, selected_tile)
		# update_position_space_for_ops([new_op], selected_tile)

		# 等到做完了local search之后一起更新=====================
		# update_position_space_for_ops_given_tiles(all_subops+all_TC_ops+[new_op], selected_tile)
		# =====================================================

		round_i+=1
		
		# print(f"Total nnz left: {tot_nnz/op.idx_lens[1]}")
		# # print(f"Select tile nnz: {selected_tile.nnz_when_selected/op.idx_lens[1]}")
		# print(f"Select tile nnz: { sum([t.nnz_when_selected/t.j_num for t in selected_tile]) }")
		# if sum([t.nnz_when_selected for t in selected_tile]) == 0: # selected_tile.nnz_when_selected == 0:
		# 	# print(f"selected_tile.avg_cost: {selected_tile.avg_cost}")
		# 	print("selected_tile nnz_uncovered is 0 when selected.")
		# 	print(selected_tile[0].get_key())
		# 	return all_subops, next_level_subops, selected_tiles, tile_dict, new_op


		print(f"Select tile nnz: { sum([t.nnz_uncovered/t.j_num for t in selected_tile]) }")
		if sum([t.nnz_uncovered for t in selected_tile]) == 0: # selected_tile.nnz_when_selected == 0:
			# print(f"selected_tile.avg_cost: {selected_tile.avg_cost}")
			print("selected_tile nnz_uncovered is 0 when selected.")
			print(selected_tile[0].get_key())
			return all_subops, next_level_subops, selected_tiles, tile_dict, new_op, selected_tile, all_TC_ops

		print(f"Already select {round_i} round tiles")

		# update
		# 调整为local search之后再统一更新，否则结果会不准
		# tot_nnz = tot_nnz - sum([t.nnz_when_selected / t.j_num for t in selected_tile]) * op.idx_lens[1]
		# print(f"tot_nnz now: {tot_nnz/op.idx_lens[1]}")
		# if tot_nnz <= 0:
		# 	break


		# ================try to find good tiles around the selected tile=============================================
		# FOR DEBUG-----------
		# if len(all_TC_ops)!=0:
		# 	second_best_pred_avg_cost, best_pred_avg_cost = float('inf'), 1/selected_tile.nnz_when_selected
		# --------------------

		good_tiles = my_branch_and_bound.local_search(op, 
			selected_tile[0], second_best_pred_avg_cost, best_pred_avg_cost, 
			ori_tot_nnz, SM_num, dsize, tile_dict, is_tb_tile, 
			max_bucket_size, 
			nnzs_dict, pos_is_dict, costs_dict, 
			TC_row_tile_dict, TC_row_costs, len(selected_tiles),
			reverse_idx_TC_op_row,
			penalty, TC_is, 
			max_avg_cost_diff = max_avg_cost_diff)

		# will select each tile in good_tiles----------------------
		# error_tile = my_fuse_formats.measure_tiles(op, good_tiles, tuning_hub, 
		# 	log_file=log_file, cuda_i = cuda_i, 
		# 	dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
		# good_tiles = [t for t in good_tiles if t.cost != float('inf')]
		# 改为不对local search出来的tile进行measure，而是直接设置他们measure之后应该具备的参数
		
		# 直接去掉这部分，改为在生成good tile的时候直接对这些参数赋值
		# for t in good_tiles:
		# 	t.cost = selected_tile[0].cost
		# 	t.best_tile_sizes = selected_tile[0].best_tile_sizes
		# 	t.best_params = selected_tile[0].best_params

		# 	# FOR DEBUG---------------
		# 	if t.pred_cost == None:
		# 		t.pred_cost = selected_tile[0].pred_cost
			# ------------------------

		
		# update selected_tile infor
		for t in good_tiles:
			update_tile_after_selection(t)
			# selected_tiles.append(t)

		selected_tiles = selected_tiles + good_tiles

		# # 更新selected_position_space 
		# my_branch_and_bound.update_covered_position_space(selected_tile, selected_position_space)

		# update the sub_ops' position_space_update
		# update_position_space_for_ops(all_subops, selected_tile)
		# update_position_space_for_ops_given_tiles(all_TC_ops, good_tiles)
		# update_position_space_for_ops_given_tiles(all_subops+all_TC_ops+[new_op], good_tiles)
		try:
			# 要一口气处理所有op，因为TC good tile的csr是从op的position space中直接提取的，而不是整合tile的position space
			# 但是因为good tile中一定会包含selected tile，所以可以不写selected_tile
			update_position_space_for_ops_given_tiles(all_subops+all_TC_ops+[new_op], good_tiles)
			# update_position_space_for_ops_given_tiles(all_TC_ops, good_tiles)
			print("finish update ops")
		except Exception as e:
			print(e)
			print("error here")
			return all_subops, all_TC_ops, good_tiles, selected_tile, new_op, selected_tiles

		# update
		# tot_nnz = tot_nnz - sum([t.nnz_when_selected / t.j_num for t in good_tiles]) * op.idx_lens[1]
		tot_nnz = new_op.position_space_update.nnz * op.idx_lens[1]
		print(f"tot_nnz now: {tot_nnz/op.idx_lens[1]}")
		if tot_nnz <= 0:
			break		
		# ==============================================================================================



		# try to find next level formats
		# <jingzhi>@response: support any level num limitation
		assert (max_level_num>1 and os.environ['single_level']=='False') or (max_level_num==1 and os.environ['single_level']=='True'), 'Wrong level num setting.'
		# next_level_condition = (level_num < float('inf')) and (get_template_str(selected_tile[0].op) in ["TensorCore_template", "TC_sddmm"])
		next_level_condition = (level_num < max_level_num) and (get_template_str(selected_tile[0].op) in ["TensorCore_template", "TC_sddmm"])
		if os.environ['single_level']=='True':
			next_level_condition = (level_num < 1)

		# if (level_num < float('inf')) and (get_template_str(selected_tile[0].op)=="TensorCore_template"): # 2:
		# if (level_num < 1): # 2:
		if next_level_condition:
			# <jingzhi>@response: we only update the level_num when we really generate new level tiles
			level_num += 1

			# return new_op, selected_tile

			# first update the position space of new_op ==> has been processed together with other ops
			# update_position_space_for_ops([new_op], selected_tile)
			# update_position_space_for_ops_given_tiles([new_op], good_tiles)

			new_op = update_op_after_tile_selection(new_op)

			print(f"new op nnz: {new_op.position_space_update.getnnz()}")

			# =============================
			next_level_subops, global_op_id = my_branch_and_bound.gen_full_next_level_subops(
				TC_is,
				global_op_id, new_op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
				reorder_k_by_nnz = reorder_k_by_nnz, 
				is_tb_tile = is_tb_tile)
			# 尝试一下每次都只从next level tiles中找tile，而不保留之前的tile
			all_subops = next_level_subops
			next_level_subops = list()
			tile_dict = dict()
			nnzs_dict, pos_is_dict, costs_dict = dict(), dict(), dict()
			# =============================

			# 以下是原有函数，暂时注释掉-----------
			# next_level_subops, global_op_id = my_branch_and_bound.gen_next_level_subops(
			# 	global_op_id, new_op, run_params, cand_ops, 
			# 	max_bucket_size = max_bucket_size, is_tb_tile = is_tb_tile)

			# 我们没有必要再加上tensor core based 的next level的tile，因为tensor core based的没有next level。因为其不能reorder index。
			# 但是其实只是不能reorder index i

			# 必须要保证每个sub_op都设置了相应的position space
			# 暂时不考虑TC tiles
			# 其实不需要这样，因为在gen_next_level_subops里面已经调用了gen_position_space_for_area了。
			# tmp_subops = list()
			# for sub_op in next_level_subops:
			# 	if (get_template_str(sub_op) == 'TensorCore_template'):
			# 		continue
			# 	gen_position_space_for_area(sub_op, 0)
			# 	tmp_subops.append(sub_op)
			# next_level_subops = tmp_subops

	return selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row, level_num





def get_best_avg_costs_for_TC_rows(selected_tiles, reverse_idx_TC_op_row, best_avg_costs, dsize):
	'''
	The best avg costs for each TC row is stored in the given best_avg_costs.
	'''
	penalty = 1
	for t in selected_tiles:
		if get_template_str(t.op) == "TensorCore_template":
			# avg_cost = my_cost_model.get_benchmark_cost_("TensorCore_template", t.tile_sizes, dsize, True, penalty)/t.nnz_when_selected
			avg_cost = t.pred_cost/t.nnz_when_selected
			best_avg_costs[t.tile_pos[0]] = min(best_avg_costs[t.tile_pos[0]], avg_cost)
		else:
			avg_cost = t.pred_cost / t.nnz_when_selected
			for ori_row in t.i_vals:
				if ori_row < len(reverse_idx_TC_op_row):
					TC_row_i = reverse_idx_TC_op_row[ori_row]
					best_avg_costs[TC_row_i//16] = min(best_avg_costs[TC_row_i//16], avg_cost)



def get_tot_costs_for_TC_rows(selected_tiles, reverse_idx_TC_op_row, tot_costs, dsize):
	'''
	The best avg costs for each TC row is stored in the given best_avg_costs.
	'''
	penalty = 1
	for t in selected_tiles:
		if get_template_str(t.op) == "TensorCore_template":
			tot_costs[t.tile_pos[0]] = tot_costs[t.tile_pos[0]] + t.pred_cost
			# tot_costs[t.tile_pos[0]] = tot_costs[t.tile_pos[0]] + my_cost_model.get_benchmark_cost_("TensorCore_template", t.tile_sizes, dsize, False, penalty)
		else:
			avg_cost = t.pred_cost / t.nnz_when_selected
			for i, ori_row in enumerate(t.i_vals):
				if ori_row < len(reverse_idx_TC_op_row):
					TC_row_i = reverse_idx_TC_op_row[ori_row]
					tot_costs[TC_row_i//16] = tot_costs[TC_row_i//16] + t.position_space_when_selected.getnnz(axis=1)[i]*avg_cost*t.j_num




def estimate_TC_tile_penalty(op, dsize, run_params, max_bucket_size, cache_set, kernel_tile_size_options,
	# only_TC, only_ELL, 
	reorder_k_by_nnz = True,
	# max_avg_cost_diff = float('inf'), 
	use_faster_tuner = False, log_file="log_hub/log.py", cuda_i = 3, 
	dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
	# 用来估计TC tile在优化后能达到满载性能的百分之几；感觉是不是能和TC tile的有效密度相关
	# 应该不是，因为密度分布可能不均匀；可以先从全ELL tile开始，判断有哪些TC行有选择TC的可能，根据密度来判断？
	# 可以比较全ELL tile方案和全TC tile 方案在每个TC行的最好avg cost，以此估计最终含TC行的数量。
	'''
	此处的参数和 greedy_search_use_cost_model_lower_bound 函数一样，但是少了 only_TC, only_ELL, max_avg_cost_diff。
	'''
	penalty = 1

	# 感觉这个地方的算法有点问题，不应该设置max_avg_cost_diff = float('inf')，因为这样得到的结果可能并不是最好的pure TC format
	max_avg_cost_diff = 0.2 # float('inf')
	only_TC, only_ELL = True, False
	selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row = greedy_search_use_cost_model_lower_bound(
		op, dsize, run_params, max_bucket_size, cache_set, 
		kernel_tile_size_options,
		only_TC, only_ELL,
		penalty,
		reorder_k_by_nnz = reorder_k_by_nnz,
		max_avg_cost_diff = max_avg_cost_diff, 
		use_faster_tuner=use_faster_tuner, 
		log_file=log_file, cuda_i = cuda_i, 
		dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
	old_selected_tiles_0 = selected_tiles
	selected_tiles_TC = my_branch_and_bound.post_process_for_withdraw(op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)
	
	# 
	# 对于ELL tiles而言，设置max_avg_cost_diff = float('inf') 没关系，因为搜索空间里的ELL 的cover范围没什么重叠，即pure ELL format唯一。
	max_avg_cost_diff = float('inf')
	only_TC, only_ELL = False, True
	op.position_space_update = op.inps[0].copy()
	selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row = greedy_search_use_cost_model_lower_bound(
		op, dsize, run_params, max_bucket_size, cache_set, 
		kernel_tile_size_options,
		only_TC, only_ELL,
		penalty,
		reorder_k_by_nnz = reorder_k_by_nnz,
		max_avg_cost_diff = max_avg_cost_diff, 
		use_faster_tuner=use_faster_tuner, 
		log_file=log_file, cuda_i = cuda_i, 
		dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
	old_selected_tiles_0 = selected_tiles
	selected_tiles_ELL = my_branch_and_bound.post_process_for_withdraw(op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)

	reverse_idx = reverse_idx_TC_op_row

	# we consider the total cost of each TC row as well
	tot_TC = np.full(len(TC_row_costs), 0.0)
	tot_ELL = np.full(len(TC_row_costs), 0.0)
	get_tot_costs_for_TC_rows(selected_tiles_TC, reverse_idx, tot_TC, dsize)
	get_tot_costs_for_TC_rows(selected_tiles_ELL, reverse_idx, tot_ELL, dsize)



	best_avg_costs_TC = np.full(len(TC_row_costs), float('inf'))
	best_avg_costs_ELL = np.full(len(TC_row_costs), float('inf'))

	cand_ops, global_op_id = gen_candidate_formats(op, max_bucket_size, kernel_tile_size_options, 
		reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = 0)

	all_TC_ops = list()
	for sub_op in cand_ops:
		gen_position_space_for_area(sub_op, 0)
		if (get_template_str(sub_op) == 'TensorCore_template'):
			all_TC_ops.append(sub_op)

	my_branch_and_bound.get_best_avg_cost_TC_by_rows(all_TC_ops[0], dsize, True, float('inf'), best_avg_costs_TC, penalty, max_bucket_size)

	# compare the best avg cost of selected_tiles_TC and selected_tiles_ELL in each TC row.
	# get_best_avg_costs_for_TC_rows(selected_tiles_TC, reverse_idx, best_avg_costs_TC, dsize)
	get_best_avg_costs_for_TC_rows(selected_tiles_ELL, reverse_idx, best_avg_costs_ELL, dsize)


	# estimate the number of TC rows where TC tiles perform better than ELL tiles
	# num_TC_rows = max(sum(best_avg_costs_TC<best_avg_costs_ELL), sum(tot_TC<tot_ELL))
	num_TC_rows = sum(np.logical_or(best_avg_costs_TC<best_avg_costs_ELL, tot_TC<tot_ELL))
	penalty = min(num_TC_rows / 108 / 64, 1)
	print(f"num_TC_rows: {num_TC_rows}  penalty: {penalty} [ TC num (atomic vs non-atomic): {sum(best_avg_costs_TC<best_avg_costs_ELL), sum(tot_TC<tot_ELL)} ]")
	return penalty





def estimate_max_bucket_size(op):
	'''
	Estimate max_bucket_size for ELL tiles given op.
	'''
	thread_num = 256 # Suppose the default thread number of ELL tiles is 256.
	tot_thread_per_SM = 2048 # in A100 GPU
	blk_num_per_SM = tot_thread_per_SM / thread_num # how many thread blocks can run on an SM at the same time
	thread_i = 8 # Suppose 8 tile i indices are bound to thread idx
	SM_num = 108

	# 此处最小的max bucket size选择为4，而不是2，因为在citeseer上自动找到为2，但是可能引起的atomic次数过多，反倒不如设成4的时候效果好。
	cand_workloads = [8*(2**i) for i in range(2, 6)][::-1]
	for cand in cand_workloads:
		if op.inps[0].nnz / (cand/2) / SM_num >= 0.5*blk_num_per_SM: # 此处用0.5因为我们观察到occupancy是0.5的时候已经能达到较高效果。
			return cand//thread_i
	# return the minimum cand anyway
	return cand//thread_i





def get_TC_tile_for_benchmark(tile_sizes, op,
	cuda_i, dtype, 
	max_bucket_size, kernel_tile_size_options, TC_k_notsorted, reorder_k_by_nnz,

	):
	'''
	INPUT: 
		tile_sizes: the TC tile sizes. only accept one TC tile shape candidate
		op:			the op to be optimized
	OUTPUT:
		selected tiles.
	'''
	os.environ['MyFileID'] = str(cuda_i)
	if dtype == "float16":
		os.environ['MydtypeIn'] = 'half'
		os.environ['MydtypeOut'] = 'half'
	elif dtype == "float32":
		os.environ['MydtypeIn'] = 'float'
		os.environ['MydtypeOut'] = 'float'

	# global global_op_id
	# cand_ops, global_op_id = gen_candidate_formats(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
	# 	reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = 0)

	# TC_op = None
	# for sub_op in cand_ops:
	# 	if (get_template_str(sub_op) in ['TC_sddmm']):
	# 		gen_position_space_for_area(sub_op, 0)
	# 		print(f"sub_op {sub_op.op_id}  initial nnz: {sub_op.position_space_update.nnz}", flush=True)
	# 		TC_op = sub_op
	# 		break

	# TC_row_num = math.ceil(TC_op.position_space_update.shape[0] / tile_sizes[0][1])
	# good_poses = list()
	# if TC_op.TC_k_notsorted:
	# 	good_poses = [ (i, 0, j) for i in range(TC_row_num) for j in TC_op.blk_ks[tile_sizes][i] ]
	# else:
	# 	good_poses = [ (i, 0, j) for i in range(TC_row_num) for j in range(TC_op.blk_nums[tile_sizes][i]) ]

	# print(len(good_poses), good_poses[:2])
	
	TC_op = op
	TC_op.loop_protocals = {0: 'uuu'}
	gen_position_space_for_area(TC_op, 0)
	TC_row_num = math.ceil(TC_op.idx_lens[0] / tile_sizes[0][1])
	k_vals = np.arange(TC_op.idx_lens[2])
	TC_op.k_vals = [k_vals for i in range(TC_row_num)]
	good_poses = [ (0, 0, 0) ]

	params = {"mma_shape_str":"m16n16k16"}
	good_tiles = [ComputeTile(TC_op, tile_sizes, tile_pos, params) for tile_pos in good_poses]
	for t in good_tiles:
		t.is_atomic_tile = False
		t.pred_cost = 1
		t.cost = 1
		t.best_tile_sizes = t.tile_sizes
		t.best_params = t.params
		t.position_space_when_selected = t.uncovered_position_space
		t.nnz_when_selected = t.nnz_uncovered
		# 此时的uncovered_position_space应该是正确的。

	return good_tiles







def get_pure_TC_formats(tile_sizes, op,
	cuda_i, dtype, 
	max_bucket_size, kernel_tile_size_options, TC_k_notsorted, reorder_k_by_nnz,

	):
	'''
	INPUT: 
		tile_sizes: the TC tile sizes. only accept one TC tile shape candidate
		op:			the op to be optimized
	OUTPUT:
		selected tiles.
	'''
	os.environ['MyFileID'] = str(cuda_i)
	if dtype == "float16":
		os.environ['MydtypeIn'] = 'half'
		os.environ['MydtypeOut'] = 'half'
	elif dtype == "float32":
		os.environ['MydtypeIn'] = 'float'
		os.environ['MydtypeOut'] = 'float'

	global global_op_id
	cand_ops, global_op_id = gen_candidate_formats(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
		reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = 0)

	TC_op = None
	for sub_op in cand_ops:
		if (get_template_str(sub_op) in ['TC_sddmm']):
			gen_position_space_for_area(sub_op, 0)
			print(f"sub_op {sub_op.op_id}  initial nnz: {sub_op.position_space_update.nnz}", flush=True)
			TC_op = sub_op
			break

	TC_row_num = math.ceil(TC_op.position_space_update.shape[0] / tile_sizes[0][1])
	good_poses = list()
	if TC_op.TC_k_notsorted:
		good_poses = [ (i, 0, j) for i in range(TC_row_num) for j in TC_op.blk_ks[tile_sizes][i] ]
	else:
		good_poses = [ (i, 0, j) for i in range(TC_row_num) for j in range(TC_op.blk_nums[tile_sizes][i]) ]

	print(len(good_poses), good_poses[:2])
	
	params = {"mma_shape_str":"m16n16k16"}
	good_tiles = [ComputeTile(TC_op, tile_sizes, tile_pos, params) for tile_pos in good_poses]
	for t in good_tiles:
		t.is_atomic_tile = False
		t.pred_cost = 1
		t.cost = 1
		t.best_tile_sizes = t.tile_sizes
		t.best_params = t.params
		t.position_space_when_selected = t.uncovered_position_space
		t.nnz_when_selected = t.nnz_uncovered
		# 此时的uncovered_position_space应该是正确的。

	return good_tiles