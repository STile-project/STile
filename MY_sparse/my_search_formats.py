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

# supporse we already know the cost of each computation tile
# we can temporarily assume we know the concrete implementation of a compute tile.
# cand_ops, cost_dict, tot_kernels_to_measure = gen_cand_compute_tiles(cand_ops, log_file="log_hub/log.py", cuda_i = 3, sub_op_rng=[-1, float('inf')])






global_op_id = 0







def update_position_space_for_ops(sub_ops, selected_tile):

	op = selected_tile.op
	if op.op_type == 'spmm':
		unfold_covered_csr = transform_covered_position_space_to_ori_space(selected_tile)
		# 
		for sub_op in sub_ops:

			tmp_csr = unfold_covered_csr[sub_op.idx_values_list[0][0],:][:,sub_op.idx_values_list[0][2]]
			if get_template_str(sub_op) == 'sparse_template_ell':
				tmp_csr = tmp_csr[sub_op.hyb_new_rows[0],:]
			

			sub_op.position_space_update = sub_op.position_space_update - tmp_csr.multiply(sub_op.position_space_update)

			sub_op.position_space_update = sub_op.position_space_update.tocsr()

			sub_op.position_space_update.eliminate_zeros()

			# if get_template_str(sub_op) == 'TensorCore_template':
			# 	if sub_op.position_space_update[48,16]==0:
			# 		print("\nWHEN UPDATE BY BEST TILE\n", selected_tile.get_key(), tmp_csr[48,16], sub_op.position_space_update[48,16])




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


	return csr[ind_i,:][:,ind_j]







def update_position_space_for_ops_given_tiles(sub_ops, selected_tiles):

	if len(selected_tiles) == 0:
		return

	op = selected_tiles[0].op
	if op.op_type in ['spmm', 'sddmm']:
		# =======================================================================================

		unfold_covered_csr = None
		if get_template_str(op) in ['TensorCore_template', 'TC_sddmm']:
			sorted_tiles = sorted(selected_tiles, key=lambda t: t.tile_pos) 
			unfold_covered_csr = merge_csrs_TC_tile(sorted_tiles)
			print("dummy_csr.nnz: ", unfold_covered_csr.nnz)
		elif get_template_str(op) == 'sparse_template_ell' :
			dummy_tile = ComputeTile(op, 
				((None, op.position_space_update.shape[0]), 
					(None, selected_tiles[0].tile_sizes[1][1]), 
					(None, op.position_space_update.shape[1])), (0, 0, 0), selected_tiles[0].params)
			dummy_indptr = np.zeros( op.position_space_update.shape[0] + 1 )
			tot, last_row = 0, 0
			sorted_tiles = sorted(selected_tiles, key=lambda t: t.tile_pos) 
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
			sorted_tiles = sorted(selected_tiles, key=lambda t: t.tile_pos) 
			unfold_covered_csr = merge_csrs_1D_SDDMM_tile(sorted_tiles)
			print("dummy_csr.nnz: ", unfold_covered_csr.nnz)


		assert (unfold_covered_csr.nnz == 0) or (max(unfold_covered_csr.data) == 1)
		print(f"unfold_covered_csr: {unfold_covered_csr.nnz}")
		# 

		for sub_op in sub_ops:

			tmp_csr = unfold_covered_csr[sub_op.idx_values_list[0][0],:][:,sub_op.idx_values_list[0][2]]
			if get_template_str(sub_op) == 'sparse_template_ell':
				# tmp_csr = tmp_csr[sub_op.hyb_new_rows[0],:]
				indptr = np.cumsum(np.concatenate( ([0], tmp_csr.getnnz(axis=1)[sub_op.hyb_new_rows[0]]) ))
				indices = np.concatenate([ tmp_csr.indices[ tmp_csr.indptr[i]:tmp_csr.indptr[i+1] ]  for i in sub_op.hyb_new_rows[0] ])
				data = np.ones(len(indices))
				tmp_csr = scipy.sparse.csr_matrix(
						(data, indices, indptr), 
						shape=sub_op.position_space_update.shape)




			print(f"before update: {sub_op.position_space_update.nnz}")
			sub_op.position_space_update = sub_op.position_space_update - tmp_csr.multiply(sub_op.position_space_update)
			print(sub_op.position_space_update.nnz)


			sub_op.position_space_update = sub_op.position_space_update.tocsr()

			sub_op.position_space_update.eliminate_zeros()

			if get_template_str(sub_op) == '1D_sddmm':

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

		









def estimate_tile_costs(tiles):
	'''
	Estimate the tile costs based on the roofline model.
	'''
	for tile in tiles:
		_, tile_sizes, tile_pos, params = tile.get_key(as_dict_key=False)
		tp = my_cost_model.cost_tb(tile.op, tile_sizes, tile_pos, params)
		cost = my_cost_model.compute_cost_tb(tile.op, tile_sizes, tile_pos, params) / tp
		tile.pred_cost = cost
		tile.set_pred_avg_cost()



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

	selected_tile.uncovered_position_space = None
	selected_tile.nnz_uncovered = 0	

	# update the pred_avg_cost and the avg_cost
	selected_tile.set_avg_cost()
	selected_tile.set_pred_avg_cost()



def greedy_search_use_cost_model_lower_bound(op, dsize, run_params, max_bucket_size, cache_set, kernel_tile_size_options,
	only_TC, only_ELL, 
	penalty, TC_k_notsorted,
	reorder_k_by_nnz = True,
	max_avg_cost_diff = float('inf'), use_faster_tuner = False, log_file="log_hub/log.py", cuda_i = 3, 
	dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
	'''
	Select the computation tiles greedily.
	Input: 
		the original op to be optimized.
	Output:
		the selected tiles.
	'''
	# tensor_core_cost_dict = dict()

	os.environ['MyFileID'] = str(cuda_i)
	if dtype == "float16":
		os.environ['MydtypeIn'] = 'half'
		os.environ['MydtypeOut'] = 'half'
	elif dtype == "float32":
		os.environ['MydtypeIn'] = 'float'
		os.environ['MydtypeOut'] = 'float'

	is_tb_tile = True

	tuning_hub = dict() #

	# 1. get level 1 possible sub_ops
	global global_op_id
	cand_ops, global_op_id = gen_candidate_formats(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
		reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = 0, gen_TC_formats = (not only_ELL))
	



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


	all_subops, next_level_subops, all_TC_ops = list(), list(), list()
	for sub_op in cand_ops:
		gen_position_space_for_area(sub_op, 0)
		print(f"sub_op {sub_op.op_id}  initial nnz: {sub_op.position_space_update.nnz}")
		if (get_template_str(sub_op) in ['TensorCore_template', 'TC_sddmm']):
			all_TC_ops.append(sub_op)
			continue

		# ====================

		if (sub_op.op_type == 'spmm') and (get_template_str(sub_op) != 'sparse_template_ell'):
			continue
		# ====================

		# gen_position_space_for_area(sub_op, 0)
		all_subops.append(sub_op)



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


	TC_row_num = math.ceil(op.inps[0].shape[0] / 16)
	TC_row_tile_dict = {i:list() for i in range(TC_row_num)}
	TC_row_costs = np.full(TC_row_num, 0.0) 
	TC_is = np.full(op.inps[0].shape[0], False)


	if len(all_TC_ops) > 0:
		selected_tiles = my_branch_and_bound.search_TC_tiles_only_by_rows(
			all_TC_ops[0], dsize, is_tb_tile, best_possible_ELL, ori_tot_nnz, all_TC_ops, all_subops, new_op, max_avg_cost_diff, 
			TC_row_tile_dict, TC_row_costs, penalty, max_bucket_size, TC_is)

		if op.op_type in ['spmm', 'sddmm']:
			tot_nnz = all_TC_ops[0].position_space_update.nnz * op.idx_lens[1]


		if (os.environ['single_level']!='True'):
			new_op = update_op_after_tile_selection(new_op)

			print(f"new op nnz: {new_op.position_space_update.getnnz()}")

			# =============================
			next_level_subops, global_op_id = my_branch_and_bound.gen_full_next_level_subops(
				TC_is,
				global_op_id, new_op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
				reorder_k_by_nnz = reorder_k_by_nnz, 
				is_tb_tile = is_tb_tile)

			all_subops = next_level_subops
			next_level_subops = list()
			tile_dict = dict()
			nnzs_dict, pos_is_dict, costs_dict = dict(), dict(), dict()



	while True:
		if tot_nnz <= 0:
			break

		# select best
		selected_tile = None
		good_tiles = list()


		second_best_pred_avg_cost = None

		# consider TC tiles========================
		# we check TC tiles first before we check ELL tiles, so that we can the avg cost of TC tiles as the start point for ELLs
		if len(all_TC_ops)!=0:
			best_tile_key_TC, best_cost_TC = my_branch_and_bound.search_best_TC_tile(all_TC_ops, dsize, 
				TC_row_tile_dict, TC_row_costs, penalty, max_bucket_size)

			# if best_tile_key_TC[2][2] == 1:
			# 	return all_TC_ops


			params = {"mma_shape_str":"m16n16k16"}
			# if best_tile_key_TC[1] == (16, 32, 32):
			# 	params = {"mma_shape_str":"m8n32k16"}
			is_atomic_tile = best_tile_key_TC[3]

			for tile_pos in best_tile_key_TC[2]:
				selected_tile = ComputeTile(cand_ops[best_tile_key_TC[0]], best_tile_key_TC[1], tile_pos, params)
				if is_atomic_tile or (op.op_type=='sddmm'):

					selected_tile.update_nnz_uncovered()
				selected_tile.pred_cost = best_cost_TC * selected_tile.nnz_uncovered 
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


		init_avg_cost = float('inf')
		# if selected_tile != None:
		# 	init_avg_cost = selected_tile.pred_avg_cost

		if len(good_tiles) > 0:
			init_avg_cost = good_tiles[0].pred_avg_cost


		selected_tile = good_tiles
		selected_TC_tile = good_tiles


		print(f"selected TC tiles: {selected_tile}")

		while True:	


			if init_avg_cost < best_possible_ELL:
				break


			if len(all_subops) == 0:
				selected_tile = selected_TC_tile
				break

			best_tile_key, second_best_pred_avg_cost = my_branch_and_bound.search_best_tile(
				op,
				all_subops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size, 
				nnzs_dict, pos_is_dict, costs_dict,
				best_pred_avg_cost = init_avg_cost)
			

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

					second_best_pred_avg_cost = min(init_avg_cost, \
						second_best_pred_avg_cost, second_best_pred_avg_cost_next_level)
					# 
					selected_tile = tile_dict_next_level[best_tile_key]
					level_num += 1
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


			if selected_tile == selected_TC_tile:
				break


			# compare best other tile and best TC tile
			error_tile = my_fuse_formats.measure_tiles(op, [selected_tile], tuning_hub, dsize, cache_set,
				max_bucket_size, use_faster_tuner = use_faster_tuner,
				log_file=log_file, cuda_i = cuda_i, 
				dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
			

			if error_tile != None:
				return error_tile, tile_dict
			# return error_tile, tile_dict

			if selected_tile.cost != float('inf'):
				break
			else:
				selected_tile = selected_TC_tile




		if selected_tile != selected_TC_tile:
			selected_tile = [selected_tile]


		if len(selected_tile) == 0:
			return op, all_subops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size, init_avg_cost, next_level_subops


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



		round_i+=1
		


		print(f"Select tile nnz: { sum([t.nnz_uncovered/t.j_num for t in selected_tile]) }")
		if sum([t.nnz_uncovered for t in selected_tile]) == 0: # selected_tile.nnz_when_selected == 0:
			# print(f"selected_tile.avg_cost: {selected_tile.avg_cost}")
			print("selected_tile nnz_uncovered is 0 when selected.")
			print(selected_tile[0].get_key())
			return all_subops, next_level_subops, selected_tiles, tile_dict, new_op, selected_tile, all_TC_ops

		print(f"Already select {round_i} round tiles")



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


		
		# update selected_tile infor
		for t in good_tiles:
			update_tile_after_selection(t)
			# selected_tiles.append(t)

		selected_tiles = selected_tiles + good_tiles

		try:

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
		next_level_condition = (level_num < float('inf')) and (get_template_str(selected_tile[0].op) in ["TensorCore_template", "TC_sddmm"])
		if os.environ['single_level']=='True':
			next_level_condition = (level_num < 1)

		# if (level_num < float('inf')) and (get_template_str(selected_tile[0].op)=="TensorCore_template"): # 2:
		# if (level_num < 1): # 2:
		if next_level_condition:
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

			all_subops = next_level_subops
			next_level_subops = list()
			tile_dict = dict()
			nnzs_dict, pos_is_dict, costs_dict = dict(), dict(), dict()
			# =============================



	return selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row


def save_selected_tiles(selected_tiles, file_name):
	with open(file_name, 'a') as f:
		f.write('NEW ROUND\n')
		for tile in selected_tiles:
			data = {'info': tile.get_key(), 'op': tile.op.get_key(), 
			'best_tile_sizes': tile.best_tile_sizes,
			'best_params': tile.best_params,
			'cost': tile.cost, 'nnz': int(tile.nnz_when_selected)}
			json.dump(data, f)
			f.write('\n')
		f.write('\n\n')





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
	penalty = 1

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

	cand_workloads = [8*(2**i) for i in range(2, 6)][::-1]
	for cand in cand_workloads:
		if op.inps[0].nnz / (cand/2) / SM_num >= 0.5*blk_num_per_SM: 
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

	return good_tiles