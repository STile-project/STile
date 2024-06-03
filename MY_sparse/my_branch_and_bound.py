from gen_formats_v2 import *

import my_cost_model

from fractions import Fraction
import functools

import os







def get_valid_tile_pos_list_for_ELL(sub_op, tile_sizes, max_bucket_size):
	'''
	Return the valid tile positions for the given sub_op and tile_sizes, so that we do not need to call is_valid_tile().
	'''
	# if sub_op.hyb_short_begin_row == None:
	# 	print("This op is wrong!!!!!", sub_op.get_key())

	if sub_op.op_type == 'spmm':
		if sub_op.loop_protocals[0][2] == 'p':
			# 先求在last bucket 之外的部分的合法tile pos
			targets = [math.prod(tile_sizes[2][1:]), (math.prod(tile_sizes[2][1:])//2)]

			target_is = [np.searchsorted(sub_op.hyb_getnnz_minus, -target) + sub_op.hyb_short_begin_row \
				for target in targets] # side=left --> "a[i-1] < v <= a[i]"
			
			list1 = list(range(math.ceil(target_is[0]/math.prod(tile_sizes[0][1:])), 
							(target_is[1]-1)//math.prod(tile_sizes[0][1:]) + 1))

			# FOR DEBUG 在所有tile sizes都和sparsetir相同之后，此处可能会有一些覆盖区域的空隙，因为list1的起点选的是ceil
			# print(tile_sizes, targets, target_is)
			# list1 = list1 + [list1[-1]+1]
			if len(list1)>0:
				# 同时还需要保证list1[-1]+1是一个有效值
				if (list1[-1]+1) * math.prod(tile_sizes[0][1:]) < len(sub_op.hyb_new_rows[0]):
					list1 = list1 + [list1[-1]+1]

			# print(math.ceil(target_is[0]/math.prod(tile_sizes[0][1:])), 
			# 				(target_is[1]-1)//math.prod(tile_sizes[0][1:]) + 1)
			if math.prod(tile_sizes[2][1:]) == max_bucket_size:
				# should contain the rows in the last bucket
				# 我们需要考虑两部分的position，即last bucket内的和非last bucket内的，最后合并列表
				list2 = list(range((sub_op.hyb_short_begin_row - 1)//math.prod(tile_sizes[0][1:]) + 1))
				# print( (sub_op.hyb_short_begin_row - 1)//math.prod(tile_sizes[0][1:]) + 1 )
				return [(i, 0, 0) for i in list2 + list1]
			else:
				return [(i, 0, 0) for i in list1]
		else:
			return get_tile_pos_list(sub_op, tile_sizes)






def merge_csrs_TC_tile_in_one_row(sorted_tiles, blk_csr):
	'''
	Generate the merged covered_position_space of the TC tiles sorted by their position.
	Do not transform the csr to be returned to the original position space.
	Assume: we assume the op of any TC tile is not kernel-tiled, and the k axis is not reordered. 
	Assume: all sorted tiles are from the same op.
	'''
	op = sorted_tiles[0].op
	(_, I), (_, J), (_, K) = sorted_tiles[0].tile_sizes

	poses = [t.tile_pos for t in sorted_tiles]
	i = poses[0][0]
	ks = None
	if op.op_type in ['spmm', 'sddmm']:
		ks = np.concatenate([range(int(K*pos[2]), min(int(K*(pos[2]+1)), len(op.k_vals[i]) )) for pos in poses ])
	csr = blk_csr[:, op.k_vals[i][ks] ]
	indices =  op.idx_values_list[0][2][ op.k_vals[i][ks][csr.indices] ]

	csr = scipy.sparse.csr_matrix(
				(csr.data, indices, csr.indptr), 
				shape=blk_csr.shape)

	return csr









def update_position_space_for_ops_given_csr(sub_ops, csr_left):
	'''
		根据一组select tile的position space when selected来更新该op的position space。
		这里我们采用的是scipy sparse csr matrix 的数据结构。
	'''
	op = sub_ops[0]

	k_axis = 2

	if op.op_type in ['spmm', 'sddmm']:
		# =======================================================================================
		# NOTE: 我们这里有一个假设，即我们所处理的所有position space里面的元素的可能值都只有0和1，不包含该矩阵原本的元素值。
		for sub_op in sub_ops:
			# 需要再把unfold_covered_csr转化成和sub_op的position space相一致的坐标的矩阵
			tmp_csr = csr_left[sub_op.idx_values_list[0][0],:][:,sub_op.idx_values_list[0][k_axis]]
			if get_template_str(sub_op) == 'sparse_template_ell':
				# tmp_csr = tmp_csr[sub_op.hyb_new_rows[0],:]
				indptr = np.cumsum(np.concatenate( ([0], tmp_csr.getnnz(axis=1)[sub_op.hyb_new_rows[0]]) ))
				indices = np.concatenate([ tmp_csr.indices[ tmp_csr.indptr[i]:tmp_csr.indptr[i+1] ]  for i in sub_op.hyb_new_rows[0] ])
				data = np.ones(len(indices))
				tmp_csr = scipy.sparse.csr_matrix(
						(data, indices, indptr), 
						shape=sub_op.position_space_update.shape)

			sub_op.position_space_update = tmp_csr.multiply(sub_op.position_space_update)
			print(sub_op.position_space_update.nnz)
			# 似乎更新之后，sub_op.position_space_update里面的0也会被同时清除，无所谓了
			sub_op.position_space_update = sub_op.position_space_update.tocsr()
			# 更新每个sub_op的position_space_update，之后还需要把里面新增的zeros全部都eliminate掉
			sub_op.position_space_update.eliminate_zeros()

			if get_template_str(sub_op) == '1D_sddmm':
				# 需要更新nnz_update_1D，nnz_data_1D，其实并不需要nnz_data_1D
				# 我们其实并不需要保证position_space_update的indices也是sorted的
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




def update_tile_after_selection(selected_tile, csr):
	'''
	Update the information of a tile after it is selected.
	'''
	# gen_updated_position_space_for_tile(selected_tile)
	op = selected_tile.op
	pos_i = selected_tile.tile_pos[0]
	if op.op_type in ['spmm', 'sddmm']:
		selected_tile.uncovered_position_space = csr[:, op.k_vals[pos_i][selected_tile.tile_k_rng[0]:selected_tile.tile_k_rng[1]+1] ]
	selected_tile.position_space_when_selected = selected_tile.uncovered_position_space
	# selected_tile.update_nnz_uncovered()
	if op.op_type in ['spmm']:
		selected_tile.nnz_uncovered = selected_tile.j_num * selected_tile.uncovered_position_space.getnnz()
	elif op.op_type == 'sddmm':
		# TODO SDDMM 此处的nnz_uncovered还需要在考虑一下，是乘k_num还是乘k_tot，因为好像在fuse formats的时候用的是乘k_tot----决定用k_tot
		selected_tile.nnz_uncovered = op.idx_lens[1] * selected_tile.uncovered_position_space.getnnz()
	selected_tile.nnz_when_selected = selected_tile.nnz_uncovered
	# print("THE NNZ UNCOVERED: ", selected_tile.nnz_when_selected)

	# print(selected_tile.tile_pos, selected_tile.position_space_when_selected.nnz)
	# print(selected_tile.tile_k_rng)
	# print(csr.getnnz(axis=0))
	
	# directly update the uncovered_position_space of selected_tile
	# 这两行的操作其实可以不要
	selected_tile.uncovered_position_space = None
	selected_tile.nnz_uncovered = 0	

	# update the pred_avg_cost and the avg_cost
	selected_tile.set_avg_cost()
	selected_tile.set_pred_avg_cost()




def get_best_avg_cost_TC_by_rows(sub_op, dsize, is_tb_tile, best_possible_ELL, best_avg_costs, penalty, max_bucket_size):
	'''
	This function is only called by " estimate_TC_tile_penalty ".
	It only gets the best TC tile (in terms of avg cost) for each TC row.
	'''
	selected_tiles = list()

	tile_sizes_idx = get_tile_sizes_idx(sub_op, None, is_tb_tile=is_tb_tile)
	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=is_tb_tile)

	# 先把tile_sizes 分组，让tile_i一样的tile sizes在一组
	tile_sizes_groups = dict()
	for tile_sizes in tile_sizes_list:
		if tile_sizes[0][1] not in tile_sizes_groups:
			tile_sizes_groups[tile_sizes[0][1]] = [tile_sizes]
		else:
			tile_sizes_groups[tile_sizes[0][1]].append(tile_sizes)	


	# 这里只考虑一个group的情况，我们在设计TC tile sizes的时候会使得只有一个group
	assert len(tile_sizes_groups) == 1

	# for tile_sizes in tile_sizes_list:
	for group in tile_sizes_groups.values():

		tile_sizes = group[0]
		row_num = len(sub_op.idx_values_list[0][0])
		blk_i_size = tile_sizes[0][1]
		row_blk_num = math.ceil(row_num / blk_i_size)
		csr = sub_op.position_space_update
		k_num = csr.shape[1]

		for blk_row_i in range(row_blk_num):
			# 每一个block row之间的搜索是独立的，更新position_space也只更新对应的部分，以此来缩短整体搜索时间
			blk_csr = csr[blk_row_i*blk_i_size:(blk_row_i+1)*blk_i_size, :]
			tot_nnz_blk_row = blk_csr.nnz
			# ori_tot_nnz = tot_nnz
			selected_tiles_this_row = list()

			# if blk_row_i == 14:
			# 	return sub_op, blk_csr, blk_row_i, dsize, tile_sizes_list, best_possible_ELL, selected_tiles

			while True:
				# print(f"1, {blk_row_i}, {blk_csr.indices}, {blk_csr.nnz}")
				# if blk_row_i == 0:
				# 	print(f"1, {blk_row_i}, {blk_csr.nnz}")
				if blk_csr.nnz <= 0:
					break

				best_tile_key, best_cost, unit_cost_dict = search_best_TC_tile_given_csr(sub_op, blk_csr, 
					blk_row_i, dsize, tile_sizes_list, best_possible_ELL, selected_tiles_this_row, penalty)
				if best_cost == float('inf'):
					break

				best_avg_costs[blk_row_i] = best_cost
				break






def search_TC_tiles_only_by_rows(sub_op, dsize, is_tb_tile, best_possible_ELL, tot_nnz, all_TC_ops, all_sub_ops, ori_op, max_avg_cost_diff, 
	TC_row_tile_dict, TC_row_costs, penalty, max_bucket_size, TC_is):
	'''
	这个函数假定我们考虑的TC op只有一个。 
	best_possible_ELL: ELL tile的最好pred avg cost是多少。
	'''
	selected_tiles = list()
	blk_csrs = list()
	ori_tot_nnz = tot_nnz

	tile_sizes_idx = get_tile_sizes_idx(sub_op, None, is_tb_tile=is_tb_tile)
	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=is_tb_tile)

	# 先把tile_sizes 分组，让tile_i一样的tile sizes在一组
	tile_sizes_groups = dict()
	for tile_sizes in tile_sizes_list:
		if tile_sizes[0][1] not in tile_sizes_groups:
			tile_sizes_groups[tile_sizes[0][1]] = [tile_sizes]
		else:
			tile_sizes_groups[tile_sizes[0][1]].append(tile_sizes)	


	# 这里只考虑一个group的情况，我们在设计TC tile sizes的时候会使得只有一个group
	assert len(tile_sizes_groups) == 1

	# for tile_sizes in tile_sizes_list:
	for group in tile_sizes_groups.values():

		tile_sizes = group[0]
		row_num = len(sub_op.idx_values_list[0][0])
		blk_i_size = tile_sizes[0][1]
		row_blk_num = math.ceil(row_num / blk_i_size)
		csr = sub_op.position_space_update
		k_num = csr.shape[1]

		for blk_row_i in range(row_blk_num):
			# 每一个block row之间的搜索是独立的，更新position_space也只更新对应的部分，以此来缩短整体搜索时间
			blk_csr = csr[blk_row_i*blk_i_size:(blk_row_i+1)*blk_i_size, :]
			tot_nnz_blk_row = blk_csr.nnz
			ori_tot_nnz = tot_nnz
			selected_tiles_this_row = list()

			print("init blk_csr.nnz: ", blk_csr.nnz)

			# if blk_row_i == 14:
			# 	return sub_op, blk_csr, blk_row_i, dsize, tile_sizes_list, best_possible_ELL, selected_tiles

			while True:
				# print(f"1, {blk_row_i}, {blk_csr.indices}, {blk_csr.nnz}")
				# if blk_row_i == 0:
				# 	print(f"1, {blk_row_i}, {blk_csr.nnz}")
				if blk_csr.nnz <= 0:
					break

				best_tile_key, best_cost, unit_cost_dict = search_best_TC_tile_given_csr(sub_op, blk_csr, 
					blk_row_i, dsize, tile_sizes_list, best_possible_ELL, selected_tiles_this_row, penalty)
				if best_cost == float('inf'):
					break

				# print(f"2, {blk_row_i}, {blk_csr.indices}, {blk_csr.nnz}")
				# 然后直接做local search-------------------------------------
				(op_id, tile_sizes, tile_poses, is_atomic_tile) = best_tile_key
				target_pred_avg_cost = best_cost * (1+max_avg_cost_diff)
				good_tiles = local_search_TC_tiles_given_csr(sub_op, blk_csr, blk_row_i, tile_sizes, 
					target_pred_avg_cost, dsize, 
					*(unit_cost_dict[tile_sizes]), selected_tiles_this_row)

				# print(f"3, {blk_row_i}, {blk_csr.indices}, {blk_csr.nnz}")
				# 处理good_tiles，更新blk_csr---------------------------------		

				# selected_tiles = selected_tiles + good_tiles # 不管有没有重复选TC tile，都应该全部存进去，否则可能有的位置的tile变空了
				update_tile_ref_csr = blk_csr
				if (sub_op.op_type == 'spmm') and (len(good_tiles) > 0) and (not good_tiles[0].is_atomic_tile): # 说明这次选了整行
					selected_tiles_this_row = good_tiles
					update_tile_ref_csr = csr[blk_row_i*blk_i_size:(blk_row_i+1)*blk_i_size, :]
				else:
					selected_tiles_this_row = selected_tiles_this_row + good_tiles

				# update selected_tile infor
				for t in good_tiles:
					update_tile_after_selection(t, update_tile_ref_csr) # 对于整行的TC tile，就会自动选择原始的blk_csr
					# selected_tiles.append(t)

				# update the sub_ops' position_space_update
				# 比如要一口气处理所有op，因为TC good tile的csr是从op的position space中直接提取的，而不是整合tile的position space
				# 但是因为good tile中一定会包含selected tile，所以可以不写selected_tile


				# print(f"4, {blk_row_i}, {blk_csr.indices}, {blk_csr.nnz}")
				sorted_tiles = sorted(good_tiles, key=lambda t: t.tile_pos) # 把tile按i轴位置排序 
				unfold_covered_csr = merge_csrs_TC_tile_in_one_row(sorted_tiles, blk_csr)
				blk_csr = blk_csr - unfold_covered_csr.multiply(blk_csr)

				blk_csr = blk_csr.tocsr()
				# 更新每个sub_op的position_space_update，之后还需要把里面新增的zeros全部都eliminate掉
				blk_csr.eliminate_zeros()

				print("blk_row_i: ", blk_row_i,  "blk_csr.nnz: ", blk_csr.nnz, "unfold_covered_csr nnz: ", unfold_covered_csr.nnz)


				if blk_csr.nnz <= 0:
					break

				# update
				# tot_nnz = tot_nnz - sum([t.nnz_when_selected / t.j_num for t in good_tiles]) * sub_op.idx_lens[1]
				# print(f"tot_nnz now: {tot_nnz/sub_op.idx_lens[1]}, blk_row_i: {blk_row_i}")

				tot_nnz = ori_tot_nnz - (tot_nnz_blk_row - blk_csr.nnz)
				print(f"tot_nnz now: {tot_nnz}, blk_row_i: {blk_row_i}")				

				# 这个判断没有必要，因为tot_nnz为0时一定有blk_csr.nnz为0
				# if tot_nnz <= 0:
				# 	break
			blk_csrs.append(blk_csr)
			selected_tiles = selected_tiles + selected_tiles_this_row
			if (sub_op.op_type == 'spmm') and (len(selected_tiles_this_row)>0) and (not selected_tiles_this_row[0].is_atomic_tile):
				TC_row_costs[blk_row_i] = -1
			else:
				TC_row_tile_dict[blk_row_i] = np.arange(len(selected_tiles) - len(selected_tiles_this_row), len(selected_tiles))
				TC_row_costs[blk_row_i] = sum([t.pred_cost for t in selected_tiles_this_row])
				print(f"\nlen(selected_tiles_this_row): {len(selected_tiles_this_row)}, TC_row_costs[{blk_row_i}]: {TC_row_costs[blk_row_i]}   tot cost: {sum([t.pred_cost for t in selected_tiles_this_row])}\n")
			
			if len(selected_tiles_this_row)>0:
				TC_is[ blk_row_i*blk_i_size:(blk_row_i+1)*blk_i_size ] = True
			# 
			# 我们不能提前退出，因为需要得到完整的blk_csr列表
			# if tot_nnz <= 0:
			# 	break			
		# 
		# 需要把所有的blk_csr整合成一个大的
		indptr = np.empty(csr.shape[0]+1)
		data = np.concatenate([blk_csr.data for blk_csr in blk_csrs])
		indices = np.concatenate([blk_csr.indices for blk_csr in blk_csrs])
		tot = 0
		for i in range(row_blk_num):
			indptr[i*blk_i_size:(i+1)*blk_i_size] = blk_csrs[i].indptr[:-1] + tot
			tot = tot + blk_csrs[i].indptr[-1]
		indptr[-1] = tot

		csr_left = scipy.sparse.csr_matrix(
				(data, indices, indptr), 
				shape=csr.shape)

		ind_i = np.argsort(sub_op.idx_values_list[0][0])

		# 把csr_left还原回原始op的in1的矩阵坐标
		csr_left = csr_left[ind_i,:]

		# 可以用这个csr_left更新别的op
		update_position_space_for_ops_given_csr(all_TC_ops + all_sub_ops + [ori_op], csr_left)

		return selected_tiles











# 这个版本允许TC tile的tile pos k是分数
# 同时，这个版本允许TC tile whole row的withdraw。其实本质上相当于我们的next level？不，withdraw和next level还是有区别的
def search_best_TC_tile_given_csr(sub_op, csr, blk_row_i, dsize, tile_sizes_list, best_possible_ELL, selected_tiles_this_row, penalty):
	'''
	Directly search the best TC tile given the csr.	
	默认csr只是一个block 行对应的csr。
	sub_op只用来提供一些类似op_type, template的信息。
	'''
	k_axis = 2

	best_tile_key, best_cost = None, float('inf')

	# 先把tile_sizes 分组，让tile_i一样的tile sizes在一组
	tile_sizes_groups = dict()
	for tile_sizes in tile_sizes_list:
		if tile_sizes[0][1] not in tile_sizes_groups:
			tile_sizes_groups[tile_sizes[0][1]] = [tile_sizes]
		else:
			tile_sizes_groups[tile_sizes[0][1]].append(tile_sizes)	


	unit_cost_dict = dict()

	# for tile_sizes in tile_sizes_list:
	for group in tile_sizes_groups.values():

		tile_sizes = group[0]
		# row_num = csr.shape[0]
		blk_i_size = tile_sizes[0][1]
		# row_blk_num = math.ceil(row_num / blk_i_size)
		k_num = csr.shape[1]
		# 
		nnzs_update_base = np.add.reduceat(
					csr.getnnz(axis=0)[ sub_op.k_vals[blk_row_i] ], 
					np.arange(0, k_num, 16)
					)

		tot_nnz_per_blk_row = np.sum(nnzs_update_base) #, axis=1)

		k_num_base = len(nnzs_update_base) # nnzs_update_base.shape[1]

		# 为了方便计算tile pos k为分数时的cost，此处nnzs_update_base 改为计算cumsum的版本
		nnzs_update_base_cumsum = np.concatenate( [[0], np.cumsum(nnzs_update_base)] )

		for tile_sizes in group:
			# find the tile position with the max nnz
			# nnzs_update_per_blk = np.add.reduceat(nnzs_update, np.arange(0, k_num, tile_sizes[2][1]), axis = 1)
			# nnzs_update_per_blk = np.add.reduceat(nnzs_update_base, np.arange(0, k_num_base, tile_sizes[2][1]//16)) #, axis = 1)

			# 考虑了分数的tile pos k，直接用np diff来计算每个position的nnz
			tmp = np.concatenate( [ nnzs_update_base_cumsum, np.full(tile_sizes[k_axis][1]//16-1, nnzs_update_base_cumsum[-1])] )
			# nnzs_update_per_blk = np.asarray([tmp[i+tile_sizes[2][1]//16] - tmp[i] for i in range(len(nnzs_update_base_cumsum)-1)])

			nnzs_update_per_blk = tmp[np.arange(tile_sizes[k_axis][1]//16, tile_sizes[k_axis][1]//16 + len(nnzs_update_base_cumsum)-1)] - tmp[:len(nnzs_update_base_cumsum)-1]

			# 我们临时在这里处理成不支持分数pos的TC tile的情况，用来获取ablation study的数据
			if os.environ['single_level']=='True':
				nnzs_update_per_blk = np.add.reduceat(nnzs_update_base, np.arange(0, k_num_base, tile_sizes[k_axis][1]//16)) #, axis = 1)

			# print(f"tile_sizes: {tile_sizes} nnzs_update_per_blk: {nnzs_update_per_blk}")

			# 此处 max_pos为分数
			# # max_pos = int(np.argpartition(np.concatenate(nnzs_update)*-1, 0)[0])
			# max_pos =  int(np.argmax(nnzs_update_per_blk))
			# # max_pos = (max_pos//len(nnzs_update_per_blk[0]) + blk_row_i, 0, max_pos%len(nnzs_update_per_blk[0]))
			# max_pos = (blk_row_i, 0, max_pos * Fraction(16, tile_sizes[2][1]))
			# tile = ComputeTile(sub_op, tile_sizes, max_pos, dict())
			# cost = my_cost_model.cost_tb_latency_given_tile(tile, dsize, None, None, True) #is_atomic is True
			# # nnz = nnzs[max_pos[0]][max_pos[2]]
			# tile.pred_cost = cost
			# # tile.update_nnz_uncovered()
			# # tile.set_pred_avg_cost()


			# 其实没必要生成tile
			max_pos =  int(np.argmax(nnzs_update_per_blk))
			cost = None
			if sub_op.op_type == 'spmm':
				cost = my_cost_model.get_benchmark_cost_("TensorCore_template", tile_sizes, dsize, True, penalty)
			elif sub_op.op_type == 'sddmm':
				cost = my_cost_model.get_benchmark_cost_("TC_sddmm", tile_sizes, dsize, False, penalty)
			unit_cost_dict[tile_sizes] = [None, cost]

			# if cost/nnz < best_cost:
			# if tile.pred_avg_cost < best_cost:
			# pred_avg_cost = cost/nnzs_update_per_blk[max_pos[2]]/tile.j_num

			pred_avg_cost = None
			if sub_op.op_type == 'spmm':
				pred_avg_cost = cost/nnzs_update_per_blk[max_pos]/tile_sizes[1][1]
			elif sub_op.op_type == 'sddmm':
				# TODO SDDMM: 此处计算avg cost是不是还得考虑在k轴这个tile能重复多少次？
				pred_avg_cost = cost*math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1]) /nnzs_update_per_blk[max_pos]/sub_op.idx_lens[1]

			# print(f"best single TC: {pred_avg_cost} cost: {cost}  nnz: {nnzs_update_per_blk[max_pos[2]]}  max_pos: {max_pos}")

			max_pos = (blk_row_i, 0, max_pos * Fraction(16, tile_sizes[k_axis][1]))
			if os.environ['single_level']=='True':
				max_pos = (blk_row_i, 0, max_pos)

			if (pred_avg_cost < best_cost) and (pred_avg_cost < best_possible_ELL):
				best_cost = pred_avg_cost # cost/nnz
				best_tile_key = (sub_op.op_id, tile_sizes, [max_pos], True)

				print("update best TC (single): ", best_cost, best_tile_key)


			if sub_op.op_type == 'sddmm':
				# 如果是sddmm，所有TC tile都是non-atomic的，不必考虑整行都是同样的TC tile的情况
				continue

			# 下面来处理这个tile把涉及到的行的所有nnz全都覆盖的情况，即打包一整行的TC tile的情况
			# 此处用到的csr应当是最原始的，没有选择过任何tile的csr
			non_empty_tile_nums = sub_op.blk_nums[tile_sizes][blk_row_i]
			
			# no_atomic_tile_cost = my_cost_model.cost_tb_latency_given_tile(tile, dsize, None, None, False) #is_atomic is False
			no_atomic_tile_cost = my_cost_model.get_benchmark_cost_("TensorCore_template", tile_sizes, dsize, False, penalty)

			unit_cost_dict[tile_sizes][0] = no_atomic_tile_cost

			# 看哪些整行TC tile的pred avg cost在允许范围之内
			# 计算平均cost的时候还是选择nnz_update

			# 处理whole row的情况，但是我们直接假设是withdraw的场景
			extra_cost = no_atomic_tile_cost * non_empty_tile_nums - sum([i.pred_cost for i in selected_tiles_this_row])

			if os.environ['no_withdraw']=='True':
				# 这个环境变量用在ablation study里面。此时我们不考虑whole row的带withdraw的实现，而是正常地计算whole row的 average cost
				extra_cost = no_atomic_tile_cost * non_empty_tile_nums

			pred_avg_costs = extra_cost / tot_nnz_per_blk_row / tile_sizes[1][1] # tile.j_num

			if (pred_avg_costs < best_cost) and (pred_avg_costs < best_possible_ELL):
				best_cost = pred_avg_costs # cost/nnz
				selected_poses = [(blk_row_i, 0, i) for i in range(int(non_empty_tile_nums)) ]
				if sub_op.TC_k_notsorted:
					selected_poses = [(blk_row_i, 0, i) for i in sub_op.blk_ks[tile_sizes][blk_row_i] ]
				best_tile_key = (sub_op.op_id, tile_sizes, selected_poses, False)

				print("update best TC (while line): ", best_cost, best_tile_key, "row ", blk_row_i, " non_empty_tile_num: ", non_empty_tile_nums)

	return best_tile_key, best_cost, unit_cost_dict











# 修改这个函数，能够支持更多种的TC tile sizes，这样的话，就需要我们不是比较nnz，而是我们估计出来的cost。
# 下面的这个函数，会对每个TC tile size比较两种类型TC tile，一种是覆盖了涉及到的行的所有nnz无atomic开销的；另一种是单个tile有atomic开销的。
# 这个版本支持分数的tile pos k
# 支持withdraw选择整行TC tile
def search_best_TC_tile_given_op(sub_op, dsize, 
	TC_row_tile_dict, TC_row_costs,
	penalty, max_bucket_size, 
	is_tb_tile = False,
	lower_bound_dict = dict(), 
	tile_dict = dict(), 
	# selected_position_space = dict()
	):
	'''
	Directly search the best TC tile given the op.	
	'''
	k_axis = 2

	tile_sizes_idx = get_tile_sizes_idx(sub_op, None, is_tb_tile=is_tb_tile)
	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=is_tb_tile)
	best_tile_key, best_cost = None, float('inf')

	# 先把tile_sizes 分组，让tile_i一样的tile sizes在一组
	tile_sizes_groups = dict()
	for tile_sizes in tile_sizes_list:
		if tile_sizes[0][1] not in tile_sizes_groups:
			tile_sizes_groups[tile_sizes[0][1]] = [tile_sizes]
		else:
			tile_sizes_groups[tile_sizes[0][1]].append(tile_sizes)	


	# for tile_sizes in tile_sizes_list:
	for group in tile_sizes_groups.values():

		tile_sizes = group[0]
		row_num = len(sub_op.idx_values_list[0][0])
		blk_i_size = tile_sizes[0][1]
		row_blk_num = math.ceil(row_num / blk_i_size)
		csr = sub_op.position_space_update
		k_num = csr.shape[1]
		# 

		# 支持分数tile pos k，直接修改其实的tile pos k
		non_empty_begin = np.asarray([ np.nonzero(csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ])[0][0]//16 \
			if csr[i*blk_i_size:(i+1)*blk_i_size, :].nnz>0 else 0\
			for i in range(row_blk_num)])

		if os.environ['single_level']=='True':
			non_empty_begin = non_empty_begin-non_empty_begin

		# nnzs_update_base = np.asarray([ np.add.reduceat(
		# 			csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ], 
		# 			np.arange(0, k_num, 16)
		# 			) \
		# 			for i in range(row_blk_num)])

		nnzs_update_base = np.asarray([ np.add.reduceat(
					np.concatenate([csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ], np.full(non_empty_begin[i]*16, 0)]), 
					np.arange(non_empty_begin[i]*16, k_num+non_empty_begin[i]*16, 16)
					) \
					for i in range(row_blk_num)])

		tot_nnz_per_blk_row = np.sum(nnzs_update_base, axis=1)

		# nnzs_update_base = np.add.reduceat(nnzs_update, np.arange(0, k_num, 16), axis = 1)
		k_num_base = nnzs_update_base.shape[1]

		# 为了支持分数tile pos
		# nnzs_update_base_cumsum = np.concatenate([np.full((row_blk_num, 1), 0), np.cumsum(nnzs_update_base, axis=1)], axis=1)

		for tile_sizes in group:
			# find the tile position with the max nnz
			# nnzs_update_per_blk = np.add.reduceat(nnzs_update, np.arange(0, k_num, tile_sizes[2][1]), axis = 1)
			nnzs_update_per_blk = np.add.reduceat(nnzs_update_base, np.arange(0, k_num_base, tile_sizes[k_axis][1]//16), axis = 1)

			max_pos =  int(np.argmax(nnzs_update_per_blk))
			max_pos = (max_pos//len(nnzs_update_per_blk[0]), 0, max_pos%len(nnzs_update_per_blk[0]))
			
			cost = None
			pred_avg_cost = None
			is_atomic = None
			if sub_op.op_type == 'spmm':
				is_atomic = True
				cost = my_cost_model.get_benchmark_cost_("TensorCore_template", tile_sizes, dsize, is_atomic, penalty)
				pred_avg_cost = cost/nnzs_update_per_blk[max_pos[0], max_pos[2]]/tile_sizes[1][1]
			elif sub_op.op_type == 'sddmm':
				is_atomic = False
				cost = my_cost_model.get_benchmark_cost_("TC_sddmm", tile_sizes, dsize, is_atomic, penalty)
				# TODO SDDMM: 关于此处的avg cost还需要重新确定一下
				pred_avg_cost = cost*math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1]) /nnzs_update_per_blk[max_pos[0], max_pos[2]]/sub_op.idx_lens[1]

			# 判断一下这一轮是否有必要生成新的ELL op，不行，这个方法还是测不了，删了这个功能吧
			# if (nnzs_update_per_blk > (cost / best_possible_ELL / tile.j_num)).any() \
			# 	and (nnzs_update_per_blk < tile.nnz_uncovered/tile.j_num/1.1).any():
			# 	no_ELL_op = True

			# if cost/nnz < best_cost:
			# max_pos = (max_pos[0], max_pos[1], max_pos[2]*Fraction(16, tile_sizes[2][1]))
			max_pos = (max_pos[0], max_pos[1], max_pos[2]+non_empty_begin[max_pos[0]]*Fraction(16, tile_sizes[2][1]))
			if pred_avg_cost < best_cost:
				best_cost = pred_avg_cost # cost/nnz
				best_tile_key = (sub_op.op_id, tile_sizes, [max_pos], is_atomic)

				print("update best TC (single): ", best_cost, best_tile_key)


			if sub_op.op_type == 'sddmm':
				continue

			# 下面来处理这个tile把涉及到的行的所有nnz全都覆盖的情况，即打包一整行的TC tile的情况
			# 此处用到的csr应当是最原始的，没有选择过任何tile的csr
			# csr = sub_op.position_space[0]  # 因为是完整覆盖整行，所以用的是position_space而不是position_space_update
			# # 
			# nnzs = [ csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ] for i in range(row_blk_num)]	
			# non_empty_tile_nums = np.ceil(np.count_nonzero(nnzs, axis=1) / tile_sizes[2][1])
			non_empty_tile_nums = sub_op.blk_nums[tile_sizes]
			
			# no_atomic_tile_cost = my_cost_model.cost_tb_latency_given_tile(tile, dsize, None, None, False) #is_atomic is False
			no_atomic_tile_cost = my_cost_model.get_benchmark_cost_("TensorCore_template", tile_sizes, dsize, False, penalty)

			# 看哪些整行TC tile的pred avg cost在允许范围之内
			# 计算平均cost的时候还是选择nnz_update
			# pred_avg_costs = no_atomic_tile_cost * non_empty_tile_nums / np.sum(nnzs_update_per_blk, axis=1) / tile.j_num

			# 支持withdraw，我们需要计算每一个TC tile row的extra cost
			extra_cost = no_atomic_tile_cost * non_empty_tile_nums - TC_row_costs

			if os.environ['no_withdraw']=='True':
				extra_cost = no_atomic_tile_cost * non_empty_tile_nums

			pred_avg_costs = extra_cost / tot_nnz_per_blk_row / tile_sizes[1][1] # tile.j_num

			min_row = int(np.nanargmin(pred_avg_costs))
			min_row_cost = np.nanmin(pred_avg_costs)
			if min_row_cost < best_cost:
				best_cost = min_row_cost # cost/nnz
				selected_poses = [(min_row, 0, i) for i in range(int(non_empty_tile_nums[min_row])) ]
				if sub_op.TC_k_notsorted:
					selected_poses = [(min_row, 0, i) for i in sub_op.blk_ks[tile_sizes][min_row] ]
				best_tile_key = (sub_op.op_id, tile_sizes, selected_poses, False)

				print("update best TC (while line): ", best_cost, best_tile_key, "row ", min_row, " non_empty_tile_num: ", non_empty_tile_nums[min_row])

	return best_tile_key, best_cost




# 从原来的比较nnz改为了比较avg cost
def search_best_TC_tile(sub_ops, dsize, TC_row_tile_dict, TC_row_costs, penalty, max_bucket_size):
	# best_tile_key, best_nnz = None, -1
	best_tile_key, best_cost = None, float('inf')
	for sub_op in sub_ops:
		key, avg_cost = search_best_TC_tile_given_op(sub_op, dsize,
			TC_row_tile_dict, TC_row_costs, 
			penalty, max_bucket_size, 
			is_tb_tile = False,
			lower_bound_dict = dict(), 
			tile_dict = dict(), 
			# selected_position_space = dict()
			)
		# if nnz > best_nnz:
		# 	best_tile_key = key
		# 	best_nnz = nnz
		if avg_cost < best_cost:
			best_tile_key = key
			best_cost = avg_cost
	# return best_tile_key, best_nnz
	return best_tile_key, best_cost





# 这个版本我们不计算lower bound，而是直接向量化地计算所有tile的cost，然后取最小值。
def search_best_tile_given_op_BSPMM(ori_op, sub_op, dsize, ori_tot_nnz, SM_num, max_bucket_size,
	best_tile_key = None, best_pred_avg_cost = float('inf'), second_best_pred_avg_cost = float('inf'),
	is_tb_tile = False,
	lower_bound_dict = dict(), 
	tile_dict = dict(), 
	nnzs_dict = dict(), pos_is_dict = dict(), costs_dict = dict(), 
	# selected_position_space = dict()
	):
	'''
	Search the tile with the minimum average cost for this sub_op.
	Use the lower bound of average cost for pruning.
	Note:
		is_tb_tile should be set to True.
	INPUT:
		tile_dict: the dict which stores the tiles that we have already explored.
		selected_position_space: the dict which stores the position of all selected nnzs.
	这个函数只考虑非tensor core的tile。 ==>现在变成了只考虑ELL tile的op.
	需要ori_op来判断一个tile是否是atomic的
	'''

	# if (sub_op.op_type == 'spmm') and (get_template_str(sub_op) == 'TensorCore_template'):


	tile_sizes_idx = get_tile_sizes_idx(sub_op, max_bucket_size, is_tb_tile=is_tb_tile)
	if sub_op.op_type == 'spmm':
		# we know with larger tile.J, the average cost will be larger. so we only need to consider
		tile_sizes_idx[1] = [sorted(tile_sizes_idx[1], key=lambda tmp: tmp[1])[-1]]

		# FOR DEBUG 因为猜测j越大的时候，nvcc在编译的过程中产生的local register的数量可能越多，从而会降低能同时在SM运行的block的数量
		# 因此此处限定tile 的j size为32
		tile_sizes_idx[1] = [(None, 32)]
		# ===============================


	# print("tile_sizes_idx: ", tile_sizes_idx)

	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=is_tb_tile)
	for tile_sizes in tile_sizes_list:
		'''
		只有ell tile 的cost在fix tile sizes，只改变tile pos的时候，pos越往下，其average cost的lower bound越大。
		csr的tile的lower bound没有这个规律。
		所以我们枚举的时候按照for tile I, for tile pos来，不过好像已经默认是这个顺序了，因为tile_sizes_list里面tile K，tile J 都不变。
		'''
		# is_ell = (get_template_str(sub_op) == 'sparse_template_ell')
		# is_csr = (get_template_str(sub_op) == 'sparse_template')
		
		# tile_pos_list = get_tile_pos_list(sub_op, tile_sizes)
		tile_pos_list = get_valid_tile_pos_list_for_ELL(sub_op, tile_sizes, max_bucket_size)

		# print("tile_sizes: ", tile_sizes, " tile_pos_list: ", tile_pos_list)

		# 直接批量计算所有可能的position上的tile的cost

		# 先得到sub_op中的不完整的行号
		# 需要去除这里面nnz已经为0的行号
		atomic_rows = np.nonzero( \
			((ori_op.inps[0].getnnz(axis=1)[sub_op.idx_values_list[0][0]][ sub_op.hyb_new_rows[0][sub_op.hyb_short_begin_row:] ]) \
			 			> sub_op.position_space_update.getnnz(axis=1)[ sub_op.hyb_short_begin_row: ]) & \
			 			(sub_op.position_space_update.getnnz(axis=1)[ sub_op.hyb_short_begin_row: ]>0) )[0] + sub_op.hyb_short_begin_row
		# 得到不完整行号，或者需折叠的行号 所在的tile pos i；这些tile都是需要atomic的
		atomic_pos_is = np.concatenate( [np.arange( ((sub_op.hyb_short_begin_row-1) // tile_sizes[0][1]) + 1 ), atomic_rows//tile_sizes[0][1]] )
		tile_pos_is = [i[0] for i in tile_pos_list]
		atomic_pos_is = np.intersect1d(tile_pos_is, atomic_pos_is)

		non_atomic_pos_is = np.setdiff1d(tile_pos_is, atomic_pos_is, assume_unique=True)

		# 要提前去除nnz为0的tile pos
		indptr = sub_op.position_space_update.indptr
		atomic_nnzs = None
		if len(atomic_pos_is) > 0:
			inds_start = indptr[tile_sizes[0][1]*atomic_pos_is]
			inds_ends = indptr[ np.minimum(tile_sizes[0][1]*(atomic_pos_is+1), len(indptr)-1) ]
			atomic_nnzs = inds_ends - inds_start
			atomic_pos_is = atomic_pos_is[atomic_nnzs>0]
			atomic_nnzs = atomic_nnzs[atomic_nnzs>0]*tile_sizes[1][1]

		non_atomic_nnzs = None
		if len(non_atomic_pos_is) > 0:
			inds_start = indptr[tile_sizes[0][1]*non_atomic_pos_is]
			inds_ends = indptr[ np.minimum(tile_sizes[0][1]*(non_atomic_pos_is+1), len(indptr)-1) ]
			non_atomic_nnzs = inds_ends - inds_start
			non_atomic_pos_is = non_atomic_pos_is[non_atomic_nnzs>0]
			non_atomic_nnzs = non_atomic_nnzs[non_atomic_nnzs>0]*tile_sizes[1][1]


		nnzs_dict[(sub_op.op_id, tile_sizes)] = atomic_nnzs, non_atomic_nnzs
		pos_is_dict[(sub_op.op_id, tile_sizes)] = atomic_pos_is, non_atomic_pos_is
		costs_dict[(sub_op.op_id, tile_sizes)] = list()

		# 分别计算cost并求最小cost
		for pos_is, is_atomic, nnzs in ((atomic_pos_is, True, atomic_nnzs), (non_atomic_pos_is, False, non_atomic_nnzs)):
			# print(f"pos_is: {pos_is}")
			if len(pos_is) == 0:
				costs_dict[(sub_op.op_id, tile_sizes)].append(np.array([]))
				continue
			costs = my_cost_model.cost_tb_latency_given_ELL_tiles(sub_op, tile_sizes, pos_is, dsize, ori_tot_nnz, SM_num, is_atomic)
			costs_dict[(sub_op.op_id, tile_sizes)].append(costs)
			avg_costs = costs / nnzs
			# print(f"avg_costs: {avg_costs}")
			# print(f"nnzs: {nnzs}")
			idx = np.nanargmin(avg_costs)
			print(f"idx: {idx}   min avg cost: {avg_costs[idx]}   best_pred_avg_cost: {best_pred_avg_cost}")
			if avg_costs[idx] < best_pred_avg_cost:
				best_pred_avg_cost = avg_costs[idx]
				best_tile_key = (sub_op, tile_sizes, (pos_is[idx], 0, 0), is_atomic, nnzs[idx])

		# ===============================================================
	return best_tile_key, best_pred_avg_cost, second_best_pred_avg_cost





def search_best_tile_given_op_BSDDMM(ori_op, sub_op, dsize, ori_tot_nnz, SM_num, max_bucket_size,
	best_tile_key = None, best_pred_avg_cost = float('inf'), second_best_pred_avg_cost = float('inf'),
	is_tb_tile = False,
	lower_bound_dict = dict(), 
	tile_dict = dict(), 
	nnzs_dict = dict(), pos_is_dict = dict(), costs_dict = dict(), 
	# selected_position_space = dict()
	):
	'''
	Search the best 1D tile for SDDMM.
	'''

	# 直接把max_bucket_size参数当成我们考虑的1D tile的output point workload
	costs = my_cost_model.cost_tb_latency_given_1D_tiles(sub_op, (max_bucket_size, ), dsize)
	# 
	# 这里的nnz不应该这么设，应该是update之后的nnz的大小
	# 我们并没有在position space中对nnz进行折叠
	nnzs = sub_op.nnz_update_1D
	# ================================================================================================================================
	# 这是之前的写法，已经被舍弃
	# nnzs = sub_op.position_space_update.getnnz(axis=1)
	# ================================================================================================================================
	
	avg_costs = costs/nnzs/sub_op.idx_lens[1] # TODO SDDMM: 此处可能要修改，因为我们打算更换SDDMM的idx的顺序
	max_pos = np.argmin(avg_costs)
	# tmp_best_pred_avg_cost = costs[max_pos]/nnzs[max_pos]
	tmp_best_pred_avg_cost = avg_costs[max_pos]
	# 
	tmp_best_tile_key = (sub_op, (max_bucket_size, ), (max_pos, ), False, nnzs[max_pos]*sub_op.idx_lens[1]) # TODO SDDMM: 此处可能要修改，因为我们打算更换SDDMM的idx的顺序

	if tmp_best_pred_avg_cost < best_pred_avg_cost:
		best_tile_key = tmp_best_tile_key
		best_pred_avg_cost = tmp_best_pred_avg_cost

	# 和SPMM不同，我们在此处直接存avg cost
	costs_dict[(sub_op.op_id, (max_bucket_size, ))] = avg_costs

	return best_tile_key, best_pred_avg_cost, second_best_pred_avg_cost






# 这个版本我们不计算lower bound，而是直接向量化地计算所有tile的cost，然后取最小值。
def search_best_tile_given_op(ori_op, sub_op, dsize, ori_tot_nnz, SM_num, max_bucket_size,
	best_tile_key = None, best_pred_avg_cost = float('inf'), second_best_pred_avg_cost = float('inf'),
	is_tb_tile = False,
	lower_bound_dict = dict(), 
	tile_dict = dict(), 
	nnzs_dict = dict(), pos_is_dict = dict(), costs_dict = dict(), 
	# selected_position_space = dict()
	):
	'''
	Search the tile with the minimum average cost for this sub_op.
	Use the lower bound of average cost for pruning.
	Note:
		is_tb_tile should be set to True.
	INPUT:
		tile_dict: the dict which stores the tiles that we have already explored.
		selected_position_space: the dict which stores the position of all selected nnzs.
	这个函数只考虑非tensor core的tile。 ==>现在变成了只考虑ELL tile的op.
	需要ori_op来判断一个tile是否是atomic的
	'''

	# if (sub_op.op_type == 'spmm') and (get_template_str(sub_op) == 'TensorCore_template'):

	if ori_op.op_type == 'spmm':
		return search_best_tile_given_op_BSPMM(ori_op, sub_op, dsize, ori_tot_nnz, SM_num, max_bucket_size,
				best_tile_key = best_tile_key, best_pred_avg_cost = best_pred_avg_cost, second_best_pred_avg_cost = second_best_pred_avg_cost,
				is_tb_tile = is_tb_tile,
				lower_bound_dict = lower_bound_dict, 
				tile_dict = tile_dict, 
				nnzs_dict = nnzs_dict, pos_is_dict = pos_is_dict, costs_dict = costs_dict, 
				)
	elif ori_op.op_type == 'sddmm':
		return search_best_tile_given_op_BSDDMM(ori_op, sub_op, dsize, ori_tot_nnz, SM_num, max_bucket_size,
				best_tile_key = best_tile_key, best_pred_avg_cost = best_pred_avg_cost, second_best_pred_avg_cost = second_best_pred_avg_cost,
				is_tb_tile = is_tb_tile,
				lower_bound_dict = lower_bound_dict, 
				tile_dict = tile_dict, 
				nnzs_dict = nnzs_dict, pos_is_dict = pos_is_dict, costs_dict = costs_dict, 
				# selected_position_space = dict()
				)






def search_best_tile(ori_op, sub_ops, tile_dict, dsize, ori_tot_nnz, SM_num, max_bucket_size,
	nnzs_dict, pos_is_dict, costs_dict,
	best_pred_avg_cost = float('inf')
	# selected_position_space
	):
	'''
	INPUT:
		sub_ops: the sub_ops we consider
		tile_dict: stores the tiles we have explored, we will also update the tiles' nnz.
		selected_position_space: the position space of the elements we have covered.
	OUTPUT:
		best_tile_key, second_best_pred_avg_cost
	'''
	lower_bound_dict = dict()
	# best_tile_key, best_pred_avg_cost, second_best_pred_avg_cost = None, float('inf'), float('inf')
	best_tile_key, second_best_pred_avg_cost = None, float('inf')
	for sub_op in sub_ops:
		# search for each sub_op, and update best_pred_avg_cost, second_best_pred_avg_cost
		best_tile_key, best_pred_avg_cost, second_best_pred_avg_cost = search_best_tile_given_op(
			ori_op,
			sub_op, dsize, ori_tot_nnz, SM_num, max_bucket_size,
			best_tile_key = best_tile_key,
			best_pred_avg_cost = best_pred_avg_cost, second_best_pred_avg_cost = second_best_pred_avg_cost,
			is_tb_tile = True, 
			lower_bound_dict = lower_bound_dict,
			tile_dict = tile_dict, 
			nnzs_dict = nnzs_dict, pos_is_dict = pos_is_dict, costs_dict = costs_dict, 
			# selected_position_space = selected_position_space
			)


	# 要把best_tile_key对应的tile存进tile_dict中
	if best_tile_key != None:
		(sub_op, tile_sizes, tile_pos, is_atomic, nnz_uncovered) = best_tile_key
		print(f"best ELL tile: {(sub_op.op_id, tile_sizes, tile_pos, is_atomic, nnz_uncovered)}")
		params = get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile = True)[0]
		tile = ComputeTile(sub_op, tile_sizes, tile_pos, params)
		tile.is_atomic_tile = is_atomic
		tile.pred_cost = nnz_uncovered * best_pred_avg_cost
		tile.update_nnz_uncovered()
		tile.set_pred_avg_cost()
		tile_dict[(sub_op.op_id, tile_sizes, tile_pos, is_atomic)] = tile
		best_tile_key = (sub_op.op_id, tile_sizes, tile_pos, is_atomic)


	# we find the first best tile to select
	return best_tile_key, second_best_pred_avg_cost





def gen_next_level_subops(global_op_id, new_op, run_params, ori_cand_subops, max_bucket_size, is_tb_tile = False):
	ret_subops = list()
	if new_op.op_type == 'spmm':
		# position space are about index i, k
		# global global_op_id
		new_subops, global_op_id = gen_candidate_formats(new_op, run_params, global_op_id)

		for new_subop, ori_subop in zip(new_subops, ori_cand_subops):
			gen_position_space_for_area(new_subop, 0)

			# new_tiles = new_tiles_list[0]
			# 我们暂时不考虑tensor core的next level的sub_op
			template_str = get_template_str(new_subop)
			if template_str == "TensorCore_template":
				# new_tiles = new_tiles_list[1]
				continue

			# directly compare the idx_values of these two areas
			# We do not tile on index k, so the tile_sizes will only change along index i. 
			# 对于index i，我们要比较每个index value的顺序，对于index k，因为不做切割且不是output index，我们只比较全部可能值的集合
			# 我们不需要管j，因为每次选择tile都会把所有j覆盖，根本上是因为j是dense的，我们不会reorder j。
			is_same = [np.array_equal(new_subop.idx_values_list[0][0], ori_subop.idx_values_list[0][0]), \
						set(new_subop.idx_values_list[0][2]) == set(ori_subop.idx_values_list[0][2])]

			# TODO: 这个地方is_same的计算在考虑了tensor core之后还是这样吗？感觉就算不考虑tensor core，也不应该只比较集合，还是应该比较list。
			# 但是又想了一下，可能是因为不同k的order但是相同k集合对于最终latency的影响并不大。所以对非tensor core的tile还是采用 set(k)来比较。
			# 要不要考虑level 2 的tensor core tile？暂时不考虑了，因为想了一下，似乎level 2 的tensor core的tile并不能保证density一定会变高，因为i轴是不能reorder的。


			if False not in is_same:
				# no need to consider the tiles in this new_subop, because all these tiles are already considered
				continue

			ret_subops.append(new_subop)
		return ret_subops, global_op_id






# 这个函数是直接以new_op为ori op来走一遍生成cand_op的流程
def gen_full_next_level_subops_based_on_new_op(global_op_id, new_op, max_bucket_size, kernel_tile_size_options, 
	TC_k_notsorted,
	reorder_k_by_nnz = True, is_tb_tile = False):
	ret_subops = list()
	if new_op.op_type in ['spmm', 'sddmm']:
		# position space are about index i, k
		# global global_op_id
		new_subops, global_op_id = gen_candidate_formats(new_op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
			reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = global_op_id, gen_TC_formats = False)

		for new_subop in new_subops:
			gen_position_space_for_area(new_subop, 0)

			# new_tiles = new_tiles_list[0]
			# 我们暂时不考虑tensor core的next level的sub_op
			template_str = get_template_str(new_subop)
			# 我们也不考虑CSR tile，也就是说，我们只考虑ELL tile
			if template_str in ["TensorCore_template", "sparse_template", 'TC_sddmm']:
				# new_tiles = new_tiles_list[1]
				continue
			ret_subops.append(new_subop)
		return ret_subops, global_op_id



# 这个op会把new op中的行按照是否涉及到已经选择的TC tile进行区分
def gen_full_next_level_subops(TC_is, global_op_id, new_op, max_bucket_size, kernel_tile_size_options, 
	TC_k_notsorted,
	reorder_k_by_nnz = True, is_tb_tile = False):
	'''
	TC_is: the row ids where selected TC tiles cover.
	'''
	if new_op.op_type == 'spmm':
		TC_is = np.nonzero(TC_is)[0]
		csr = new_op.inps[0]
		indptr = np.zeros( csr.shape[0]+1 )
		indptr[TC_is+1] = csr.getnnz(axis=1)[TC_is]
		indptr = np.cumsum(indptr)
		indices = np.concatenate( [csr.indices[ csr.indptr[i]:csr.indptr[i+1] ] for i in TC_is] + [np.array([], dtype='int')] )
		csr_affected = scipy.sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=csr.shape)
		csr_other = csr - csr_affected
		csr_other = csr_other.tocsr()
		csr_other.eliminate_zeros()

		inps1 = [csr_affected, new_op.inps[1]]
		sparse_inps_transposed1 = [ inps1[0].tocsc() ]
		inps2 = [csr_other, new_op.inps[1]]
		sparse_inps_transposed2 = [ inps2[0].tocsc() ]
		new_op1 = Op(new_op.op_type, new_op.sidxs, new_op.ridxs, new_op.idx_lens, new_op.idx_graph, inps1, sparse_inps_transposed1)
		new_op2 = Op(new_op.op_type, new_op.sidxs, new_op.ridxs, new_op.idx_lens, new_op.idx_graph, inps2, sparse_inps_transposed2)

		# directly call gen_full_next_level_subops_based_on_new_op
		ret = list()
		for op in [new_op1, new_op2]:
			ret_subops, global_op_id = gen_full_next_level_subops_based_on_new_op(
				global_op_id, op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
				reorder_k_by_nnz = reorder_k_by_nnz, is_tb_tile = is_tb_tile)
			ret = ret + ret_subops

		print(f"next level op num: {len(ret)}")

		return ret, global_op_id
	elif new_op.op_type == 'sddmm':
		ret = gen_full_next_level_subops_based_on_new_op(
			global_op_id, new_op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
			reorder_k_by_nnz = reorder_k_by_nnz, is_tb_tile = is_tb_tile)		
		print("ret: ",  ret)
		ret_subops, global_op_id = ret
		print(f"next level op num: {len(ret)}")
		return ret_subops, global_op_id



def update_covered_position_space(selected_tile, selected_position_space):
	'''
	Update the covered position space: selected_position_space.
	'''
	if selected_tile.op.op_type == 'spmm':
		for i, ks in selected_tile.position_space_when_selected['in1'].items():
			if i not in selected_position_space:
				selected_position_space[i] = set()
			selected_position_space[i].update(ks)








# 这个版本会考虑分数形式的tile pos k
# 这个版本考虑的whole row TC tile直接就是withdraw的版本
def local_search_TC_tiles_given_csr(sub_op, csr, blk_row_i, tile_sizes, target_pred_avg_cost, dsize, 
	no_atomic_tile_cost, atomic_tile_cost, selected_tiles_this_row):
	'''
	Local search if selected_tile is a TC tile.
	在指定的csr上寻找good tile。给定的csr是单个block行的csr。
	sub_op只用来提供op_type和template的信息。
	'''
	k_axis = 2

	blk_i_size = tile_sizes[0][1]

	# 我们先考虑TC tile覆盖整行的情况
	k_num = csr.shape[1]
	# 
	# nnzs_update = np.add.reduceat(
	# 				csr.getnnz(axis=0)[ sub_op.k_vals[blk_row_i] ], 
	# 				np.arange(0, k_num, tile_sizes[2][1])
	# 				)

	# 因为我们会考虑分数的tile pos k，而local search又会同时得到一组互不相干的tile，所以此处的nnz_update会从第一个non-empty的列组开始
	# 然后tile之间没有交叉
	non_empty_begin = np.nonzero(csr.getnnz(axis=0)[ sub_op.k_vals[blk_row_i] ])[0][0]//16
	if os.environ['single_level']=='True':
		non_empty_begin = non_empty_begin-non_empty_begin

	nnzs_update = np.add.reduceat(
					csr.getnnz(axis=0)[ sub_op.k_vals[blk_row_i] ], 
					np.arange(non_empty_begin*16, k_num, tile_sizes[k_axis][1])
					)


	good_poses = list()
	good_tiles = list()
	params = {"mma_shape_str":"m16n16k16"}
	if sub_op.op_type == 'spmm':
		# non_empty_tile_nums = np.ceil(np.count_nonzero(nnzs, axis=1) / tile_sizes[2][1])
		non_empty_tile_nums = sub_op.blk_nums[tile_sizes][blk_row_i]
		
		# no_atomic_tile_cost = my_cost_model.cost_tb_latency_given_tile(selected_tile, dsize, None, None, False) #is_atomic is False

		# 看哪些整行TC tile的pred avg cost在允许范围之内
		# 在此处依然使用nnz_update来计算平均cost，尽管我们生成的时候并不允许别的tile可以cover这些行的nnz
		# 考虑withdraw的情况
		extra_cost = no_atomic_tile_cost * non_empty_tile_nums - sum([i.pred_cost for i in selected_tiles_this_row])

		if os.environ['no_withdraw']=='True':
			extra_cost = no_atomic_tile_cost * non_empty_tile_nums

		pred_avg_costs = extra_cost / np.sum(nnzs_update) / tile_sizes[1][1]
		# selected_rows = np.nonzero(pred_avg_costs < target_pred_avg_cost)[0]

		good_poses = list()
		if pred_avg_costs <= target_pred_avg_cost:
			good_poses = [(int(blk_row_i), 0, i) for i in range(int(non_empty_tile_nums))]
			if sub_op.TC_k_notsorted:
				good_poses = [(int(blk_row_i), 0, i) for i in sub_op.blk_ks[tile_sizes][blk_row_i] ]

		print("whole: ", good_poses)
		
		# params = {"mma_shape_str":"m16n16k16"}
		good_tiles = [ComputeTile(sub_op, tile_sizes, tile_pos, params) for tile_pos in good_poses]
		for t in good_tiles:
			t.is_atomic_tile = False
			t.pred_cost = no_atomic_tile_cost
			t.cost = no_atomic_tile_cost
			t.best_tile_sizes = t.tile_sizes
			t.best_params = t.params
			# 此时的uncovered_position_space应该是正确的。

	# 接下来考虑单个TC tile（需要atomic）的情况==================================================
	# atomic_tile_cost = my_cost_model.cost_tb_latency_given_tile(selected_tile, dsize, None, None, True) #is_atomic is True

	if len(good_poses)>0:
		# 因为已经把整行的TC tile都找到了
		print(f"WE FIND {len(good_tiles)} GOOD TILES whole row")
		return good_tiles

	tile_pos_k_begin = non_empty_begin*Fraction(16, tile_sizes[k_axis][1])
	if len(good_poses) == 0:
		# 此处对于SDDMM使用atomic_tile_cost也没问题，因为我们设置得atomic_tile_cost和non_atomic_tile_cost是一样的
		if sub_op.op_type == 'spmm':
			good_poses = [(int(blk_row_i), 0, int(i) + tile_pos_k_begin ) \
				for i in np.nonzero(atomic_tile_cost <= nnzs_update*tile_sizes[1][1]*target_pred_avg_cost)[0] ]
		elif sub_op.op_type == 'sddmm':
			good_poses = [(int(blk_row_i), 0, int(i) + tile_pos_k_begin ) \
				for i in np.nonzero(atomic_tile_cost*math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1]) <= nnzs_update* sub_op.idx_lens[1] *target_pred_avg_cost)[0] ]
	else:
		good_poses = list()

	print("single: ", good_poses[:5], '...')


	good_tiles2 = [ComputeTile(sub_op, tile_sizes, tile_pos, params) for tile_pos in good_poses]
	for t in good_tiles2:
		t.is_atomic_tile = True
		t.pred_cost = atomic_tile_cost
		t.cost = atomic_tile_cost
		if sub_op.op_type == 'sddmm':
			t.is_atomic_tile = False
			t.pred_cost = atomic_tile_cost * math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1])
			t.cost = atomic_tile_cost * math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1])
		t.best_tile_sizes = t.tile_sizes
		t.best_params = t.params

	print(f"WE FIND {len(good_tiles)+len(good_tiles2)} GOOD TILES")

	return good_tiles + good_tiles2









# 会同时考虑整行的TC tile和单独的TC tile
# 支持分数的tile pos k
# 支持以withdraw的方式选择整行TC tile
# 在withdraw地选择整行TC tile的时候，允许考虑所有可能的tile shape
def local_search_TC_tiles(selected_tile, target_pred_avg_cost, dsize, 
	TC_row_tile_dict, TC_row_costs, cur_selected_tile_num, penalty, max_bucket_size, TC_is):
	'''
	Local search if selected_tile is a TC tile.
	'''
	tile_sizes = selected_tile.tile_sizes
	sub_op = selected_tile.op
	row_num = len(sub_op.idx_values_list[0][0])
	blk_i_size = tile_sizes[0][1]
	row_blk_num = math.ceil(row_num / blk_i_size)

	# 我们先考虑TC tile覆盖整行的情况
	# csr = sub_op.position_space[0]  # 因为是完整覆盖整行，所以用的是position_space而不是position_space_update
	# 
	# nnzs = np.array([ csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ] for i in range(row_blk_num)])	

	csr = sub_op.position_space_update
	k_num = csr.shape[1]
	# 
	# nnzs_update = np.asarray([ np.add.reduceat(
	# 				csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ], 
	# 				np.arange(0, k_num, tile_sizes[2][1])
	# 				)  \
	# 			for i in range(row_blk_num)])


	# 为了支持分数的tile pos k，先找到非空的16列组
	non_empty_begin = np.asarray([ np.nonzero(csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ])[0][0]//16 \
			if csr[i*blk_i_size:(i+1)*blk_i_size, :].nnz>0 else 0\
			for i in range(row_blk_num)])


	if os.environ['single_level']=='True':
		non_empty_begin = non_empty_begin-non_empty_begin

	# print("non_empty_begin[1590]: ", non_empty_begin[1590])
	# # 如果选择的TC tile并不能从第一个非零列组开始得到，那就需要修改第一个非零列组的值
	# if (selected_tile.tile_pos[2] * tile_sizes[2][1] - non_empty_begin[selected_tile.tile_pos[0]] * 16) % tile_sizes[2][1] != 0:
	# 	dist = math.ceil((selected_tile.tile_pos[2] * tile_sizes[2][1] - non_empty_begin[selected_tile.tile_pos[0]] * 16)/tile_sizes[2][1])
	# 	dist = dist * tile_sizes[2][1]
	# 	non_empty_begin[selected_tile.tile_pos[0]] = (selected_tile.tile_pos[2] * tile_sizes[2][1] - dist)//16

	# print("non_empty_begin[1590]: ", non_empty_begin[1590])

	nnzs_update = np.asarray([ np.add.reduceat(
					np.concatenate([ csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)[ sub_op.k_vals[i] ],  np.full(non_empty_begin[i]*16, 0)]), 
					np.arange(non_empty_begin[i]*16, k_num+non_empty_begin[i]*16, tile_sizes[2][1])
					)  \
				for i in range(row_blk_num)])

	# # non_empty_tile_nums = np.ceil(np.count_nonzero(nnzs, axis=1) / tile_sizes[2][1])
	# non_empty_tile_nums = sub_op.blk_nums[tile_sizes]
	
	# no_atomic_tile_cost = my_cost_model.cost_tb_latency_given_tile(selected_tile, dsize, None, None, False) #is_atomic is False
	# # 看哪些整行TC tile的pred avg cost在允许范围之内
	# # 在此处依然使用nnz_update来计算平均cost，尽管我们生成的时候并不允许别的tile可以cover这些行的nnz
	# # 为了支持withdraw，此处需要先计算extra cost
	# extra_cost = no_atomic_tile_cost * non_empty_tile_nums - TC_row_costs
	# pred_avg_costs = extra_cost / np.sum(nnzs_update, axis=1) / selected_tile.j_num
	# selected_rows = np.nonzero(pred_avg_costs < target_pred_avg_cost)[0]

	# # 为了支持withdraw，还需要更新TC_row_costs，令被完全覆盖的行的cost为-1，这样的话，还可以知道哪些行发生了withdraw的操作
	# TC_row_costs[selected_rows] = -1
	
	# good_poses = [(int(row_i), 0, i) for row_i in selected_rows for i in range(int(non_empty_tile_nums[row_i]))]

	good_tiles = list()
	params = selected_tile.params
	if sub_op.op_type == 'spmm':
		# 支持同时考虑所有shape的whole-row TC tiles
		tile_sizes_idx = get_tile_sizes_idx(sub_op, None, is_tb_tile=True)
		tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=True)
		non_empty_tile_nums = np.asarray([sub_op.blk_nums[tmp] for tmp in tile_sizes_list]) # [nums for tile_sizes1, nums...2, ...]
		no_atomic_tile_costs = np.asarray([my_cost_model.get_benchmark_cost_("TensorCore_template", tmp, dsize, False, penalty) for tmp in tile_sizes_list ]) #is_atomic is False
		extra_costs = (no_atomic_tile_costs.reshape((-1, 1))) * non_empty_tile_nums - TC_row_costs
		
		if os.environ['no_withdraw']=='True':
			extra_costs = (no_atomic_tile_costs.reshape((-1, 1))) * non_empty_tile_nums

		# pred_avg_costs = extra_costs / np.sum(nnzs_update, axis=1) / selected_tile.j_num
		# 下面的这个函数在只有np.nan和np.inf的时候无用。但是我们可以先比较extra_costs，这样就避免了这个问题
		cand_tile_sizes_idx = np.nanargmin(extra_costs, axis=0) # 即每一个TC row最好的tile sizes的idx
		# 使用<=主要是为了两边都为0的情况，但是好像没用，因为0*inf=nan；也就是说我们当target是inf的时候，我们其实不会选对应的行；那这样会遇到死循环
		# 我们还是应该直接忽略extra cost为0，nnz也为0的情况
		# selected_rows = np.nonzero(np.nanmin(extra_costs, axis=0) <= (np.sum(nnzs_update, axis=1) * selected_tile.j_num * target_pred_avg_cost))[0]
		selected_rows = np.nonzero(np.nanmin(extra_costs, axis=0)/np.sum(nnzs_update, axis=1)/selected_tile.j_num <= target_pred_avg_cost)[0]
		print(f"selected_rows: {selected_rows}")
		
		print(f"current min avg cost for whole row: {np.nanmin(extra_costs, axis=0)/np.sum(nnzs_update, axis=1)/selected_tile.j_num}, {np.nanmin(np.nanmin(extra_costs, axis=0)/np.sum(nnzs_update, axis=1)/selected_tile.j_num)}, target: {target_pred_avg_cost}")

		TC_row_costs[selected_rows] = -1
		good_poses = [(int(row_i), 0, i) for row_i in selected_rows \
											for i in range(int(non_empty_tile_nums[cand_tile_sizes_idx[row_i]][row_i]))]


		if sub_op.TC_k_notsorted:
			blk_ks = [sub_op.blk_ks[tmp] for tmp in tile_sizes_list]
			good_poses = [(int(row_i), 0, i) for row_i in selected_rows \
											for i in blk_ks[cand_tile_sizes_idx[row_i]][row_i]]

		TC_is[ np.concatenate([ np.arange(i*blk_i_size, min((i+1)*blk_i_size, row_num)) for i in selected_rows ] + [np.array([], dtype='int')]) ] = True

		# params = selected_tile.params
		# good_tiles = [ComputeTile(sub_op, tile_sizes, tile_pos, params) for tile_pos in good_poses]
		good_tiles = [ComputeTile(sub_op, tile_sizes_list[cand_tile_sizes_idx[tile_pos[0]]], tile_pos, params) for tile_pos in good_poses]
		for t in good_tiles:
			t.is_atomic_tile = False
			t.pred_cost = no_atomic_tile_costs[ cand_tile_sizes_idx[t.tile_pos[0]] ]
			# t.pred_cost = no_atomic_tile_cost
			t.best_tile_sizes = t.tile_sizes
			t.best_params = t.params
			t.cost = t.pred_cost

		# print(f"whole: {good_poses}")

	# 接下来考虑单个TC tile（需要atomic）的情况==================================================

	# print(nnzs[0], len(nnzs), target_pred_avg_cost, np.concatenate(nnzs))
	# print(np.nonzero( np.concatenate(nnzs) >= 1/target_pred_avg_cost))

	# we are interested in the tiles whose pred_avg_cost is < target_pred_avg_cost
	# good_poses = [ (i//len(nnzs[0]), 0, i%len(nnzs[0])) for i in np.nonzero( np.concatenate(nnzs) > 1/target_pred_avg_cost)[0]]
	# good_poses = [ (i//len(nnzs[0]), 0, i%len(nnzs[0])) \
	# 	for i in np.nonzero( np.concatenate(nnzs) > selected_tile.pred_cost/target_pred_avg_cost/selected_tile.j_num)[0]]
	# # print(tile_sizes, good_poses[0])
	# print(selected_tile.pred_cost/target_pred_avg_cost/selected_tile.j_num)


	is_atomic = True
	if sub_op.op_type == 'sddmm':
		is_atomic = False
	atomic_tile_cost = my_cost_model.cost_tb_latency_given_tile(selected_tile, dsize, None, None, is_atomic, penalty) #is_atomic is True
	
	tile_pos_k_begin = non_empty_begin*Fraction(16, tile_sizes[2][1])
	good_poses = None
	if sub_op.op_type == 'spmm':
		good_poses = [(int(row_i), 0, int(i)+tile_pos_k_begin[row_i]) \
			for row_i, i in zip(*np.nonzero(atomic_tile_cost/nnzs_update/selected_tile.j_num <= target_pred_avg_cost)) \
			if row_i not in selected_rows]
	elif sub_op.op_type == 'sddmm':
		good_poses = [(int(row_i), 0, int(i)+tile_pos_k_begin[row_i]) \
			for row_i, i in zip(*np.nonzero(atomic_tile_cost*math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1])/nnzs_update/sub_op.idx_lens[1] <= target_pred_avg_cost))]


	# 为了支持withdraw,需要更新 TC_row_tile_dict, TC_row_costs
	rows, counts = np.unique([pos[0] for pos in good_poses], return_counts=True) # keys will be sorted
	start = cur_selected_tile_num+len(good_tiles)
	for row_i, num in zip(rows, counts):
		TC_row_tile_dict[row_i] = np.concatenate([TC_row_tile_dict[row_i], np.arange(start, start+num)])
		TC_row_costs[row_i] = TC_row_costs[row_i] + atomic_tile_cost*num
		start = start + num
		print(f"TC_row_costs[{row_i}]: {TC_row_costs[row_i]}, atomic_tile_cost, num: {atomic_tile_cost, num} ori: {TC_row_costs[row_i] - atomic_tile_cost*num}, nnzs_update: {np.unique(nnzs_update[row_i])}")
		TC_is[ row_i*blk_i_size:(row_i+1)*blk_i_size ] = True

	# print(f"single: {good_poses}")

	good_tiles2 = [ComputeTile(sub_op, tile_sizes, tile_pos, params) for tile_pos in good_poses]
	for t in good_tiles2:
		t.is_atomic_tile = is_atomic
		t.pred_cost = atomic_tile_cost
		if sub_op.op_type == 'sddmm':
			t.pred_cost = atomic_tile_cost*math.ceil(sub_op.idx_lens[1]/tile_sizes[1][1])
		t.best_tile_sizes = t.tile_sizes
		t.best_params = t.params
		t.cost = t.pred_cost

	print(f"WE FIND {len(good_tiles)+len(good_tiles2)} GOOD TILES")

	return good_tiles + good_tiles2










def local_search_ELL(target_pred_avg_cost, ori_op, selected_tile, second_best_cost, best_pred_avg_cost,
	ori_tot_nnz, SM_num, dsize, tile_dict, is_tb_tile, 
	max_bucket_size, 
	nnzs_dict, pos_is_dict, costs_dict, 
	TC_row_tile_dict, TC_row_costs, cur_selected_tile_num,
	reverse_idx_TC_op_row,
	penalty, TC_is, 
	max_avg_cost_diff = float('inf')):
	sub_op = selected_tile.op
	
	good_tiles = list()

	# =============================================================================================
	tile_sizes = selected_tile.tile_sizes
	
	# tile_pos_list = get_tile_pos_list(sub_op, tile_sizes)
	tile_pos_list = get_valid_tile_pos_list_for_ELL(sub_op, tile_sizes, max_bucket_size)

	# 可以直接利用在search best ELL tile的时候得到的pred_costs和nnz，那其实也就是直接利用search best ELL tile的时候得到的avg cost
	# 不需要任何的更新没错.
	atomic_nnzs, non_atomic_nnzs = nnzs_dict[(sub_op.op_id, tile_sizes)]
	atomic_pos_is, non_atomic_pos_is = pos_is_dict[(sub_op.op_id, tile_sizes)]
	atomic_costs, non_atomic_costs = costs_dict[(sub_op.op_id, tile_sizes)]
	# 此处的nnzs是包含了j轴的nnzs
	# print(atomic_pos_is)
	# print(non_atomic_pos_is)
	for pos_is, is_atomic, nnzs, costs in ((atomic_pos_is, True, atomic_nnzs, atomic_costs), (non_atomic_pos_is, False, non_atomic_nnzs, non_atomic_costs)):
		if len(pos_is) == 0:
			continue
		avg_costs = costs / nnzs
		# print(f"costs: {costs}")
		# print(f"nnzs: {nnzs}")
		# print(f"avg_costs: {avg_costs}")
		good_idx = avg_costs <= target_pred_avg_cost
		good_pos_is = pos_is[good_idx]
		good_costs = costs[good_idx]
		# 需要生成所有对应的compute tile

		for cost, pos_i in zip(good_costs, good_pos_is):
			params = get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile = is_tb_tile)[0]
			tile = ComputeTile(sub_op, tile_sizes, (pos_i, 0, 0), params)
			tile.is_atomic_tile = is_atomic
			tile.pred_cost = cost
			tile.update_nnz_uncovered()
			tile.set_pred_avg_cost()

			tile.best_tile_sizes = selected_tile.best_tile_sizes
			tile.best_params = selected_tile.best_params
			tile.cost = tile.pred_cost

			good_tiles.append(tile)

			# 为了支持withdraw，要找到完全在一个TC row中的ELL tile
			rows = sub_op.hyb_new_rows[0][ np.arange(pos_i*tile_sizes[0][1], min((pos_i+1)*tile_sizes[0][1], len(sub_op.hyb_new_rows[0])) ) ]
			rows = sub_op.idx_values_list[0][0][rows]
			blk_rows = reverse_idx_TC_op_row[rows]//16
			if len(np.unique(blk_rows)) == 1:
				# 完全在一个TC row中
				TC_row_tile_dict[blk_rows[0]] = np.concatenate([TC_row_tile_dict[blk_rows[0]], [cur_selected_tile_num+len(good_tiles)-1] ])
				TC_row_costs[blk_rows[0]] = TC_row_costs[blk_rows[0]] + cost
				print(f"TC_row_costs[{blk_rows[0]}]: {TC_row_costs[blk_rows[0]]}")
		# 
	return good_tiles





def local_search_1D_BSDDMM(target_pred_avg_cost, ori_op, selected_tile, second_best_cost, best_pred_avg_cost,
	ori_tot_nnz, SM_num, dsize, tile_dict, is_tb_tile, 
	max_bucket_size, 
	nnzs_dict, pos_is_dict, costs_dict, 
	TC_row_tile_dict, TC_row_costs, cur_selected_tile_num,
	reverse_idx_TC_op_row,
	penalty, TC_is, 
	max_avg_cost_diff = float('inf')):
	'''
	Used for the 1D tiles of BSDMM
	'''

	sub_op = selected_tile.op
	
	good_tiles = list()

	# =============================================================================================
	tile_sizes = selected_tile.tile_sizes
	

	# 可以直接利用在search best ELL tile的时候得到的pred_costs和nnz，那其实也就是直接利用search best ELL tile的时候得到的avg cost
	# 不需要任何的更新没错.
	avg_costs = costs_dict[(sub_op.op_id, tile_sizes)]
	
	# 因为我们不在position space里对nonzero进行折叠，所以需要改变写法求nnz
	# nnzs = sub_op.position_space_update.getnnz(axis=1)
	# nnzs = sub_op.nnz_update_1D

	good_idx = (avg_costs <= target_pred_avg_cost)
	good_pos_is = np.arange(len(avg_costs))[good_idx]
	good_costs = avg_costs[good_idx]
	# 需要生成所有对应的compute tile
	for cost, pos_i in zip(good_costs, good_pos_is):
		params = get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile = is_tb_tile)[0]
		tile = ComputeTile(sub_op, tile_sizes, (pos_i, ), params)
		tile.is_atomic_tile = False
		tile.update_nnz_uncovered()
		tile.pred_cost = cost * tile.nnz_uncovered # TODO SDDMM: 因为我们之后可能会想要调整SDDMM的idx顺序
		tile.set_pred_avg_cost()

		tile.best_tile_sizes = selected_tile.best_tile_sizes
		tile.best_params = selected_tile.best_params
		tile.cost = tile.pred_cost

		good_tiles.append(tile)
 
	
	print(f"Find {len(good_tiles)} good 1D tiles!")
	return good_tiles



# 这个新的版本直接计算所有可能的cost，然后选择在目标cost范围内的。
def local_search(ori_op, selected_tile, second_best_cost, best_pred_avg_cost,
	ori_tot_nnz, SM_num, dsize, tile_dict, is_tb_tile, 
	max_bucket_size, 
	nnzs_dict, pos_is_dict, costs_dict, 
	TC_row_tile_dict, TC_row_costs, cur_selected_tile_num,
	reverse_idx_TC_op_row,
	penalty, TC_is, 
	max_avg_cost_diff = float('inf')):
	'''
	Given the selected tile, search other tiles from selected_tile.op which are around it and with close avg_cost.
	INPUT:
		selected_tile: the selected tile.
		second_best_cost: the second best avg cost we find during search.
		max_avg_cost_diff: the max difference between the selected tile and the other close tiles which we are interested
	NOTE:
		当已经选完一个op，并且已经对各个涉及到的数据，e.g., 各个sub_op的position space进行更新之后，才会调用这个函数。
		=>并不需要对sub_op的position space进行更新之后才会调用这个函数。现在是local search完之后再更新。
	'''
	sub_op = selected_tile.op

	# we are interested in the tiles whose pred_avg_cost is < target_pred_avg_cost
	# target_pred_avg_cost = 2 * second_best_cost - best_pred_avg_cost
	# if max_avg_cost_diff < float('inf'):
	# 	target_pred_avg_cost = max(target_pred_avg_cost, best_pred_avg_cost*(1+max_avg_cost_diff))

	# 我们现在舍弃second_best_cost这个限制条件，只根据 max_avg_cost_diff 来确定target_pred_avg_cost
	target_pred_avg_cost = None
	# target_pred_avg_cost = max(target_pred_avg_cost, best_pred_avg_cost*(1+max_avg_cost_diff))
	if best_pred_avg_cost >= 0:
		target_pred_avg_cost = best_pred_avg_cost*(1+max_avg_cost_diff)
	else:
		target_pred_avg_cost = best_pred_avg_cost*(1-max_avg_cost_diff)
	print(second_best_cost, best_pred_avg_cost, "target_pred_avg_cost: ", target_pred_avg_cost)
	
	good_tiles = list()

	if get_template_str(sub_op) in ['TensorCore_template', 'TC_sddmm']:
		return local_search_TC_tiles(selected_tile, target_pred_avg_cost, dsize, 
			TC_row_tile_dict, TC_row_costs, cur_selected_tile_num, penalty, max_bucket_size, TC_is)

	elif get_template_str(sub_op) == 'sparse_template_ell':
		return local_search_ELL(target_pred_avg_cost, ori_op, selected_tile, second_best_cost, best_pred_avg_cost,
			ori_tot_nnz, SM_num, dsize, tile_dict, is_tb_tile, 
			max_bucket_size, 
			nnzs_dict, pos_is_dict, costs_dict, 
			TC_row_tile_dict, TC_row_costs, cur_selected_tile_num,
			reverse_idx_TC_op_row,
			penalty, TC_is, 
			max_avg_cost_diff = max_avg_cost_diff)
	elif get_template_str(sub_op) == '1D_sddmm':
		return local_search_1D_BSDDMM(target_pred_avg_cost, ori_op, selected_tile, second_best_cost, best_pred_avg_cost,
			ori_tot_nnz, SM_num, dsize, tile_dict, is_tb_tile, 
			max_bucket_size, 
			nnzs_dict, pos_is_dict, costs_dict, 
			TC_row_tile_dict, TC_row_costs, cur_selected_tile_num,
			reverse_idx_TC_op_row,
			penalty, TC_is, 
			max_avg_cost_diff = max_avg_cost_diff)


	# =============================================================================================
	tile_sizes = selected_tile.tile_sizes
	
	# tile_pos_list = get_tile_pos_list(sub_op, tile_sizes)
	tile_pos_list = get_valid_tile_pos_list_for_ELL(sub_op, tile_sizes, max_bucket_size)

	# 可以直接利用在search best ELL tile的时候得到的pred_costs和nnz，那其实也就是直接利用search best ELL tile的时候得到的avg cost
	# 不需要任何的更新没错.
	atomic_nnzs, non_atomic_nnzs = nnzs_dict[(sub_op.op_id, tile_sizes)]
	atomic_pos_is, non_atomic_pos_is = pos_is_dict[(sub_op.op_id, tile_sizes)]
	atomic_costs, non_atomic_costs = costs_dict[(sub_op.op_id, tile_sizes)]
	# 此处的nnzs是包含了j轴的nnzs
	# print(atomic_pos_is)
	# print(non_atomic_pos_is)
	for pos_is, is_atomic, nnzs, costs in ((atomic_pos_is, True, atomic_nnzs, atomic_costs), (non_atomic_pos_is, False, non_atomic_nnzs, non_atomic_costs)):
		if len(pos_is) == 0:
			continue
		avg_costs = costs / nnzs
		# print(f"costs: {costs}")
		# print(f"nnzs: {nnzs}")
		# print(f"avg_costs: {avg_costs}")
		good_idx = avg_costs <= target_pred_avg_cost
		good_pos_is = pos_is[good_idx]
		good_costs = costs[good_idx]
		# 需要生成所有对应的compute tile

		for cost, pos_i in zip(good_costs, good_pos_is):
			params = get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile = is_tb_tile)[0]
			tile = ComputeTile(sub_op, tile_sizes, (pos_i, 0, 0), params)
			tile.is_atomic_tile = is_atomic
			tile.pred_cost = cost
			tile.update_nnz_uncovered()
			tile.set_pred_avg_cost()

			tile.best_tile_sizes = selected_tile.best_tile_sizes
			tile.best_params = selected_tile.best_params
			tile.cost = tile.pred_cost

			good_tiles.append(tile)

			# 为了支持withdraw，要找到完全在一个TC row中的ELL tile
			rows = sub_op.hyb_new_rows[0][ np.arange(pos_i*tile_sizes[0][1], min((pos_i+1)*tile_sizes[0][1], len(sub_op.hyb_new_rows[0])) ) ]
			rows = sub_op.idx_values_list[0][0][rows]
			blk_rows = reverse_idx_TC_op_row[rows]//16
			if len(np.unique(blk_rows)) == 1:
				# 完全在一个TC row中
				TC_row_tile_dict[blk_rows[0]] = np.concatenate([TC_row_tile_dict[blk_rows[0]], [cur_selected_tile_num+len(good_tiles)-1] ])
				TC_row_costs[blk_rows[0]] = TC_row_costs[blk_rows[0]] + cost
				print(f"TC_row_costs[{blk_rows[0]}]: {TC_row_costs[blk_rows[0]]}")
		# 
	return good_tiles



# 这个函数用来整理支持withdraw之后得到的selected tiles
def post_process_for_withdraw_BSPMM(ori_op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row):
	# 我们只会用到ori_op的idx_len
	# 首先要删掉哪些被替换的tile，其次，在剩下的tile中，要修改ELL tile的output i范围，如果这个ELL tile和whole row 的TC tile范围有交叉
	whole_TC_blk_rows = np.nonzero(TC_row_costs==-1)[0]
	to_del = np.concatenate([TC_row_tile_dict[blk_row_i] for blk_row_i in whole_TC_blk_rows ] + [[]])
	remain_idx = np.setdiff1d(np.arange(len(selected_tiles)), to_del, assume_unique=True)
	remains = [selected_tiles[i] for i in remain_idx]
	# 
	for t in remains:
		if (get_template_str(t.op) == 'TensorCore_template'):
			if (not t.is_atomic_tile):
				# 需要调整position space的位置
				init_nnz(t)
				t.position_space_when_selected = t.uncovered_position_space
				t.nnz_when_selected = t.j_num * t.position_space_when_selected.nnz
				t.get_k_vals()
				t.tile_k_rng = None
			else:
				# 需要设置k_vals
				t.get_k_vals()
				t.tile_k_rng = None
		else:
			# ELL tile
			# 首先判断其output范围是否和whole-row TC tile有重合
			out_rows = t.op.idx_values_list[0][0][ t.op.hyb_new_rows[0][ t.tile_i_rng[0]:t.tile_i_rng[1]+1 ] ]
			rows = reverse_idx_TC_op_row[out_rows] // 16
			# 
			rows, idx = np.unique(rows, return_inverse=True)
			inter_rows, _, y_ind = np.intersect1d(whole_TC_blk_rows, rows, assume_unique=True, return_indices=True)
			# 得到tile的i range中哪些i是需要修改的
			t.i_vals = np.copy(out_rows)
			if len(inter_rows) > 0:
				rows = np.nonzero(functools.reduce(np.logical_or, [idx==i for i in y_ind]))[0]
				t.i_vals[rows] = ori_op.idx_lens[0]

				# 还是直接调整这个tile的position space好了。算了，还是先不管这个了。

				# 是否需要在重新调整一下is_atomic_tile？已经调整了
				if t.is_atomic_tile:
					rows = np.setdiff1d(np.arange(len(out_rows)), rows, assume_unique=True)
					if ori_op.inps[0][out_rows[rows]].nnz==t.position_space_when_selected[rows].nnz:
						t.is_atomic_tile = False

			t.tile_i_rng = None

			if not t.is_atomic_tile:
				# 还需要把t涉及到的空行调整为invalid的值
				rows = np.nonzero( t.position_space_when_selected.getnnz(axis=1)==0 )[0]
				t.i_vals[rows] = ori_op.idx_lens[0]

	return remains





# 这个函数用来整理支持withdraw之后得到的selected tiles
def post_process_for_withdraw_BSDDMM(ori_op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row):
	# 我们只会用到ori_op的idx_len
	# 需要准备TC tile和1D tile需要的Metadata，另外可能还需要调整一下各个tile所cover的non-zero集合？感觉是不是暂时不需要？暂时不要这个而且也还没有想到什么比较高效的实现方式。暂时不管

	ops_1d = list()
	ops_1d_id = list()
	for t in selected_tiles:
		op = t.op
		if op.op_id not in ops_1d_id:
			ops_1d_id.append(op.op_id)
			if get_template_str(op) == '1D_sddmm':
				op.nnz_id_matrix = ori_op.nnz_id_matrix.multiply(op.inps[0])[ op.idx_values_list[0][0],:][:,op.idx_values_list[0][2] ]
				op.nnz_id_matrix.eliminate_zeros()
				op.nnz_id_matrix.has_sorted_indices = False
				op.nnz_id_matrix.sort_indices()
				op.nnz_id_matrix = op.nnz_id_matrix.data-1
				# to_pad = get_pad_num(len(op.nnz_id_matrix), t.tile_sizes[0])
				# op.nnz_id_matrix = np.concatenate((op.nnz_id_matrix, np.full(to_pad, op.nnz_id_matrix[-1])))
				# 
				csr = op.inps[0][op.idx_values_list[0][0],:][:,op.idx_values_list[0][2]]
				rows = np.repeat(op.idx_values_list[0][0], np.diff(csr.indptr))
				# op.ori_row_ids_1d = np.concatenate((rows, np.full(to_pad, rows[-1])))
				op.ori_row_ids_1d = rows
				# 
				csr.has_sorted_indices = False
				csr.sort_indices()
				cols = op.idx_values_list[0][2][ csr.indices ]
				# op.ori_col_ids_1d = np.concatenate((cols, np.full(to_pad, cols[-1])))
				op.ori_col_ids_1d = cols
			elif get_template_str(op) == 'TC_sddmm':
				# NOTE: We do not check to_pad == 0 here because we initialize the idx_lens of the ori_op to be large enough (not multiples of TC tile shapes) for valid TC tiles.
				# to_pad = get_pad_num(len(op.idx_values_list[0][0]), t.tile_sizes[0][1])
				# assert to_pad == 0, f"{t.tile_sizes[0][1], len(op.idx_values_list[0][0]), to_pad}"
				# to_pad = get_pad_num(len(op.idx_values_list[0][2]), t.tile_sizes[2][1])
				# assert to_pad == 0, f"{t.tile_sizes[2][1], len(op.idx_values_list[0][2]), to_pad}"
				# 
				op.nnz_id_matrix = ori_op.nnz_id_matrix.multiply(op.inps[0])[ op.idx_values_list[0][0],:][:,op.idx_values_list[0][2] ]
				op.nnz_id_matrix.eliminate_zeros()

	a = [t for t in selected_tiles if get_template_str(t.op) == '1D_sddmm']
	b = [t for t in selected_tiles if get_template_str(t.op) == 'TC_sddmm']

	return a+b  # selected_tiles 现在的hybrid format实现只有当所有1D tile都被放在TC tile的前面的时候才能成功编译。



def post_process_for_withdraw(ori_op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row):
	if ori_op.op_type == 'spmm':
		return post_process_for_withdraw_BSPMM(ori_op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)
	elif ori_op.op_type == 'sddmm':
		return post_process_for_withdraw_BSDDMM(ori_op, selected_tiles, TC_row_tile_dict, TC_row_costs, reverse_idx_TC_op_row)
