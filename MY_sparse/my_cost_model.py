# This file estimate the cost of a computation tile (corresponds to a thread block)
# import gen_formats
# from importlib import reload
# gen_formats = reload(gen_formats)
 
from gen_formats_v2 import Op, SparseTensor, get_nnz_from_dict, get_val_2_ind, gen_position_space_for_area, get_template_str, gen_tile_sizes_given_tile_tb, get_tile_sizes_idx, get_tile_sizes_list
import math
import numpy as np

# <jingzhi>@revision: Allow other cost model
import os
import torch


# cost_dict = dict()
# from log_hub import data_128_128_128p_1218_2 # the file name should be changed

# now the value of cost_dict has been set



def compute_cost_tb_impl(op, tile_sizes, tile_pos, params):
	'''
	Estimate the computation cost of a computation tile which we know the concrete implementation.
	'''
	comp_cost = None
	area_i = op.this_area_i
	idx_values = op.idx_values_list[area_i]
	# gen_position_space_for_area(op, area_i)
	assert area_i == 0, "Each op should only represent one area."
	if op.op_type == 'spmm':
		comp_cost = 0
		# we need to first know the template corresponding to this recored
		template_str = get_template_str(op)
		if template_str == "sparse_template":
			# we do not tile on index k, but the computation below still deals with the case which tiles on k
			# there is no padding in this template
			tile_rngs = [(tile_pos[i]*math.prod(tile_sizes[i][1:]), (tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))
						for i in op.idxs]
			j_len = min(tile_rngs[1][1], len(idx_values[1])) - tile_rngs[1][0]
			
			row_nnzs = op.inp_getnnz[0][ tile_rngs[0][0]:tile_rngs[0][1] ]
			comp_cost = sum([ min(nnz, tile_rngs[2][1]) - min(nnz, tile_rngs[2][0])\
							for nnz in row_nnzs ]) * j_len * 2

		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			comp_cost = math.prod([math.prod(tile_sizes[i][1:]) for i in op.idxs]) * 2
		elif template_str == "TensorCore_template":
			print("This recored is about tensor cores, we do not need to estimate cost.")
		return comp_cost




def compute_cost_tb_impl_given_tile(tile):
	'''
	Estimate the computation cost of a computation tile which we know the concrete implementation.
	'''
	comp_cost = None
	op = tile.op
	if op.op_type == 'spmm':
		comp_cost = 0
		# we need to first know the template corresponding to this recored
		template_str = get_template_str(op)
		if template_str == "sparse_template":
			# we do not tile on index k, but the computation below still deals with the case which tiles on k
			# there is no padding in this template
			comp_cost = tile.nnz_uncovered * 2

		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			comp_cost = math.prod([math.prod(tile.tile_sizes[i][1:]) for i in op.idxs]) * 2
		elif template_str == "TensorCore_template":
			# print("This recored is about tensor cores, we do not need to estimate cost.")
			comp_cost = math.prod([math.prod(tile.tile_sizes[i][1:]) for i in op.idxs]) * 2
		return comp_cost



def compute_cost_tb_impl_given_tile_sizes(template_str, tile_sizes):
	'''
	Estimate the computation cost of a computation tile which we know the concrete implementation.
	'''
	comp_cost = 0
	if template_str == "sparse_template_ell":
		# the inputs are padded if necessary (on both index i, j, k)
		comp_cost = math.prod([math.prod(i[1:]) for i in tile_sizes]) * 2
	else:
		assert False, "Only support ELL tiles."
	return comp_cost




def get_index(point_id, space_shape):
	'''
		Get the index of the point_id in the given space with space_shape.
	'''
	index = list()
	for i in space_shape[::-1]:
		index.append(point_id%i)
		point_id = point_id // i
	return index[::-1]










def memory_cost_tb_impl(tile, tile_sizes, dsize):
	'''
	Estimate the memory transaction amount of a computation tile which we know the concrete implementation.
	tile_sizes: the concrete tiling sizes.
	# params: the concrete parameters.
	OUTPUT:
		不光会return memory cost, 还会return 单个warp在任意时刻对应的i的数量。
	'''
	# print("CALL memory_cost_tb_impl--------------")
	# print(tile.get_key(), tile_sizes)

	tile_pos = tile.tile_pos
	op = tile.op
	memory_cost = None
	idx_values = op.idx_values_list[0]
	cache_line_size = 32*8/dsize #  8 # i.e., 32 byte = 8 float32
	if op.op_type == 'spmm':
		memory_cost = 0
		template_str = get_template_str(op)
		# print(f"template_str{template_str}")

		warp_size = 32
		warp_num = int((tile_sizes[0][2]*tile_sizes[1][2]) / warp_size) # 我们限制thread number是warp size的倍数
		tile_rngs = [(tile_pos[i]*math.prod(tile_sizes[i][1:]), (tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))
					for i in op.idxs]

		thread_shape = [tile_sizes[i][2] for i in [0,1]] # j->thread.x, i->thread.y

		if template_str == "sparse_template":
			# we do not tile on index k, but the computation below still deals with the case which tiles on k
			# there is no padding in this template
			# 我们在这里需要考虑每个thread负责的部分，在template中，tile size的设置是[#block, #virtual thread, #thread].
			# in1，理论上来说，基本可以认为in1上的data load是不连续的。但是还是要考虑warp---------------------------------------

			# 但是下面的计算没有考虑：当in1的每行的nnz为1时，连续性也是很好的这一点，但是如果要精确地计算这一点，时间可能会很长，所以此处只能简单估计一下，但是简单估计的公式不太对，还是先精确计算吧。
			# 这个地方还是写错了,因为sparse_template这个模板里面写的还是j对于threadx，i对应thready。

			# =======================================================
			# 从这里开始
			cacheline_n = 0
			
			for i1 in range(tile_sizes[0][1]): # the number of i values for each thread
				for k_i in range(tile_rngs[2][0], tile_rngs[2][1]): 
					# k = idx_values[2][k_i]
					# 
					for warp_i in range(warp_num):
						start, end = warp_i * warp_num, (warp_i+1)*warp_num - 1
						start_idx, end_idx = get_index(start, thread_shape), get_index(end, thread_shape)
						# 存这个warp在这一轮次读取的in1中的元素在内存中实际存储的位置，其实是对应的cache_line id
						positions = list()
						for i2 in range(start_idx[0], end_idx[0]+1):
							new_i = i1*tile_sizes[0][2] + i2 + tile_rngs[0][0]
							if new_i >= len(idx_values[0]):
								continue
							if k_i >= tile.op.position_space_update.indptr[new_i+1]-tile.op.position_space_update.indptr[new_i]:
								positions.append((tile.op.position_space_update.indptr[new_i] + k_i) // cache_line_size)
						# 
						memory_cost = memory_cost + len(positions)

			# then compute the load cost for in2---------------------------------------
			# 之前为了能实现atomic addition （应该是在ell的模板的部分），把iteration的顺序切换成了ijk，但是我们的in2的layout和这个顺序并不契合，
			# 所以要么测量时间的时候，把in2进行转置，要么看看iteration的顺序能不能换成ikj
			# 其实和in2的layout没有什么矛盾的地方，因为我们希望一个warp在读取in2的时候内存是连续的即可。
			# 但是这个模板的时间没有问题
			cacheline_n = 0
			for i1 in range(tile_sizes[0][1]): # the number of i values for each thread
				for k_i in range(tile_rngs[2][0], tile_rngs[2][1]): 
					for j1 in range(tile_sizes[1][1]): # the number of j values for each thread

						for warp_i in range(warp_num):
							start, end = warp_i * warp_num, (warp_i+1)*warp_num - 1
							start_idx, end_idx = get_index(start, thread_shape), get_index(end, thread_shape)

							for i2 in range(start_idx[0], end_idx[0]+1):
								new_i = i1*tile_sizes[0][2] + i2 + tile_rngs[0][0]
								if new_i >= len(idx_values[0]):
									continue

								if k_i >= tile.op.position_space_update.indptr[new_i+1]-tile.op.position_space_update.indptr[new_i]:
									# do not need to load in2
									continue

								if i2 == start_idx[0]:
									j2_start, j2_end = start_idx[1], thread_shape[1]
								elif i2 == end_idx[0]:
									j2_start, j2_end = 0, end_idx[1] + 1
								else:
									j2_start, j2_end = 0, thread_shape[1]


								# 这个地方其实简化了一点，因为没有考虑k是否是相邻的取值
								cacheline_n = cacheline_n + math.ceil((j2_end - j2_start) / cache_line_size)

			memory_cost = memory_cost + cacheline_n


		elif template_str == "sparse_template_ell":
			# in1 
			# we use shared memory to store in1 here
			# 此处我们假定不对index k的维度进行tiling
			# 并且对于ell而言，所有input都被完美pad了
			# assert tile_rngs[2][1] - tile_rngs[2][0] == len(idx_values[2]), f"{tile_sizes} {len(idx_values[2])}We do not support tiling on index k now!"
			# 尽管我们不会在k轴tile，但是k轴的tile size并不是等于len(idx_values[2])， 而是等于几个固定值。
			memory_cost = memory_cost + math.ceil( (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) / cache_line_size )

			# 是否需要在此处考虑shared memory的cost？感觉不用了，因为不是一个量级的，不能直接相加。

			# in2
			# in2 is still loaded directly, and for this template, temporaroly, we directly load it into local memory
			# 按照 i->thread.y j->thread.x来计算
			# 感觉也得来一遍上面的计算过程，但是整个过程可以被简化

			# 整个计算的过程中其实并没有考虑到L1 cache会缓存！

			# 此处我们假设所有tile sizes都是2的指数倍，因此下面原有的计算in2的load cost的过程可以被化简
			cacheline_n = 0
			warp_i_size = None
			if thread_shape[1] > warp_size:
				# 说明每个warp在任意时刻都对应同一行，即同一个i
				cacheline_n = tile_sizes[0][1] * (tile_rngs[2][1] - tile_rngs[2][0]) * tile_sizes[1][1] * \
					warp_num * math.ceil(warp_size / cache_line_size)
				warp_i_size = 1
			else:
				# 每个warp在任意时刻对应>1个i值
				# 首先得到这个tile对应的in1的部分每一行的nnz数量
				nnzs = tile.op.position_space_update.getnnz(axis=1)[ tile.tile_i_rng[0] : tile.tile_i_rng[1] + 1 ]
				pad_i = math.prod(tile_sizes[0][1:]) - len(nnzs)
				nnzs = np.concatenate([nnzs, [0 for tmp in range(pad_i)]])
				warp_i_size = warp_size // thread_shape[1]
				nnzs_list = np.split(nnzs, math.prod(tile_sizes[0][1:])//warp_i_size )
				# 这个k_num 并没有考虑可能有重复的k值
				k_num = sum(nnzs) + sum([ math.prod(tile_sizes[2][1:]) - min(tmp) for tmp in nnzs_list])
				cacheline_n = k_num * math.ceil(thread_shape[1] / cache_line_size) * tile_sizes[1][1]

			memory_cost = memory_cost + cacheline_n
			return memory_cost, -warp_i_size # 我们认为warp_i_size越大cost越大，因为重复的k可能越多。
			# cacheline_n_new = cacheline_n

			# print(cacheline_n_new, "\nStart old comp\n\n")
			# cacheline_n = 0
			# for i1 in range(tile_sizes[0][1]): # the number of i values for each thread
			# 	for k_i in range(tile_rngs[2][0], tile_rngs[2][1]): 
			# 		for j1 in range(tile_sizes[1][1]): # the number of j values for each thread

			# 			for warp_i in range(warp_num):
			# 				start, end = warp_i * warp_size, (warp_i+1)*warp_size - 1
			# 				start_idx, end_idx = get_index(start, thread_shape), get_index(end, thread_shape)

			# 				same_line_j_rng = [thread_shape[1], 0] # for those padded 0 in in1, as their k are set to the same value
			# 				for i2 in range(start_idx[0], end_idx[0]+1):
			# 					same_line = False

			# 					new_i = i1*tile_sizes[0][2] + i2 + tile_rngs[0][0]
								
			# 					if new_i >= len(op.hyb_new_rows[0]):
			# 						same_line = True
			# 					else:
			# 						if k_i >= tile.op.position_space_update.indptr[new_i+1]-tile.op.position_space_update.indptr[new_i]:
			# 							same_line = True

			# 					j2_start, j2_end = None, None
			# 					if start_idx[0] == end_idx[0]:
			# 						j2_start, j2_end = start_idx[1], end_idx[1] + 1
			# 					else:
			# 						if i2 == start_idx[0]:
			# 							j2_start, j2_end = start_idx[1], thread_shape[1]
			# 						elif i2 == end_idx[0]:
			# 							j2_start, j2_end = 0, end_idx[1] + 1
			# 						else:
			# 							j2_start, j2_end = 0, thread_shape[1]

			# 					# print(j2_start, j2_end)
			# 					if same_line:
			# 						same_line_j_rng = [min(same_line_j_rng[0], j2_start), max(same_line_j_rng[1], j2_end)]
			# 					else:
			# 						# 这个地方其实简化了一点，因为没有考虑k是否是相邻的取值
			# 						cacheline_n = cacheline_n + math.ceil((j2_end - j2_start) / cache_line_size)

			# 				if same_line_j_rng[0] <= same_line_j_rng[1]:
			# 					cacheline_n = cacheline_n + math.ceil((same_line_j_rng[1] - same_line_j_rng[0]) / cache_line_size)

			# 				# print(cacheline_n)
			# memory_cost = memory_cost + cacheline_n
			# assert cacheline_n_new == cacheline_n, print(cacheline_n_new, cacheline_n, tile_sizes)

		elif template_str == "TensorCore_template":
			assert False, "This recored is about tensor cores, we do not need to estimate cost."
		return memory_cost





# 在忽略具体tile size的cost model中，我们应该计算的是cost的lower bound。-------------------------------------

def compute_cost_tb(op, tile_sizes, tile_pos, params):
	'''
	We omit the concrete tiling sizes in each kind of template here.
	tile_sizes: only tells the workload of a thread block.
	'''
	return compute_cost_tb_impl(op, tile_sizes, tile_pos, params)





def memory_cost_tb(op, tile_sizes, tile_pos, params):
	'''
	We omit the concrete tiling sizes in each kind of template here.
	tile_sizes: only tells the workload of a thread block.
	'''
	memory_cost = None
	area_i = op.this_area_i
	d_size = 32 / 8 # unit is byte
	assert area_i == 0, "Each op should only represent one area."
	idx_values = op.idx_values_list[area_i]
	cache_line_size = 8 # i.e., 32 byte = 4 float32
	template_str = get_template_str(op)
	

	if op.op_type == 'spmm':  

		memory_cost = 0
		tile_rngs = [(tile_pos[i]*math.prod(tile_sizes[i][1:]), (tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))
					for i in op.idxs]

		if template_str == "sparse_template":
			# 此处基于的假设是：我们考虑每个数据只被读取一遍。
			# in1---------------
			# 假定每个warp读取in1的时候都是读取相同的位置的值，因为这样可能可以使相同warp数的情况下，load的次数尽可能少？
			k_is = list()
			cacheline_n = 0
			for i in range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))):
				# print("row_i", i)

				indices = get_nnz_from_dict(op.position_space[area_i], [i])[ tile_rngs[2][0] : tile_rngs[2][1] ]
				indices = idx_values[2][indices]

				cacheline_n = cacheline_n + len(indices)  # op.inps[0].nnz_num((i,))			
				k_is = k_is + list(indices)


			# in2---------------
			# 假定单个warp内的线程对于in2的读取都是连续的
			k_is = set(k_is)
			j_len = min(tile_rngs[1][1], len(idx_values[1])) - tile_rngs[1][0]
			cacheline_n = cacheline_n + math.ceil(j_len / cache_line_size) * len(k_is)
			memory_cost = memory_cost + cacheline_n

		elif template_str == "sparse_template_ell":
			# 此处基于的假设依然是：我们考虑每个数据只被读取一遍，同时，对于in1的读取，是连续的，并且使用了shared memory作为缓存。
			# in1------------------
			memory_cost = memory_cost + math.ceil( (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) / cache_line_size )
			# in2------------------
			k_is = list()
			for i in range(tile_rngs[0][0], tile_rngs[0][1]):
				if i >= len(op.hyb_new_rows[0]):
					k_is.append(0)
				else:
					# =====================
					# 此处并没有对tile之间在k轴上划分workload这一情况做处理
					indices = get_nnz_from_dict(op.position_space[area_i], [i])
					indices = idx_values[2][indices]
					k_is = k_is + list(indices)
					if len(indices) < (tile_rngs[2][1] - tile_rngs[2][0]):
						k_is.append(0)

			k_is = set(k_is)
			j_len = tile_rngs[1][1] - tile_rngs[1][0]
			memory_cost = memory_cost + math.ceil(j_len / cache_line_size) * len(k_is)

		elif template_str == "TensorCore_template":
			print("This recored is about tensor cores, we do not need to estimate cost.")
			return -1
		return memory_cost




def memory_cost_tb_given_tile(tile, dsize): 
	'''
	We omit the concrete tiling sizes in each kind of template here.
	tile_sizes: only tells the workload of a thread block.
	INPUT:	dsize: the number of bits of a data. The default is 32 bit (float32).

	NOTE: 目前暂时只实现了ELL tile的store cost的更新, 别的tile都没有计算这个cost.
	OUTPUT: 对于ELL tile, 会同时返回non-atomic和atomic版本的memory cost
	'''
	memory_cost = None
	op = tile.op
	tile_pos = tile.tile_pos
	tile_sizes = tile.tile_sizes

	area_i = op.this_area_i
	# d_size = dsize / 8 # unit is byte
	assert area_i == 0, "Each op should only represent one area."
	idx_values = op.idx_values_list[area_i]
	cache_line_size = 32*8/dsize #  8 # i.e., 32 byte = 8 float32
	bank_width = 32 # shared memory 中一个bank是32 bit
	template_str = get_template_str(op)
	

	if op.op_type == 'spmm':  

		memory_cost = 0
		tile_rngs = [(tile_pos[i]*math.prod(tile_sizes[i][1:]), (tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))
					for i in op.idxs]

		if template_str == "sparse_template":
			# 此处基于的假设是：我们考虑每个数据只被读取一遍。
			# in1---------------
			# 假定每个warp读取in1的时候都是读取相同的位置的值，因为这样可能可以使相同warp数的情况下，load的次数尽可能少？
			cacheline_n = tile.nnz
			# k_is = idx_values[2][np.array(set(tile.uncovered_position_space['in1'].indices))]

			# 此处我们是针对不在initialize Tile的时候设置其uncovered position space的情况计算cost
			indptr = tile.op.position_space[0].indptr
			inds = indptr[ tile_rngs[0][0] ], \
				indptr[ min(tile_rngs[0][1], len(indptr)-1) ]
			k_is = set(tile.op.position_space[0].indices[ inds[0]:inds[1] ])

			# in2---------------
			# 假定单个warp内的线程对于in2的读取都是连续的
			# k_is = set(k_is)
			# j_len = min(tile_rngs[1][1], len(idx_values[1])) - tile_rngs[1][0]
			j_len = tile.j_num

			cacheline_n = cacheline_n + math.ceil(j_len / cache_line_size) * len(k_is)
			memory_cost = memory_cost + cacheline_n

			memory_cost = [memory_cost, memory_cost]

		elif template_str == "sparse_template_ell":
			# 此处基于的假设依然是：我们考虑每个数据只被读取一遍，同时，对于in1的读取，是连续的，并且使用了shared memory作为缓存。
			# in1------------------
			memory_cost = memory_cost + math.ceil( (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) / cache_line_size )
			
			# check the shared memory constraint
			if (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) * dsize > 49152 * 8:
				return float('inf')

			# FOR DEBUG 假设我们规定每个tile内thread的数量都相同，为了是的occupancy最大化，我们对shared memory有要求
			# 假设每个tile分给128个线程，则A100上一个SM最多同时有16个block
			# NOTE: 此处我们考虑为了避免bank conflict而做的一些shared memory上的padding所占用的空间，使用upper bound.
			# if ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])) * dsize + \
			# 	(tile_rngs[0][1] - tile_rngs[0][0]) * bank_width > 167936 * 8 / 8:
			# if ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])) * dsize > 167936 * 8 / 8:
			# 	return float('inf')

			# in2------------------
			# k_is = idx_values[2][np.array(set(tile.uncovered_position_space['in1'].indices))]
			# if tile.nnz < ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])):
			# 	k_is = set(k_is).add(0)

			# 此处我们是针对不在initialize Tile的时候设置其uncovered position space的情况计算cost
			indptr = tile.op.position_space[0].indptr
			inds = indptr[ tile_rngs[0][0] ], \
				indptr[ min(tile_rngs[0][1], len(indptr)-1) ]
			k_is = set(tile.op.position_space[0].indices[ inds[0]:inds[1] ])
			# k_is = idx_values[2][np.array(k_is)]
			if tile.nnz < ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])):
				k_is.add(0)

			# TODO: 这个地方的j_len是不是有点问题，因为这里还是假定我们会对j轴也进行padding，但是事实可能是不padding
			# 但是好像要改得话，也只是改这里就ok了
			j_len = tile_rngs[1][1] - tile_rngs[1][0]
			memory_cost = memory_cost + math.ceil(j_len / cache_line_size) * len(k_is)

			# 还需要考虑store cost
			cacheline_bit = 32*8
			warp_size = 32
			store_cost = math.ceil(math.prod(tile_sizes[1][1:]) * dsize / cacheline_bit) * math.prod(tile_sizes[0][1:])
			# if is_atomic:
			# 	store_cost = store_cost * 2
			# memory_cost = memory_cost + store_cost

			memory_cost = [memory_cost + store_cost, memory_cost + store_cost * 2]


		elif template_str == "TensorCore_template":
			# print("This recored is about tensor cores, we do not need to estimate cost.")
			# assert False
			# return -1
			# 我们的TC schedule如下：把A读进register中（合作读取的），把B先读进shared memory中，再读进register中，
			# 最后把结果先写回shared memory （可能涉及到i的reorder），再写回global memory。

			# ！之前对ELL和CSR的cost的计算其实并没有加上写回的cost，是不是要加上？暂时都不加，之后再加
			# in1
			memory_cost = math.prod(tile_sizes[0][1:]) * (math.prod(tile_sizes[2][1:]) / cache_line_size)
			# in2
			memory_cost = memory_cost + math.prod(tile_sizes[1][1:]) * math.prod(tile_sizes[2][1:]) / cache_line_size

			memory_cost = [memory_cost, memory_cost]

		return memory_cost





# 这个函数批量计算给定tile_sizes在多个tile_poses各自的memory cost
# 目前只支持ELL tile。
def memory_cost_tb_given_tile_sizes_and_poses(tile_sizes, tile_pos_is, template_str, dsize, op): 
	'''
	We omit the concrete tiling sizes in each kind of template here.
	tile_sizes: only tells the workload of a thread block.
	INPUT:	dsize: the number of bits of a data. The default is 32 bit (float32).

	NOTE: 目前暂时只实现了ELL tile的store cost的更新, 别的tile都没有计算这个cost.
	OUTPUT: 对于ELL tile, 会同时返回non-atomic和atomic版本的memory cost
	'''
	cache_line_size = 32*8/dsize #  8 # i.e., 32 byte = 8 float32
	# bank_width = 32 # shared memory 中一个bank是32 bit

	memory_cost = 0
	if template_str == "sparse_template_ell":
		# 此处基于的假设依然是：我们考虑每个数据只被读取一遍，同时，对于in1的读取，是连续的，并且使用了shared memory作为缓存。
		# in1------------------
		memory_cost_A = math.ceil( tile_sizes[0][1] * tile_sizes[2][1] / cache_line_size )
		
		# check the shared memory constraint
		# if (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) * dsize > 49152 * 8:
		# 	return float('inf')

		# FOR DEBUG 假设我们规定每个tile内thread的数量都相同，为了是的occupancy最大化，我们对shared memory有要求
		# 假设每个tile分给128个线程，则A100上一个SM最多同时有16个block
		# NOTE: 此处我们考虑为了避免bank conflict而做的一些shared memory上的padding所占用的空间，使用upper bound.
		# if ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])) * dsize + \
		# 	(tile_rngs[0][1] - tile_rngs[0][0]) * bank_width > 167936 * 8 / 8:
		# if ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])) * dsize > 167936 * 8 / 8:
		# 	return float('inf')

		# in2------------------
		# k_is = idx_values[2][np.array(set(tile.uncovered_position_space['in1'].indices))]
		# if tile.nnz < ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])):
		# 	k_is = set(k_is).add(0)

		# 此处我们是针对不在initialize Tile的时候设置其uncovered position space的情况计算cost
		# TODO: 这个地方的j_len是不是有点问题，因为这里还是假定我们会对j轴也进行padding，但是事实可能是不padding
		# 但是好像要改得话，也只是改这里就ok了


		# tile_pos_is = np.asarray([i[0] for i in tile_poses])
		# print("tile_pos_is:", tile_pos_is)
		indptr = op.position_space[0].indptr
		inds_start = indptr[tile_sizes[0][1]*tile_pos_is]
		inds_ends = indptr[ np.minimum(tile_sizes[0][1]*(tile_pos_is+1), len(indptr)-1) ]
		pads = (inds_ends - inds_start) < (tile_sizes[0][1] * tile_sizes[2][1])
		# print("pads:", pads, inds_ends, inds_start)
		k_is = [ len(np.unique(np.concatenate((op.position_space[0].indices[ inds_start[i]:inds_ends[i] ], [0])))) if pads[i] \
				else\
				len(np.unique( op.position_space[0].indices[ inds_start[i]:inds_ends[i] ] ))\
				for i in range(len(pads))]
		j_len = tile_sizes[1][1]
		memory_cost_B = math.ceil(j_len / cache_line_size) * np.asarray(k_is)


		# 还需要考虑store cost
		cacheline_bit = 32*8
		warp_size = 32
		store_cost = math.ceil(tile_sizes[1][1] * dsize / cacheline_bit) * tile_sizes[0][1]

		memory_cost = [memory_cost_A + store_cost + memory_cost_B, memory_cost_A + store_cost * 2 + memory_cost_B]

	else:
		assert False, "We only support ELL tiles now."

	return memory_cost






def cost_tb(op, tile_sizes, tile_pos, params):
	compute_cost = compute_cost_tb(op, tile_sizes, tile_pos, params) 
	memory_cost = memory_cost_tb(op, tile_sizes, tile_pos, params)
	# print(compute_cost, memory_cost)

	# 默认调用此函数的tile不可能没有计算和memory开销。暂时取消这一默认。
	if memory_cost == 0:
		# print(op.op_id, tile_sizes, tile_pos, params)
		assert compute_cost == 0
		return 0  # 因为这种tile并不可以被选择，所以我们将其throughput设为0，然后在计算latency的时候将其latency设为inf，即avg_cost也inf
	return min(compute_cost / (memory_cost * 32), 19.49/1.555)



def cost_tb_given_tile(tile, dsize):
	# TC tile的峰值速率应该不是624，有点问题，应该是312
	# memory_cost 现在会同时返回non-atomic版本和atomic-版本的，虽然我们现在只支持了ELL tile的。
	compute_cost = compute_cost_tb_impl_given_tile(tile) 
	# memory_cost = memory_cost_tb_given_tile(tile, dsize)
	memory_costs = memory_cost_tb_given_tile(tile, dsize)
	# print(compute_cost, memory_costs)

	# 默认调用此函数的tile不可能没有计算和memory开销。暂时取消这一默认。
	# if memory_cost == 0:
	if 0 in [memory_costs]:
		# print(op.op_id, tile_sizes, tile_pos, params)
		assert compute_cost == 0
		return 0, 0  # 因为这种tile并不可以被选择，所以我们将其throughput设为0，然后在计算latency的时候将其latency设为inf，即avg_cost也inf
	if get_template_str(tile.op) == 'TensorCore_template':
		# 此处的624是假设我们用的是float16
		# FOR DEBUG===========================
		# 此处throughput乘10，是基于实验观察到的假设，感觉还是得换cost model，基于真实latency。
		# return min(compute_cost / (memory_cost * 32) * 10, 312/1.555) #624/1.555)  312/1.555) #

		return [min(compute_cost / (memory_cost * 32) * 10, 312/1.555) for memory_cost in memory_costs] #624/1.555)  312/1.555) #

		# ====================================
	else:
		if dsize == 32:
			return [min(compute_cost / (memory_cost * 32), 19.49/1.555) for memory_cost in memory_costs]
		elif dsize == 16:
			return [min(compute_cost / (memory_cost * 32), 78/1.555) for memory_cost in memory_costs]








def get_benchmark_cost_(template_str, tile_sizes, dsize, is_atomic, penalty):
	'''
	penalty: the performance degradation ratio computed by TC tile occupancy.
	'''
	if template_str == 'TensorCore_template':
		if dsize == 16:  # 暂时先认为两种dtype的TC tile cost一样，我们得到benchmark结果之后再修正。
			# output dsize is 16 in this case
			# 以下数据是lccpu27上的，但是因为实验在lccpu28上做，所以更新cost数据
			# TC_cost_dict = {
			# 	(16, 32, 16): 0.03913872/100, 
			# 	(16, 32, 32): 0.05328592/100,
			# 	(16, 32, 48): 0.06522246/100,
			# 	(16, 32, 64): 0.07834872/100,
			# 	(16, 32, 80): 0.10675935/100,
			# 	(16, 32, 96): 0.10595788/100,
			# 	(16, 32, 112): 0.11501513/100,
			# 	(16, 32, 128): 0.13/100, # 这个结果需要更新
			# 	(16, 32, 144): 0.14248087/100,
			# 	(16, 32, 160): 0.15248087/100, # 这个结果需要更新 
			# 	}
			# TC_cost_dict_atomic = {
			# 	(16, 32, 16): 0.13574204/100, 
			# 	(16, 32, 32): 0.14316941/100,
			# 	(16, 32, 48): 0.15268948/100,
			# 	(16, 32, 64): 0.16785308/100,
			# 	(16, 32, 80): 0.18458567/100,
			# 	(16, 32, 96): 0.20565027/100,
			# 	(16, 32, 112): 0.21700917/100,
			# 	(16, 32, 128): 0.24/100, # 这个结果需要更新
			# 	(16, 32, 144): 0.2612427/100,
			# 	(16, 32, 160): 0.2812427/100, # 这个结果需要更新
			# 	}
			# 以下是lccpu28上的数据
			TC_cost_dict = {
				(16, 32, 16): 0.0002922496385872364,
				(16, 32, 32): 0.00041533440351486205,
				(16, 32, 48): 0.0005173248425126075,
				(16, 32, 64): 0.0006273023784160614,
				(16, 32, 80): 0.0007631871849298477,
				(16, 32, 96): 0.0008801279217004776,
				(16, 32, 112): 0.0009804800152778625,
				(16, 32, 128): 0.001063014417886734,
				(16, 32, 144): 0.0012102656066417695,
				(16, 32, 160): 0.001244262307882309,
				}
			TC_cost_dict_atomic = {
				(16, 32, 16): 0.0009056255966424942,
				(16, 32, 32): 0.0010332158952951432,
				(16, 32, 48): 0.001133260801434517,
				(16, 32, 64): 0.0012712959945201873,
				(16, 32, 80): 0.0014235648512840272,
				(16, 32, 96): 0.0015714305639266968,
				(16, 32, 112): 0.0017159168422222137,
				(16, 32, 128): 0.0018328575789928437,
				(16, 32, 144): 0.0021766144037246703,
				(16, 32, 160): 0.0021382145583629607,
				}

			# tile_sizes = tile.tile_sizes
			I, J, K = tile_sizes[0][1], tile_sizes[1][1], tile_sizes[2][1]
			key = (I, J, K)
			if is_atomic:
				return TC_cost_dict_atomic[key] / penalty # 暂时假设只能达到满载效率的70%
			else:
				return TC_cost_dict[key] / penalty # 暂时假设只能达到满载效率的70%
		else:
			assert False, "we only support output type fp16 for TC tiles."
	elif template_str == 'sparse_template_ell':
		if dsize == 16:
			ELL_cost_dict = {
				(256, 32, 1): 0.342255333/100,
				(128, 32, 2): 0.27794799/100,
				(64, 32, 4): 0.214812706/100,
				(32, 32, 8): 0.189769996/100,
				(16, 32, 16): 0.170571889/100,
				(8, 32, 32): 0.226909105/100,
			}
			ELL_cost_dict_atomic = {
				(256, 32, 1): 1.324675103/100,
				(128, 32, 2): 0.866597613/100,
				(64, 32, 4): 0.493691849/100,
				(32, 32, 8): 0.315331313/100,
				(16, 32, 16): 0.229816495/100,
				(8, 32, 32): 0.240047723/100,
			}
			# tile_sizes = tile.tile_sizes
			I, J, K = tile_sizes[0][1], tile_sizes[1][1], tile_sizes[2][1]
			key = (I, J, K)
			if is_atomic:
				return ELL_cost_dict_atomic[key]
			else:
				return ELL_cost_dict[key]
		else:
			assert False, "we only support output type fp16 for ELL tiles."
	elif template_str == 'TC_sddmm':
		# return 1 # TODO SDDMM: 暂时先随便设一个数据，之后再重新补充。
		if dsize == 16:
			TC_cost_dict = {
				(16, 16, 16): 0.019077118486166/100,
				(16, 32, 16): 0.027914240956306458/100,
				(16, 64, 16): 0.04804608225822449/100,
				(16, 16, 32): 0.026531841605901718/100,
				(16, 32, 32): 0.0391475185751915/100,
				(16, 64, 32): 0.06515711545944214/100,
				(16, 16, 48): 0.03529728576540947/100,
				(16, 32, 48): 0.05030912533402443/100,
				(16, 64, 48): 0.07875583320856094/100,
				(16, 16, 64): 0.044257279485464096/100,
				(16, 32, 64): 0.06239231675863266/100,
				(16, 64, 64): 0.10551295429468155/100,
				(16, 16, 80): 0.05752832442522049/100,
				(16, 32, 80): 0.07265280187129974/100,
				(16, 64, 80): 0.118947833776474/100,
				(16, 16, 96): 0.06372351944446564/100,
				(16, 32, 96): 0.08369152247905731/100,
				(16, 64, 96): 0.1488281637430191/100,
				(16, 16, 112): 0.07803903520107269/100,
				(16, 32, 112): 0.09360384196043015/100,
				(16, 64, 112): 0.15593470633029938/100,
				(16, 16, 128): 0.07907328009605408/100,
				(16, 32, 128): 0.10522624105215073/100,
				(16, 64, 128): 0.18569216132164001/100,
			}
			I, J, K = tile_sizes[0][1], tile_sizes[1][1], tile_sizes[2][1]
			key = (I, J, K)
			return TC_cost_dict[key]
		else:
			assert False, "we only support output type fp16 for TC tiles."
	else:
		assert False, "we only support TC and ELL tiles."



def get_benchmark_cost(tile, dsize, is_atomic, penalty):
	template_str = get_template_str(tile.op)
	tile_sizes = tile.tile_sizes
	return get_benchmark_cost_(template_str, tile_sizes, dsize, is_atomic, penalty)






# def ELL_linear_cost_model_coeff():
	# return [2.95507917e-07, -1.97445294e-03]
	# return [1, 0] # 变回naive cost model，以先获得citeseer上的ELL tile的benchMark data

def ELL_non_atomic_coeff(dsize):
	if dsize == 16:
		# return [9.74636589e-08, 7.22826260e-04] # 这个应该是lccpu27上的数据
		return [8.80425616e-08, 4.17697078e-04] # 这个应该是lccpu28上的数据
	else:
		# dsize == 32
		return [8.23071690e-08, 3.42962663e-03]

def ELL_atomic_extra_coeff(dsize):
	if dsize == 16:
		# return [4.62653486, -3.74684073] # 这个应该是lccpu27上的数据
		return [ 8.80688635, -7.9094305 ] # 这个应该是lccpu28上的数据
	else:
		# dsize == 32
		return [0.67944825, 0.29144282]


# def ELL_linear_cost_model(x):
# 	coeff = ELL_linear_cost_model_coeff()
# 	return x * coeff[0] + coeff[1]


def ELL_cost_model(x_nonatomic, x_atomic, is_atomic, dsize):
	# 我们分别用两个线性函数来拟合non-atomic的cost以及atomic的影响
	a1, b1 = ELL_non_atomic_coeff(dsize)
	a2, b2 = ELL_atomic_extra_coeff(dsize)
	if is_atomic:
		return (a1 * x_nonatomic + b1) * (a2 * x_atomic/x_nonatomic + b2)
	else:
		return (a1 * x_nonatomic + b1)




def lower_bound_ELL_cost_BSPMM(sub_op, max_bucket_size, dsize):
	'''
	Compute the best possible pred avg cost of an ELL tile, so that we can select more TC tiles at once.
	NOTE: 此处假定atomic一定比non-atomic的慢。所以我们只考虑non-atomic的版本就好了。
	'''
	is_tb_tile = True
	tile_sizes_idx = get_tile_sizes_idx(sub_op, max_bucket_size, is_tb_tile=is_tb_tile)
	if sub_op.op_type == 'spmm':
		# we know with larger tile.J, the average cost will be larger. so we only need to consider
		# tile_sizes_idx[1] = [sorted(tile_sizes_idx[1], key=lambda tmp: tmp[1])[-1]]

		# FOR DEBUG 因为猜测j越大的时候，nvcc在编译的过程中产生的local register的数量可能越多，从而会降低能同时在SM运行的block的数量
		# 因此此处限定tile 的j size为32
		tile_sizes_idx[1] = [(None, 32)]
		# ===============================
	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=is_tb_tile)
	cacheline_bit = 32*8
	pred_avg_costs = list()
	for tile_sizes in tile_sizes_list:
		I, J, K = [i[1] for i in tile_sizes]
		# 
		# 不需要做下面的判断因为在 get_tile_sizes_list 里面已经过滤掉不符合这些要求的了
		# if I*K != 256:
		# 	continue
		# if I*J < 256:
		# 	continue
		# 
		compute_cost = math.prod([I, J, K]) * 2
		# 计算最好的non-atomic的memory cost
		memory_cost = math.ceil(I*K*dsize/cacheline_bit) + math.ceil(J*dsize/cacheline_bit)*(K//2+1) + I*math.ceil(J*dsize/cacheline_bit)
		# 计算最差的non-atomic的memory cost
		# mem_non_atomic_max = math.ceil(I*K*dsize/cacheline_bit) + math.ceil(J*dsize/cacheline_bit)*(K*I) + I*math.ceil(J*dsize/cacheline_bit)
		# 
		tp_non_atomic = None
		# 
		if dsize == 32:
			tp_non_atomic = min(compute_cost / (memory_cost * 32), 19.49/1.555)
		elif dsize == 16:
			tp_non_atomic = min(compute_cost / (memory_cost * 32), 78/1.555)
		# 
		pred_avg_cost = ELL_cost_model(compute_cost / tp_non_atomic, None, False, dsize) / (I*K*J)
		pred_avg_costs.append(pred_avg_cost)
		print(f"{tile_sizes}, {pred_avg_cost}, max_bucket_size[{max_bucket_size}]")
	# 
	return min(pred_avg_costs)



def lower_bound_1D_cost_BSDDMM(sub_op, max_bucket_size, dsize):
	# # 暂时先随便设一个值，之后当我们确定了candidate tile sizes的时候在重新写
	# return -1

	# NOTE: 如果cost model 的 coefficient和计算方式改了的话，这个函数还需要修改-->现在不需要改，只需改_cost_tb_latency_1D_tiles_given_features

	# 此处我们求最小可能得memory cost，即row_num + col_num最小
	# 这个地方的最小memory cost不好求啊，因为可能存在nnz不满max_bucket_size的情况，感觉这样的话，如果要严格一点，我们的lower bound可能会非常松，不管了。
	# memory_costs = list()
	# for row_num in range(1, max_bucket_size+1):
	# 	memory_costs.append( row_num + max_bucket_size )
	# 
	# 我们在此处分情况讨论一下
	row_nums = np.concatenate([range(1, nnz+1) for nnz in range(1, max_bucket_size+1)])
	nnzs = np.repeat(np.arange(1, max_bucket_size+1), np.arange(1, max_bucket_size+1))
	col_nums = nnzs/row_nums
	avg_costs = _cost_tb_latency_1D_tiles_given_features(row_nums, col_nums, sub_op.idx_lens[1], (max_bucket_size, ), dsize) / nnzs / sub_op.idx_lens[1]
	# print(list(avg_costs))
	assert min(avg_costs) > 0, "Negative lower bound."
	return min(avg_costs)





def lower_bound_ELL_cost(sub_op, max_bucket_size, dsize):

	# <jingzhi>@ revision:
	# for other cost model (e.g., MLP), we do not have lower bound for ELL tiles
	if 'other_cost_model' in os.environ:
		if os.environ['other_cost_model'] == 'MLP':	
			return 0

	if sub_op.op_type == 'spmm':
		return lower_bound_ELL_cost_BSPMM(sub_op, max_bucket_size, dsize)
	elif sub_op.op_type == 'sddmm':
		return lower_bound_1D_cost_BSDDMM(sub_op, max_bucket_size, dsize)
















# <jingzhi>@revision
# compute the features in a batch, for higher speed
# the features we compute here will also be a bit different from in get_MLP_features
# A faster version
def get_MLP_features_in_batch(op, tile_sizes, tile_pos_is):

	# print("MLP: run in batches")
	# import time
	# time1 = time.time()

	features = list()
	row_num = tile_sizes[0][1]
	col_num = op.idx_lens[2]

	tile_i_rngs = [tile_sizes[0][1] * tile_pos_is, np.minimum(tile_sizes[0][1] * (tile_pos_is+1), op.position_space[0].shape[0]) - 1]
	# csr = op.position_space[0][range(tile_rngs[0][0], min(tile_rngs[0][1], len(op.hyb_new_rows[0]))),:]
	

	densitys = (op.position_space[0].indptr[tile_i_rngs[1] + 1] - op.position_space[0].indptr[tile_i_rngs[0]])/row_num/col_num

	nnzs = op.position_space[0].getnnz(axis = 1)
	valid_row_nums = tile_i_rngs[1]+1 - tile_i_rngs[0]

	nnzs_by_rows = np.zeros((len(tile_pos_is), row_num))
	for i in range(len(tile_pos_is)):
		nnzs_by_rows[i][:valid_row_nums[i]] = nnzs[ tile_i_rngs[0][i]:tile_i_rngs[1][i]+1 ]

	nnzs_by_rows_feats = np.mean(nnzs_by_rows, axis=1), np.max(nnzs_by_rows, axis=1), np.min(nnzs_by_rows, axis=1)
	# 
	same_cache_lines = np.zeros( (len(tile_pos_is), row_num, tile_sizes[2][1]) )
	
	indices = op.position_space[0].indices
	indptr = op.position_space[0].indptr

	for i in range(len(tile_pos_is)):
		for j in range(valid_row_nums[i]):		
			same_cache_lines[i][j][ :nnzs[ tile_i_rngs[0][i]+j ] ] = indices[ indptr[ tile_i_rngs[0][i]+j ]:indptr[ tile_i_rngs[0][i]+j+1 ] ]


	same_cache_lines = same_cache_lines.reshape( (len(tile_pos_is), -1) )
	distinct_cache_lines_perwarps = np.zeros(len(tile_pos_is))
	same_cache_lines_feats = [np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is))]
	total_cache_lines_perwarps = np.zeros(len(tile_pos_is))
	for i, cache_lines in enumerate(same_cache_lines):
		tmp, counts = np.unique(cache_lines, return_counts=True)
		same_cache_lines_feats[0][i] = np.mean(counts)
		same_cache_lines_feats[1][i] = np.max(counts)
		same_cache_lines_feats[2][i] = np.min(counts)
		distinct_cache_lines_perwarps[i] = len(tmp)
		total_cache_lines_perwarps[i] = sum(counts) # should be the same for all ELL blocks, may delete it
	# 
	adjacent_vector_distances = np.zeros((len(tile_pos_is), row_num-1))
	for i, cache_lines in enumerate(same_cache_lines):
		tmp = cache_lines.reshape((row_num, tile_sizes[2][1]))
		adjacent_vector_distances[i] = np.asarray([ len(np.union1d(tmp[j], tmp[j+1])) - len(np.intersect1d(tmp[j], tmp[j+1])) for j in range(row_num-1) ])
	adjacent_vector_distances_feats = np.mean(adjacent_vector_distances, axis=1), np.max(adjacent_vector_distances, axis=1), np.min(adjacent_vector_distances, axis=1)
	# 
	# add feature about the unique output row number and total output row number
	disctict_row_nums = np.zeros(len(tile_pos_is))
	same_row_nums_feats = [np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is)), np.zeros(len(tile_pos_is))]
	for i in range(len(tile_pos_is)):
		indices_reordered_i = np.full(row_num, op.hyb_new_rows[0][-1])
		start, end = tile_i_rngs[0][i], tile_i_rngs[1][i]
		indices_reordered_i[:end+1-start] = op.hyb_new_rows[0][start: end+1]
		tmp, counts = np.unique(indices_reordered_i, return_counts=True)
		disctict_row_nums[i] = len(tmp)
		same_row_nums_feats[0][i] = np.mean(counts)
		same_row_nums_feats[1][i] = np.max(counts)
		same_row_nums_feats[2][i] = np.min(counts)
	# 
	features = [np.full(len(tile_pos_is), row_num), np.full(len(tile_pos_is), tile_sizes[2][1]), densitys, *nnzs_by_rows_feats, *same_cache_lines_feats, 
		distinct_cache_lines_perwarps, total_cache_lines_perwarps, *adjacent_vector_distances_feats, 
		disctict_row_nums, *same_row_nums_feats]
	features = np.asarray(features).astype('float').T


	# time2 = time.time()
	# print(f"time2-time1: {time2-time1}, num of tiles: {len(tile_pos_is)}, avg compute time: {(time2-time1)/len(tile_pos_is)}")

	return features









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

			# output layer
			torch.nn.Linear(32, 1),
		)

	def forward(self, x):
		logits = self.all_layers(x)
		return logits




def load_model(PATH = 'trained_model', param_path = 'preprcess_param.pt'):
	loaded = torch.load(param_path)
	num_features = loaded['num_features']
	means = loaded['means']
	stds = loaded['stds']
	# PATH = 'trained_model'
	saved_model = PyTorchMLP(num_features)
	saved_model.load_state_dict(torch.load(PATH))#, map_location="cuda:0"))
	# 
	# device = torch.device("cuda")
	# saved_model.to(device)
	# saved_model = saved_model.to(0)
	# 
	saved_model.eval()
	return saved_model, means, stds




MLP_nonatomic, means_nonatomic, std_nonatomic = load_model(PATH = 'trained_model', param_path = 'preprcess_param.pt')
MLP_atomic, means_atomic, std_atomic = load_model(PATH = 'trained_model_atomic', param_path = 'preprcess_param_atomic.pt')


# <jingzhi>@revision
# use the trained MLP cost model
def get_MLP_pred(tile, is_atomic):
	features = get_MLP_features_in_batch(tile.op, tile.tile_sizes, [tile.tile_pos[0]])
	features = torch.as_tensor(features, dtype=torch.float)
	with torch.no_grad():
		if is_atomic:
			features = (features - means_atomic) / std_atomic
			# features = features.to(0)
			return MLP_atomic(features).numpy()[0][0]/100
		else:
			features = (features - means_nonatomic) / std_nonatomic
			# features = features.to(0)
			return MLP_nonatomic(features).numpy()[0][0]/100




# <jingzhi>@revision
# use the trained MLP cost model
# predict based on op, tile_sizes and tile_pos information directly
def get_MLP_pred_no_tile_given(op, tile_sizes, tile_pos_is, is_atomic):
	features = get_MLP_features_in_batch(op, tile_sizes, tile_pos_is)
	features = torch.as_tensor(features, dtype=torch.float)
	with torch.no_grad():
		if is_atomic:
			features = (features - means_atomic) / std_atomic
			# features = features.to(0)
			# ret = MLP_atomic(features).cpu().numpy()/100
			ret = MLP_atomic(features).numpy()/100
			return ret.flatten()
		else:
			features = (features - means_nonatomic) / std_nonatomic
			# features = features.to(0)
			# ret = MLP_nonatomic(features).cpu().numpy()/100
			ret = MLP_nonatomic(features).numpy()/100
			return ret.flatten()





def cost_tb_latency_given_tile(tile, dsize, ori_tot_nnz, SM_num, is_atomic, penalty):
	'''
	Estimate the tile costs based on the roofline model.
	'''

	# print(tile.op.op_id, tile.tile_sizes, tile.tile_pos)

	try:
		if tile.nnz == 0:
			# 我们不考虑过“空”的tile
			return float('inf')	

		if (tile.op.op_type == 'spmm') and (get_template_str(tile.op) != 'TensorCore_template'):
			# 以下的检查是用来控制ELL tile的workload的，（CSR tile目前处于舍弃状态）

			# FOR DEBUG 此处是用于控制tile的workload  暂时删掉这个条件
			# if (ori_tot_nnz / (tile.nnz) < SM_num): # or (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]!=512*8):
			# # if (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]>512) or (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]<256):
			# 	# 我们不考虑过大的tile
			# 	print("condition 1 fail.")
			# 	return float('inf')

			# FOR DEBUG 直接用计算量来控制workload在合理范围
			if (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]<32*8) or (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]>32*8):
				# print("condition 2 fail.")
				return float('inf') # 每个tile 的计算量的范围，一定程度体现shared memory，register的范围。
			if (tile.tile_sizes[0][1]*tile.tile_sizes[1][1]<256): # 要能允许256个线程
				# print("condition 3 fail.")
				return float('inf')

			# FOR DEBUG 假设我们在这里再考虑一下每个tile的register的限制，也就是给定了一个tile的thread的数量，这个tile的size为多大才是合适的
			# 这个值我们能算出来吗？ 另外，这个地方需要知道我们推测出来的最优tile size是多少，
			# if tile.tile_sizes[0][1]*tile.tile_sizes[1][1]/256 * (2*tile.tile_sizes[2][1]+1)**2 > 96**2:
			# 	return float('inf')
			# best_tile_sizes = fast_tile_tuner(tile, dsize, 256)[0]
			# if best_tile_sizes[0][1]*best_tile_sizes[2][1] + best_tile_sizes[1][1]*best_tile_sizes[2][1]+best_tile_sizes[0][1]*best_tile_sizes[1][1]> 512*3: #96*2:
			# 	return float('inf')

		if get_template_str(tile.op) in ['TensorCore_template', 'TC_sddmm']:
			return get_benchmark_cost(tile, dsize, is_atomic, penalty)
		# elif get_template_str(tile.op) == 'sparse_template_ell':
		# 	return get_benchmark_cost(tile, dsize, is_atomic)


		# <jingzhi>@revision: Allow other cost model
		if 'other_cost_model' in os.environ:
			if os.environ['other_cost_model'] == 'MLP':
				# use MLP as the cost model
				return get_MLP_pred(tile, is_atomic)


		tp, tp_atomic = cost_tb_given_tile(tile, dsize)
		if 0 in [tp, tp_atomic]: # tp == 0:
			# this tile's nnz is 0, we will not select this tile
			# print("zero throughput.")
			return float('inf')

		# cost = (compute_cost_tb_impl_given_tile(tile) / tp * 2.95507917e-05 - 1.97445294e-01)/100  #我们使用benchmark的数据拟合了一个线性函数
		
		# print("in cost model: ", compute_cost_tb_impl_given_tile(tile), tp, tp_atomic)

		# cost =  ELL_linear_cost_model(compute_cost_tb_impl_given_tile(tile) / tp)
		# return cost

		comp_cost = compute_cost_tb_impl_given_tile(tile)
		return ELL_cost_model(comp_cost/tp, comp_cost/tp_atomic, is_atomic, dsize)

	except Exception as e:
		print(tile.get_key())
		assert False






# 这个函数批量计算给定的tiles的pred cost
def cost_tb_latency_given_ELL_tiles(op, tile_sizes, tile_pos_is, dsize, ori_tot_nnz, SM_num, is_atomic):
	'''
	Estimate the tile costs based on the roofline model.
	只针对ELL tile来批量计算cost。
	'''

	# print(tile.op.op_id, tile.tile_sizes, tile.tile_pos)

	try:
		# <jingzhi>@revision:
		if 'other_cost_model' in os.environ:
			if os.environ['other_cost_model'] == 'MLP':
				return get_MLP_pred_no_tile_given(op, tile_sizes, tile_pos_is, is_atomic)

		template_str = "sparse_template_ell"
		compute_cost = compute_cost_tb_impl_given_tile_sizes(template_str, tile_sizes)

		# print("tile_pos_is 11111: ", tile_pos_is)

		memory_costs = memory_cost_tb_given_tile_sizes_and_poses(tile_sizes, tile_pos_is, template_str, dsize, op)

		if not is_atomic:
			memory_costs = (memory_costs[0], np.array([1]))

		if dsize == 32:
			tp_non_atomic, tp_atomic = [np.minimum(compute_cost / (memory_cost * 32), 19.49/1.555) for memory_cost in memory_costs]
		elif dsize == 16:
			tp_non_atomic, tp_atomic = [np.minimum(compute_cost / (memory_cost * 32), 78/1.555) for memory_cost in memory_costs]


		if is_atomic:
			return ELL_cost_model(compute_cost/tp_non_atomic, compute_cost/tp_atomic, is_atomic, dsize)
		else:
			return ELL_cost_model(compute_cost/tp_non_atomic, None, is_atomic, dsize)

		# ====================================
	except Exception as e:
		print(e)
		assert False




def _cost_tb_latency_1D_tiles_given_features(row_nums, col_nums, K, tile_sizes, dsize):
	compute_cost = K * tile_sizes[0] * 2
	# 除了input的memory movement，还有output的memory movement，按照output是连续的来估计cost。
	# memory_costs = ((op.row_nums_1d + op.col_nums_1d) * K + max_bucket_size) * dsize / 8
	# 但是因为我们的cost model 系数还没有更新，所以这里还是改成原来的写法
	# memory_costs = ((row_nums + col_nums) * K) * dsize / 8
	memory_costs = ((row_nums + col_nums) * K + tile_sizes[0]) * dsize / 8
	
	throughputs = None
	if dsize == 32:
		throughputs = np.minimum(compute_cost / memory_costs, 19.49/1.555)
	elif dsize == 16:
		throughputs = np.minimum(compute_cost / memory_costs, 78/1.555)


	costs = compute_cost/throughputs

	# NOTE  TODO！！！！！！
	a, b = None, None
	if dsize == 16:
		# a, b = [ 8.68478513e-08, -5.00523362e-07]
		# a, b = [ 8.74624689e-08, -1.16813420e-05] # 在把output cost也考虑到memory cost之后更新了的参数
		a, b = [5.40647867e-08, 2.14658164e-04] # 把1D tile进行full tune之后（thread可能>32)得到的数据
	elif dsize == 32:
		# a, b = [5.90763791e-08, 1.91908603e-04]
		a, b = [5.90185816e-08, 1.83606878e-04] # 在把output cost也考虑到memory cost之后更新了的参数


	costs = a*costs+b
	return costs





# 这个函数批量计算给定的tiles的pred cost
def cost_tb_latency_given_1D_tiles(op, tile_sizes, dsize):
	'''
	Estimate the tile costs based on the roofline model.
	只针对1D tile来批量计算cost。计算所有可能得1D tile的cost。
	'''

	# 这个地方是不是要存一些meta data会更方便一点?先不管这个

	max_bucket_size = tile_sizes[0]
	pad_num = math.ceil(op.position_space[0].nnz/max_bucket_size)*max_bucket_size - op.position_space[0].nnz
	# position_space[0]一开始就是indices sorted的，而我们不会更新这个变量
	csr = op.position_space[0]
	
	# 首先需要求一些metadata
	if len(op.row_nums_1d) == 0:
		# 我们会将每个1D tile的row number存在row_nums_1d里，因为这个值是不会随着position space的更新而改变的。
		rows = np.repeat(op.idx_values_list[0][0], np.diff(csr.indptr))
		rows = np.concatenate([rows, np.full(pad_num, rows[-1])]).reshape((-1, max_bucket_size))
		op.row_nums_1d = np.asarray([len(np.unique(vs)) for vs in rows])

	if len(op.col_nums_1d) == 0:
		cols = csr.indices
		cols = np.concatenate([cols, np.full(pad_num, cols[-1])]).reshape((-1, max_bucket_size))
		op.col_nums_1d = np.asarray([len(np.unique(vs)) for vs in cols])

	K = op.idx_lens[1]

	return _cost_tb_latency_1D_tiles_given_features(op.row_nums_1d, op.col_nums_1d, K, tile_sizes, dsize)
	# ==============================================================================================================






'''
cost 的计算公式为：
csr：max{ [ #nnz + (J/cache_line_size)*|set(k)| ] * 32, #nnz*J*2/PEAK }
ell：max{ ( IK/cache_line_size + (J/cache_line_size)*|set(k)| ) * 32, IKJ*2/PEAK }

考虑store cost之后，应为
ell：max{ ( IK/cache_line_size + (J/cache_line_size)*|set(k)| + IJ/cache_line_size ) * 32, IKJ*2/PEAK }
考虑了linear cost model之后，应为
ell：max{ ( IK/cache_line_size + (J/cache_line_size)*|set(k)| + IJ/cache_line_size ) * 32 * a + b, IKJ*2/PEAK * a + b }


avg cost 的计算公式：
csr：max{ [ 1/J + ceil(J/cache_line_size)/J * |set(k)|/nnz ] * 32, 2/PEAK }
但是如果我们假设J是cache_line_size的倍数，那么就会变成
     max{ [ 1/J + 1/cache_line_size * |set(k)|/nnz ] * 32, 2/PEAK }

对于ell，我们也假设J是8的倍数
ell：max{ ( IK/(cache_line_size*J*nnz) + (1/cache_line_size)*|set(k)|/nnz ) * 32, IK*2/nnz/PEAK }
增加store cost之后，应为
max{ ( IK/(cache_line_size*J*nnz) + (1/cache_line_size)*|set(k)|/nnz + I/cache_line_size/nnz ) * 32, IK*2/nnz/PEAK }
考虑了linear cost model之后，应为
max{ ( IK/(cache_line_size*J*nnz) + (1/cache_line_size)*|set(k)|/nnz + I/cache_line_size/nnz ) * 32*a+b/J*nnz, IK*2/nnz/PEAK*a+b/J*nnz }

cost 的bound的计算公式为：
csr：[ max{ [ 1/J + 1/cache_line_size * |set(k)|/nnz ] * 32, 2/PEAK } ]
 >=	[ max{ [ 1/J + 1/cache_line_size * k_max/(I*k_max) ] * 32, 2/PEAK } ]
 >=	[ max{ [ 1/J + 1/cache_line_size/I ] * 32, 2/PEAK } ]

[ max{ [ 1/J + 1/8 * |set(k)|/nnz ] * 32, 2/PEAK } ]
<= [ max{ [ 1/J + 1/8 * K/(I*k_min) ] * 32, 2/PEAK } ]

ell: max{ ( IK/(cache_line_size*J*nnz) + (1/cache_line_size)*|set(k)|/nnz ) * 32, IK*2/nnz/PEAK }
>= max{ ( IK/(cache_line_size*J*I*k_max) + (1/cache_line_size)*k_max/(I*k_max) ) * 32, IK*2/(I*k_max)/PEAK }
>= max{ ( K/(cache_line_size*J*k_max) + (1/cache_line_size)/I ) * 32, K*2/k_max/PEAK }
增加了store cost后：
>= max{ ( K/(cache_line_size*J*k_max) + (1/cache_line_size)/I + 1/cache_line_size/k_max ) * 32, K*2/k_max/PEAK }
考虑了linear cost model之后，应为
>= max{ ( K/(cache_line_size*J*k_max) + (1/cache_line_size)/I + 1/cache_line_size/k_max ) * 32*a+b/J*k_max, K*2/k_max/PEAK*a+b/J*k_max }
因为此处b是负数，如果b为正数，则为b/IJ/k_max

max{ ( IK/(8*J*nnz) + (1/8)*|set(k)|/nnz ) * 32, IK*2/nnz/PEAK }
<= max{ ( IK/(8*J*I*k_min) + (1/8)*K/(I*k_min) ) * 32, IK*2/(I*k_min)/PEAK }
<= max{ ( K/(8*J*k_min) + (1/8)*K/(I*k_min) ) * 32, K*2/k_min/PEAK }
增加了store cost后：
<= max{ ( K/(8*J*k_min) + (1/8)*K/(I*k_min) + 1/cache_line_size/k_min ) * 32, K*2/k_min/PEAK }
'''

