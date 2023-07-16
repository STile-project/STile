# This file estimate the cost of a computation tile (corresponds to a thread block)
# import gen_formats
# from importlib import reload
# gen_formats = reload(gen_formats)
 
from gen_formats_v2 import Op, SparseTensor, get_nnz_from_dict, get_val_2_ind, gen_position_space_for_area, get_template_str, gen_tile_sizes_given_tile_tb, get_tile_sizes_idx, get_tile_sizes_list
import math
import numpy as np

# cost_dict = dict()
# from log_hub import data_128_128_128p_1218_2 # the file name should be changed

# now the value of cost_dict has been set


def compute_cost_tb_impl_old(op, tile_sizes, tile_pos, params):
	comp_cost = None
	area_i = op.this_area_i
	idx_values = op.idx_values_list[area_i]
	gen_position_space_for_area(op, area_i)
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
			for i in range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))):
				ori_row_i = idx_values[0][i]
				comp_cost = comp_cost + j_len * 2 * len(list(get_nnz_from_dict(op.position_space[area_i], [ori_row_i]))[ tile_rngs[2][0] : tile_rngs[2][1] ])  # op.inps[0].nnz_num((i,))
		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			comp_cost = math.prod([math.prod(tile_sizes[i][1:]) for i in op.idxs]) * 2
		elif template_str == "TensorCore_template":
			print("This recored is about tensor cores, we do not need to estimate cost.")
		return comp_cost


def compute_cost_tb_impl(op, tile_sizes, tile_pos, params):

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





# def memory_cost_tb_given_tile(tile, dsize): 
# 	'''
# 	We omit the concrete tiling sizes in each kind of template here.
# 	tile_sizes: only tells the workload of a thread block.
# 	INPUT:	dsize: the number of bits of a data. The default is 32 bit (float32).
# 	'''
# 	memory_cost = None
# 	op = tile.op
# 	tile_pos = tile.tile_pos
# 	tile_sizes = tile.tile_sizes



def memory_cost_tb_impl(tile, tile_sizes, dsize):
	'''
	Estimate the memory transaction amount of a computation tile which we know the concrete implementation.
	tile_sizes: the concrete tiling sizes.
	# params: the concrete parameters.
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
			cacheline_n = 0
			
			for i1 in range(tile_sizes[0][1]): # the number of i values for each thread
				for k_i in range(tile_rngs[2][0], tile_rngs[2][1]): 
					# k = idx_values[2][k_i]
					# 
					for warp_i in range(warp_num):
						start, end = warp_i * warp_num, (warp_i+1)*warp_num - 1
						start_idx, end_idx = get_index(start, thread_shape), get_index(end, thread_shape)
						positions = list()
						for i2 in range(start_idx[0], end_idx[0]+1):
							new_i = i1*tile_sizes[0][2] + i2 + tile_rngs[0][0]
							if new_i >= len(idx_values[0]):
								continue
							if k_i >= tile.op.position_space_update.indptr[new_i+1]-tile.op.position_space_update.indptr[new_i]:
								positions.append((tile.op.position_space_update.indptr[new_i] + k_i) // cache_line_size)
						# 
						memory_cost = memory_cost + len(positions)


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



								cacheline_n = cacheline_n + math.ceil((j2_end - j2_start) / cache_line_size)

			memory_cost = memory_cost + cacheline_n


		elif template_str == "sparse_template_ell":

			memory_cost = memory_cost + math.ceil( (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) / cache_line_size )


			cacheline_n = 0
			warp_i_size = None
			if thread_shape[1] > warp_size:

				cacheline_n = tile_sizes[0][1] * (tile_rngs[2][1] - tile_rngs[2][0]) * tile_sizes[1][1] * \
					warp_num * math.ceil(warp_size / cache_line_size)
				warp_i_size = 1
			else:

				nnzs = tile.op.position_space_update.getnnz(axis=1)[ tile.tile_i_rng[0] : tile.tile_i_rng[1] + 1 ]
				pad_i = math.prod(tile_sizes[0][1:]) - len(nnzs)
				nnzs = np.concatenate([nnzs, [0 for tmp in range(pad_i)]])
				warp_i_size = warp_size // thread_shape[1]
				nnzs_list = np.split(nnzs, math.prod(tile_sizes[0][1:])//warp_i_size )

				k_num = sum(nnzs) + sum([ math.prod(tile_sizes[2][1:]) - min(tmp) for tmp in nnzs_list])
				cacheline_n = k_num * math.ceil(thread_shape[1] / cache_line_size) * tile_sizes[1][1]

			memory_cost = memory_cost + cacheline_n
			return memory_cost, -warp_i_size 


		elif template_str == "TensorCore_template":
			assert False, "This recored is about tensor cores, we do not need to estimate cost."
		return memory_cost






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

			k_is = list()
			cacheline_n = 0
			for i in range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))):
				# print("row_i", i)

				indices = get_nnz_from_dict(op.position_space[area_i], [i])[ tile_rngs[2][0] : tile_rngs[2][1] ]
				indices = idx_values[2][indices]

				cacheline_n = cacheline_n + len(indices)  # op.inps[0].nnz_num((i,))			
				k_is = k_is + list(indices)



			k_is = set(k_is)
			j_len = min(tile_rngs[1][1], len(idx_values[1])) - tile_rngs[1][0]
			cacheline_n = cacheline_n + math.ceil(j_len / cache_line_size) * len(k_is)
			memory_cost = memory_cost + cacheline_n

		elif template_str == "sparse_template_ell":

			# in1------------------
			memory_cost = memory_cost + math.ceil( (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) / cache_line_size )
			# in2------------------
			k_is = list()
			for i in range(tile_rngs[0][0], tile_rngs[0][1]):
				if i >= len(op.hyb_new_rows[0]):
					k_is.append(0)
				else:
					# =====================

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
	bank_width = 32 
	template_str = get_template_str(op)
	

	if op.op_type == 'spmm':  

		memory_cost = 0
		tile_rngs = [(tile_pos[i]*math.prod(tile_sizes[i][1:]), (tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))
					for i in op.idxs]

		if template_str == "sparse_template":

			cacheline_n = tile.nnz
			# k_is = idx_values[2][np.array(set(tile.uncovered_position_space['in1'].indices))]


			indptr = tile.op.position_space[0].indptr
			inds = indptr[ tile_rngs[0][0] ], \
				indptr[ min(tile_rngs[0][1], len(indptr)-1) ]
			k_is = set(tile.op.position_space[0].indices[ inds[0]:inds[1] ])


			j_len = tile.j_num

			cacheline_n = cacheline_n + math.ceil(j_len / cache_line_size) * len(k_is)
			memory_cost = memory_cost + cacheline_n

			memory_cost = [memory_cost, memory_cost]

		elif template_str == "sparse_template_ell":

			# in1------------------
			memory_cost = memory_cost + math.ceil( (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) / cache_line_size )
			
			# check the shared memory constraint
			if (tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0]) * dsize > 49152 * 8:
				return float('inf')



			indptr = tile.op.position_space[0].indptr
			inds = indptr[ tile_rngs[0][0] ], \
				indptr[ min(tile_rngs[0][1], len(indptr)-1) ]
			k_is = set(tile.op.position_space[0].indices[ inds[0]:inds[1] ])
			# k_is = idx_values[2][np.array(k_is)]
			if tile.nnz < ((tile_rngs[0][1] - tile_rngs[0][0]) * (tile_rngs[2][1] - tile_rngs[2][0])):
				k_is.add(0)


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

			# in1
			memory_cost = math.prod(tile_sizes[0][1:]) * (math.prod(tile_sizes[2][1:]) / cache_line_size)
			# in2
			memory_cost = memory_cost + math.prod(tile_sizes[1][1:]) * math.prod(tile_sizes[2][1:]) / cache_line_size

			memory_cost = [memory_cost, memory_cost]

		return memory_cost






def memory_cost_tb_given_tile_sizes_and_poses(tile_sizes, tile_pos_is, template_str, dsize, op): 

	cache_line_size = 32*8/dsize #  8 # i.e., 32 byte = 8 float32
	# bank_width = 32 

	memory_cost = 0
	if template_str == "sparse_template_ell":
		# in1------------------
		memory_cost_A = math.ceil( tile_sizes[0][1] * tile_sizes[2][1] / cache_line_size )
		



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


		cacheline_bit = 32*8
		warp_size = 32
		store_cost = math.ceil(tile_sizes[1][1] * dsize / cacheline_bit) * tile_sizes[0][1]

		memory_cost = [memory_cost_A + store_cost + memory_cost_B, memory_cost_A + store_cost * 2 + memory_cost_B]

	else:
		assert False, "We only support ELL tiles now."

	return memory_cost





def cost_tb_impl(op, tile_sizes, tile_pos, params):
	return compute_cost_tb_impl(op, tile_sizes, tile_pos, params) / memory_cost_tb_impl(op, tile_sizes, tile_pos, params)


def cost_tb(op, tile_sizes, tile_pos, params):
	compute_cost = compute_cost_tb(op, tile_sizes, tile_pos, params) 
	memory_cost = memory_cost_tb(op, tile_sizes, tile_pos, params)
	# print(compute_cost, memory_cost)

	if memory_cost == 0:
		# print(op.op_id, tile_sizes, tile_pos, params)
		assert compute_cost == 0
		return 0  
	return min(compute_cost / (memory_cost * 32), 19.49/1.555)



def cost_tb_given_tile(tile, dsize):

	compute_cost = compute_cost_tb_impl_given_tile(tile) 
	# memory_cost = memory_cost_tb_given_tile(tile, dsize)
	memory_costs = memory_cost_tb_given_tile(tile, dsize)
	# print(compute_cost, memory_costs)


	# if memory_cost == 0:
	if 0 in [memory_costs]:
		# print(op.op_id, tile_sizes, tile_pos, params)
		assert compute_cost == 0
		return 0, 0  
	if get_template_str(tile.op) == 'TensorCore_template':


		return [min(compute_cost / (memory_cost * 32) * 10, 312/1.555) for memory_cost in memory_costs] #624/1.555)  312/1.555) #

		# ====================================
	else:
		if dsize == 32:
			return [min(compute_cost / (memory_cost * 32), 19.49/1.555) for memory_cost in memory_costs]
		elif dsize == 16:
			return [min(compute_cost / (memory_cost * 32), 78/1.555) for memory_cost in memory_costs]



def cost_tb_latency(op, tile_sizes, tile_pos, params = None):
	'''
	Estimate the tile costs based on the roofline model.
	'''
	try:
		tp = cost_tb(op, tile_sizes, tile_pos, params)
		if tp == 0:
			# this tile's nnz is 0, we will not select this tile
			return float('inf')

		cost = compute_cost_tb(op, tile_sizes, tile_pos, params) / tp
		return cost
	except Exception as e:
		print(op.op_id, tile_sizes, tile_pos)
		assert False



def fast_tile_tuner(tile, dsize, thread_num, max_bucket_size):
    sub_op = tile.op
    tile_sizes = tile.tile_sizes
    params = tile.params

    all_tile_sizes = [tile_sizes]
    is_tb_tile = False
    if max([len(tile_size) for tile_size in tile_sizes]) == 2:
        all_tile_sizes = gen_tile_sizes_given_tile_tb(tile, max_bucket_size)
        is_tb_tile = True

    mem_cost_dict = dict()
    for tile_sizes in all_tile_sizes:
    	# print(tile_sizes)
        thread_num_tmp = tile_sizes[0][2]*tile_sizes[1][2]
        if thread_num_tmp != thread_num:
        	continue
        if thread_num not in mem_cost_dict:
            mem_cost_dict[thread_num] = list()
        mem_cost = memory_cost_tb_impl(tile, tile_sizes, dsize)
        mem_cost_dict[thread_num].append((tile_sizes, mem_cost))

    ret = list()
    for thread_num in sorted(mem_cost_dict.keys(), reverse=True):
        costs = mem_cost_dict[thread_num]
        costs = sorted(costs, key=lambda tmp:tmp[1])
        ret.append(costs[0][0])

    return ret




def TC_cost_in_a_row(tile, dsize, max_bucket_size):

	TC_cost_dict = {
		(16, 32, 16): 0.03913872/100, 
		(16, 32, 32): 0.05328592/100,
		(16, 32, 48): 0.06522246/100,
		(16, 32, 64): 0.07834872/100,
		(16, 32, 80): 0.10675935/100,
		(16, 32, 96): 0.10595788/100,
		(16, 32, 112): 0.11501513/100,
		(16, 32, 128): 0.13/100, 
		(16, 32, 144): 0.14248087/100,
		}
	TC_cost_dict_atomic = {
		(16, 32, 16): 0.13574204/100, 
		(16, 32, 32): 0.14316941/100,
		(16, 32, 48): 0.15268948/100,
		(16, 32, 64): 0.16785308/100,
		(16, 32, 80): 0.18458567/100,
		(16, 32, 96): 0.20565027/100,
		(16, 32, 112): 0.21700917/100,
		(16, 32, 128): 0.24/100, 
		(16, 32, 144): 0.2612427/100,
		}

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
	op = tile.op
	tile_sizes = tile.tile_sizes
	I, J, K = tile_sizes[0][1], tile_sizes[1][1], tile_sizes[2][1]
	tile_pos = tile.tile_pos
	num = math.ceil(len(op.k_vals[tile_pos[0]]) / K)

	csr = op.position_space_update[tile.tile_i_rng[0]:tile.tile_i_rng[1]+1]
	nnzs = csr[:, op.k_vals[tile_pos[0]]].getnnz(axis=0)
	zero_col = np.searchsorted(-(nnzs>0), 0)
	non_zero_tile_pos_k = (zero_col-1) // K

	best_cost = float('inf')
	best_i = None
	for i in range(num+1):

		cost = 0
		if i-1 >= non_zero_tile_pos_k:

			cost = TC_cost_dict[(I, J, K)] * (non_zero_tile_pos_k+1) # 总体cost
		elif i == 0:

			remain_csr = csr
			for row_i in range(remain_csr.shape[0]):
				row = remain_csr[row_i]
				nnz = row.nnz
				ELL_K = 2**(math.ceil(math.log(nnz, 2)))

				if ELL_K > max_bucket_size:

					ELL_K = max_bucket_size

					cost = cost + ELL_cost_dict_atomic[(256//ELL_K, 32, ELL_K)] / (256//ELL_K) * math.ceil(nnz/ELL_K)
				else:

					cost = cost + ELL_cost_dict[(256//ELL_K, 32, ELL_K)] / (256//ELL_K)
		else:
			cost = TC_cost_dict_atomic[(I, J, K)] * i

			remain_csr = csr[:, K * i:]
			for row_i in range(remain_csr.shape[0]):
				row = remain_csr[row_i]
				nnz = row.nnz
				ELL_K = 2**(math.ceil(math.log(nnz, 2)))

				if ELL_K > max_bucket_size:

					ELL_K = max_bucket_size

					cost = cost + ELL_cost_dict_atomic[(256//ELL_K, 32, ELL_K)] / (256//ELL_K) * math.ceil(nnz/ELL_K)
				else:

					cost = cost + ELL_cost_dict_atomic[(256//ELL_K, 32, ELL_K)] / (256//ELL_K)
		# 
		if cost < best_cost:
			best_cost = cost
			best_i = i
	return best_cost, best_i






def get_benchmark_cost_(template_str, tile_sizes, dsize, is_atomic, penalty):
	'''
	penalty: the performance degradation ratio computed by TC tile occupancy.
	'''
	if template_str == 'TensorCore_template':
		if dsize == 16:  

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
				return TC_cost_dict_atomic[key] / penalty 
			else:
				return TC_cost_dict[key] / penalty 
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




def ELL_non_atomic_coeff(dsize):
	if dsize == 16:
		# return [9.74636589e-08, 7.22826260e-04] 
		return [8.80425616e-08, 4.17697078e-04] 
	else:
		# dsize == 32
		return [8.23071690e-08, 3.42962663e-03]

def ELL_atomic_extra_coeff(dsize):
	if dsize == 16:
		# return [4.62653486, -3.74684073]
		return [ 8.80688635, -7.9094305 ]
	else:
		# dsize == 32
		return [0.67944825, 0.29144282]




def ELL_cost_model(x_nonatomic, x_atomic, is_atomic, dsize):

	a1, b1 = ELL_non_atomic_coeff(dsize)
	a2, b2 = ELL_atomic_extra_coeff(dsize)
	if is_atomic:
		return (a1 * x_nonatomic + b1) * (a2 * x_atomic/x_nonatomic + b2)
	else:
		return (a1 * x_nonatomic + b1)




def lower_bound_ELL_cost_BSPMM(sub_op, max_bucket_size, dsize):

	is_tb_tile = True
	tile_sizes_idx = get_tile_sizes_idx(sub_op, max_bucket_size, is_tb_tile=is_tb_tile)
	if sub_op.op_type == 'spmm':

		tile_sizes_idx[1] = [(None, 32)]
		# ===============================
	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile=is_tb_tile)
	cacheline_bit = 32*8
	pred_avg_costs = list()
	for tile_sizes in tile_sizes_list:
		I, J, K = [i[1] for i in tile_sizes]
		# 

		# if I*K != 256:
		# 	continue
		# if I*J < 256:
		# 	continue
		# 
		compute_cost = math.prod([I, J, K]) * 2

		memory_cost = math.ceil(I*K*dsize/cacheline_bit) + math.ceil(J*dsize/cacheline_bit)*(K//2+1) + I*math.ceil(J*dsize/cacheline_bit)

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

	row_nums = np.concatenate([range(1, nnz+1) for nnz in range(1, max_bucket_size+1)])
	nnzs = np.repeat(np.arange(1, max_bucket_size+1), np.arange(1, max_bucket_size+1))
	col_nums = nnzs/row_nums
	avg_costs = _cost_tb_latency_1D_tiles_given_features(row_nums, col_nums, sub_op.idx_lens[1], (max_bucket_size, ), dsize) / nnzs / sub_op.idx_lens[1]
	# print(list(avg_costs))
	assert min(avg_costs) > 0, "Negative lower bound."
	return min(avg_costs)





def lower_bound_ELL_cost(sub_op, max_bucket_size, dsize):
	if sub_op.op_type == 'spmm':
		return lower_bound_ELL_cost_BSPMM(sub_op, max_bucket_size, dsize)
	elif sub_op.op_type == 'sddmm':
		return lower_bound_1D_cost_BSDDMM(sub_op, max_bucket_size, dsize)








def cost_tb_latency_given_tile(tile, dsize, ori_tot_nnz, SM_num, is_atomic, penalty):
	'''
	Estimate the tile costs based on the roofline model.
	'''

	# print(tile.op.op_id, tile.tile_sizes, tile.tile_pos)

	try:
		if tile.nnz == 0:

			return float('inf')	

		if (tile.op.op_type == 'spmm') and (get_template_str(tile.op) != 'TensorCore_template'):



			if (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]<32*8) or (tile.tile_sizes[0][1]*tile.tile_sizes[2][1]>32*8):
				# print("condition 2 fail.")
				return float('inf') 
			if (tile.tile_sizes[0][1]*tile.tile_sizes[1][1]<256): 
				# print("condition 3 fail.")
				return float('inf')



		if get_template_str(tile.op) in ['TensorCore_template', 'TC_sddmm']:
			return get_benchmark_cost(tile, dsize, is_atomic, penalty)
		# elif get_template_str(tile.op) == 'sparse_template_ell':
		# 	return get_benchmark_cost(tile, dsize, is_atomic)

		tp, tp_atomic = cost_tb_given_tile(tile, dsize)
		if 0 in [tp, tp_atomic]: # tp == 0:
			# this tile's nnz is 0, we will not select this tile
			# print("zero throughput.")
			return float('inf')

		# cost = (compute_cost_tb_impl_given_tile(tile) / tp * 2.95507917e-05 - 1.97445294e-01)/100 
		
		# print("in cost model: ", compute_cost_tb_impl_given_tile(tile), tp, tp_atomic)

		# cost =  ELL_linear_cost_model(compute_cost_tb_impl_given_tile(tile) / tp)
		# return cost

		comp_cost = compute_cost_tb_impl_given_tile(tile)
		return ELL_cost_model(comp_cost/tp, comp_cost/tp_atomic, is_atomic, dsize)

	except Exception as e:
		print(tile.get_key())
		assert False







def cost_tb_latency_given_ELL_tiles(op, tile_sizes, tile_pos_is, dsize, ori_tot_nnz, SM_num, is_atomic):


	# print(tile.op.op_id, tile.tile_sizes, tile.tile_pos)

	try:
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

	memory_costs = ((row_nums + col_nums) * K + tile_sizes[0]) * dsize / 8
	
	throughputs = None
	if dsize == 32:
		throughputs = np.minimum(compute_cost / memory_costs, 19.49/1.555)
	elif dsize == 16:
		throughputs = np.minimum(compute_cost / memory_costs, 78/1.555)


	costs = compute_cost/throughputs

	a, b = None, None
	if dsize == 16:
		# a, b = [ 8.68478513e-08, -5.00523362e-07]
		# a, b = [ 8.74624689e-08, -1.16813420e-05] 
		a, b = [5.40647867e-08, 2.14658164e-04] # 
	elif dsize == 32:
		# a, b = [5.90763791e-08, 1.91908603e-04]
		a, b = [5.90185816e-08, 1.83606878e-04] #


	costs = a*costs+b
	return costs






def cost_tb_latency_given_1D_tiles(op, tile_sizes, dsize):


	max_bucket_size = tile_sizes[0]
	pad_num = math.ceil(op.position_space[0].nnz/max_bucket_size)*max_bucket_size - op.position_space[0].nnz

	csr = op.position_space[0]
	

	if len(op.row_nums_1d) == 0:

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


	compute_cost = K * tile_sizes[0] * 2

	memory_costs = ((op.row_nums_1d + op.col_nums_1d) * K) * dsize / 8
	
	throughputs = None
	if dsize == 32:
		throughputs = np.minimum(compute_cost / memory_costs, 19.49/1.555)
	elif dsize == 16:
		throughputs = np.minimum(compute_cost / memory_costs, 78/1.555)


	costs = compute_cost/throughputs


	a, b = None, None
	if dsize == 16:
		a, b = [ 8.68478513e-08, -5.00523362e-07]
	elif dsize == 32:
		a, b = [5.90763791e-08, 1.91908603e-04]


	costs = a*costs+b
	return costs







def get_k_max(op, tile_sizes, tile_pos):

	if op.op_type == 'spmm':
		# we have reordered index i so that rows with larger k are in the upper part of the matrix
		tile_rngs = [(tile_pos[i]*math.prod(tile_sizes[i][1:]), (tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))
					for i in op.idxs]
		template_str = get_template_str(op)
		if template_str == "sparse_template":
			i = tile_rngs[0][0]
			return min(op.inp_getnnz[0][i], tile_rngs[2][1]) - min(op.inp_getnnz[0][i], tile_rngs[2][0])
		elif template_str == 'sparse_template_ell':			
			k_lens = list()
			for i in [tile_rngs[0][0], tile_rngs[0][0]+1]:
				if i >= len(op.hyb_new_rows[0]):
					k_lens.append(0)
				else:
					k_lens.append(op.inp_getnnz[0][i])
			return max(k_lens)



 

def cost_lower_bound_tb(op, tile_sizes, tile_pos, dsize):

	if op.op_type == 'spmm':
		cache_line_size = 32*8/dsize #  8 # i.e., 32 byte = 8 float32

		template_str = get_template_str(op)
		I, J, K = tile_sizes[0][1], tile_sizes[1][1], tile_sizes[2][1]
		if dsize == 32:
			PEAK = 19.49/1.555
		elif dsize == 16:
			PEAK = 78/1.555
		if template_str == "sparse_template":
			# max{ [ 1/J + 1/cache_line_size/I ] * 32, 2/PEAK }
			return max((1/J + 1/cache_line_size/I) * 32, 2/PEAK)
		elif template_str == 'sparse_template_ell':
			k_max = get_k_max(op, tile_sizes, tile_pos)
			# max{ ( K/(cache_line_size*J*k_max) + (1/cache_line_size)/I ) * 32, K*2/k_max/PEAK }

			if k_max == 0:
				# this tile's nnz is 0, we will not select this tile
				return float('inf')

			# a, b = ELL_linear_cost_model_coeff()
			a, b = ELL_non_atomic_coeff()

			return max(( K/(cache_line_size*J*k_max) + (1/cache_line_size)/I + 1/cache_line_size/k_max ) * 32 *a+b/(I*J*k_max), 
				K*2/k_max/PEAK *a+b/(I*J*k_max))
		# elif template_str == 'TensorCore_template':
		# 	return I*K*dsize/cache_line_size + k*J*dsize/cache_line_size
		else:
			assert False, "We only support lower bound computation for csr/ell-like spmm now."

  