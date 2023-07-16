

import math
import itertools
import copy
import multiprocessing
from multiprocessing import shared_memory
import dgl
import scipy
import torch
import json
# import ray

# ray.init(ignore_reinit_error=True)

# from tc_spmm import *
import tvm
# from tvm.sparse.format import condense
import tvm.testing
import tvm.tir as tir
# import argparse
# import torch as th
import numpy as np
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter
# from my_formats import *
# from tc_spmm import *

# from utils import get_dataset
from typing import Any

import os

from sparsetir_artifact import profile_pytorch_ms

class DenseTensor(object):
	"""docstring for DenseTensor"""
	def __init__(self, data):
		super(DenseTensor, self).__init__()
		# the data should be just numpy array
		self.data = data
		





# the data structure for an operator, similar to that one in eto
class Op(object):
	"""docstring for Op"""
	def __init__(self, op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed):
		super(Op, self).__init__()
		print("haha")
		self.op_type = op_type
		self.sidxs = sidxs
		self.ridxs = ridxs
		self.idxs = sidxs + ridxs
		self.idx_graph = idx_graph # 2D list: the dependence graph of the indices in the iteration space
		self.idx_lens = idx_lens # complete index lens
		# complete inps, each inp is of CSR or Dense formats (no concrete value); consistent with idx_graph
		# each input is of type SparseTensor
		self.inps = inps
		self.sparse_inps_transposed = sparse_inps_transposed # store the sparse inputs which are transposed for faster computation, e.g., csc format in1 for spmm.
		# 
		# self.srngs_list = [srngs] # 2D list: index range in coordinate space, each sublist for a sub-kernel
		# self.rrngs_list = [rrngs]
		self.idx_values_list = [[np.arange(max_v) for max_v in self.idx_lens]] # Initialize the index values for the op.
		# self.complete_idx_values = None
		# self.rid_values_list = [[list(rng) for rng in self.rrngs]]
		# self.inps_list = [inps] # 2D list: each sublist is of CSR or Dense formats (no concrete value) for a sub-kernel
		# 
		self.nnz_dict = dict()
		# ------------the parameters below are set after calling the existing asymptotic cost model-------------------
		self.loop_order = {0:self.idxs} # list of lists, each sub-list for an area, storing the order of the loops
		self.loop_protocals = None # focused on whether the loop is compressed or not
		self.inp_protocals = None # list of lists, each sub-list for an area, storing whether an index is compressed or not
		if op_type == 'spmm':
			self.loop_protocals = {0:'uuc'} # which means only the k index is compressed
			self.inp_protocals = {0:['uc', 'uu']} # each symbol in the str are in the order of the indices
		elif op_type == 'sddmm':
			self.loop_protocals = {0: 'cu'}
			self.inp_protocals = {0:['c', 'uu', 'uu']}
		else:
			assert False, f"We do not support {op_type} currently!"
		self.position_space = dict() # key is the area id, value is the tree structure of the position space
		self.inp_getnnz = None # the result of calling getnnz() on the scipy.sparse.csr_matrix of the sparse inputs. Set together with position_space.
		self.pos_space_infor = dict() # key is the area id, value is the dimension infor of the position space
		self.position_space_update = self.inps[0].copy() # this variable will be updated when we select tiles 
		# ------------
		self.idx_reordered = {0 : [False for idx_i in self.idxs]} # record an index has been reordered or not
		self.this_area_i = 0 # record the area index of this op/sub-op
		# ------------
		self.hyb_new_rows = dict()
		self.hyb_short_begin_row = None
		self.hyb_getnnz_minus = None
		self.hyb_ori_row_ends = None # 
		# ------------
		# self.hyb_row_rng = None # 
		self.ori_row_ids_1d = None # 
		self.ori_col_ids_1d = None # 
		self.row_nums_1d = list() # 
		self.col_nums_1d = list() # 
		# self.row_matrix_1d = None 
		self.nnz_id_matrix = None 
		self.nnz_update_1D = None 
		self.nnz_data_1D = None 
		self.max_bucket_size = None 
		# ------------
		self.k_vals = None # 
		self.blk_nums = dict()
		self.blk_ks = dict() # stores the non-empty block index in each block row for each tile sizes, when we do not reorder k for the TC op.
		self.TC_k_notsorted = False
		# ------------
		self.kernel_tile_Ns = [1 for idx_i in self.idxs]
		self.kernel_tile_times = 0
		self.area_pos = list() # store the area_pos for each kernel level tile
		# ------------
		self.op_id = None # the unique id of an op
	
		# ------------
		if self.op_type == 'sddmm':
			gen_nnz_id_matrix(self)

	def print_infor(self):
		print([(min(d), max(d)) for ds in self.idx_values_list for d in ds], self.loop_order, self.loop_protocals, self.inp_protocals, self.position_space, self.pos_space_infor, self.idx_reordered, self.this_area_i)

	#
	def get_key(self):
		'''
			Return the key of this operator.
			Key is a tuple of (op_type, idx_lens, kernel_tile_Ns, kernel_tile_times, this_area_i, idx_reordered, loop_protocals, loop_order).
		'''
		return (self.op_type, self.idx_lens, self.kernel_tile_Ns, self.kernel_tile_times, self.this_area_i, self.idx_reordered, self.loop_protocals, self.loop_order)






# the rules to generate formats
def kernel_tile_condition(op, tile_Ns):
	'''
	Divide the iteration space into several areas so that we can generate formats for them adaptively
	We only do one level of such kernel tile, i.e., the op must have not been tiled.
	INPUT:
		op:		Op.		The op object.
	OUTPUT:
		True: if op can be tiled on the kernel level; else False.
	'''
	# return len(op.idx_values_list) == 1
	for last_tile_N, this_tile_N in zip(op.kernel_tile_Ns, tile_Ns):
		if (last_tile_N > 1) and (this_tile_N > 1):
			return False
	if op.kernel_tile_times > 0:
		for idx_i, n in enumerate(tile_Ns):
			if (not op.idx_reordered[0][idx_i]) and (n > 1):
				return False
			
			if (op.op_type == 'spmm') and (idx_i == 0) and (n > 1):
				return False
	return True



def kernel_tile(op, tile_Ns):
	'''
	Divide the iteration space into several areas so that we can generate formats for them adaptively
	We only do one level of such kernel tile.
	INPUT:
		op:			Op.		The op object.
		tile_Ns:	list of ints. The kernel level tile numbers.
	OUTPUT:
		# No output, directly change the related parameters of op, i.e., op.idx_values_list.
		return the list of generated sub-operators.
	'''
	if not kernel_tile_condition(op, tile_Ns):
		return list()

	ret = list()
	area_poses = list(itertools.product(*[range(n) for n in tile_Ns]))
	tile_sizes = [math.ceil(len(values) / n) for values, n in zip(op.idx_values_list[0], tile_Ns)]
	# idx_values_list = [[op.idx_values_list[0][i]\
	# 						[area_pos[i]*tile_sizes[i] : (area_pos[i]+1)*tile_sizes[i]] \
	# 								for i in range(len(op.idx_lens))]
	# 								for area_pos in area_poses]
	
	# New Implementation =====================================================================
	idx_values_list = list()
	for idx_values in op.idx_values_list:
		idx_values_list = idx_values_list + [[idx_values[i]\
							[area_pos[i]*tile_sizes[i] : (area_pos[i]+1)*tile_sizes[i]] \
									for i in range(len(op.idx_lens))]
									for area_pos in area_poses]
	

	for idx_values, area_pos in zip(idx_values_list, area_poses):

		new_op = copy.copy(op)
		new_op.idx_values_list = [idx_values]
		new_op.kernel_tile_Ns = [i * j for i, j in zip(op.kernel_tile_Ns, tile_Ns)]
		new_op.kernel_tile_times = op.kernel_tile_times + 1
		new_op.this_area_i = 0
		new_op.area_pos = new_op.area_pos + [area_pos]
		if sum(tile_Ns)!=len(tile_Ns): # tile_Ns = [1,1,1] for spmm
			new_op.nnz_dict = dict()
		ret.append(new_op)

	return ret








def reorder_idx(op, area_i, idx_i):
	'''
	Sort the given index by the nnz (from large to small).
	INPUT:
		op:		Op.		The op object.
		area_i: int.	The id of the area of the complete iteration space.
		idx_i: 	int.	The index id.
	Output:
		no output, directly change the corresponding id values.
	'''
	# compute the NNZs along the input index.
	if op.op_type in ['spmm', 'sddmm'] and idx_i == 1:
		op.idx_reordered[area_i][idx_i] = False
		return
	# if op.op_type == 'sddmm' and idx_i == 2:
	# 	op.idx_reordered[area_i][idx_i] = False
	# 	return
	
	

	if op.op_type in ['spmm', 'sddmm']:
		idx_values = op.idx_values_list[area_i]
		position_space = op.inps[0][idx_values[0],:][:, idx_values[2]]
		nnzs = None
		if idx_i == 0:
			nnzs = position_space.getnnz(axis=1)
		else:
			nnzs = position_space.getnnz(axis=0)

		order = np.argsort( nnzs*(-1), kind='stable' )
		op.idx_values_list[area_i][idx_i] = np.array(idx_values[idx_i])[order]
		op.idx_reordered[area_i][idx_i] = True






def regroup_for_strided_pattern(op, group_size):

	if op.op_type == 'spmm':
		# we only regroup the rows
		tot = (op.idx_lens[0] // group_size) * group_size
		row_is = np.arange(tot)
		row_is = np.reshape(row_is, (-1, group_size))
		row_is = row_is.T
		last_group = None
		if tot == op.idx_lens[0]:
			last_group = [list() for i in range(group_size)]
		else:
			last_group = [[i] for i in range(tot, op.idx_lens[0])] + [list() for i in range(op.idx_lens[0] - tot)]

		new_row_is = np.concatenate([np.concatenate((row, tail)) for row, tail in zip(row_is, last_group)])
		op.idx_values_list[0][0] = new_row_is



def reorder_for_strided_pattern(op):

	if op.op_type == 'spmm':
		# we only regroup the rows

		pos = np.searchsorted(op.inps[0].indptr, op.inps[0].indptr[-1])
		poses = [op.inps[0].indices[i] for i in op.inps[0].indptr[:pos]] + [float('inf') for i in range(op.inps[0].shape[0]-pos)]
		order = np.argsort( poses, kind='stable' )
		op.idx_values_list[0][0] = order
		






def gen_position_space_for_area(op, area_i):
	'''
	Generate the corresponding tree structure for the area_i, according to the setting in op.loop_order, op.loop_protocals.
	It seems that op.loop_protocals will not be used here.
	INPUT:
		area_i:		int.	The id of the area.
	OUTPUT:
		No output, we directly add the position space into op.position_space.
	'''
	if area_i in op.position_space:
		return
	# loop_order = op.loop_order[area_i]
	# loop_protocals = op.loop_protocals[area_i]
	# we need to get all the none-zeros in the iteration space for area i, and build the position tree at the same time
	idx_values = op.idx_values_list[area_i]
	
	# position_space = dict()
	# coordinate_to_position_space(op, idx_values, loop_order, position_space)
	# position_space = coordinate_to_position_space_parallel(op, idx_values, loop_order)
	if op.op_type in ['spmm', 'sddmm']:
		position_space = op.inps[0][idx_values[0],:][:, idx_values[2]]
		op.position_space[area_i] = position_space
		op.inp_getnnz = [position_space.getnnz(axis = 1)]

		op.position_space_update = position_space.copy()








def get_val_2_ind(vals):
	'''
	Generate a mapping from the value to its index in vals.
	INPUT:	
		vals:		list of ints.		The input value list.
	OUTPUT:
		mapping:	dict.				key is a value in vals, key is its index.
	'''
	mapping = dict()
	for count, v in enumerate(vals):
		mapping[v] = count
	return mapping



def get_nnz_from_dict_old(val_dict, prefix_pos):
	'''
	Get the NNZs in the direct childs of position prefix_pos in the tree.
	'''
	for i in prefix_pos:
		if i not in val_dict:
			return list()
		val_dict = val_dict[i]
	if isinstance(val_dict, dict):
		return val_dict.keys()
	else:
		return val_dict



def get_nnz_from_dict(val_dict, prefix_pos):

	if len(prefix_pos) == 1:

		return val_dict.getrow(prefix_pos[0]).indices
	# for i in prefix_pos:
	# 	if i not in val_dict:
	# 		return list()
	# 	val_dict = val_dict[i]
	# if isinstance(val_dict, dict):
	# 	return val_dict.keys()
	# else:
	# 	return val_dict





def SPMM_inner_product(op, area_i):
	'''
	for i, j, k. Set compressed or not here. No loop fusion here. 
	INPUT:
		op:		Op.		The operator to be scheduled.
	OUTPUT:
		No output, directly change op's loop_order, loop_protocals, inp_protocals.
		NOTE:
			values in loop_protocals and inp_protocals are in the default order.
	'''
	op.loop_order[area_i] = [0, 1, 2]
	op.loop_protocals = {area_i : 'uuc'}
	op.inp_protocals = {area_i : ['uc', 'uu']}






def SPMM_Gustavson_product(op, area_i):
	'''
	for i, k, j. Set compressed or not here. No loop fusion here. 
	INPUT:
		op:		Op.		The operator to be scheduled.
	OUTPUT:
		No output, directly change op's loop_order, loop_protocals, inp_protocals.
		NOTE:
			values in loop_protocals and inp_protocals are in the default order.
	'''
	op.loop_order[area_i] = [0, 2, 1]
	op.loop_protocals = {area_i : 'uuc'}
	op.inp_protocals = {area_i : ['uc', 'uu']}




def SDDMM_Gustavson_product(op, area_i):
	op.loop_order[area_i] = None
	op.loop_protocals = {area_i : 'cu'}
	op.inp_protocals = {area_i : ['c', 'uu', 'uu']}


def SPMM_inner_product(op, area_i):
	'''
	for k, i, j. Set compressed or not here. No loop fusion here. 
	INPUT:
		op:		Op.		The operator to be scheduled.
	OUTPUT:
		No output, directly change op's loop_order, loop_protocals, inp_protocals.
		NOTE:
			values in loop_protocals and inp_protocals are in the default order.
	'''
	op.loop_order[area_i] = [2, 0, 1]
	op.loop_protocals[area_i] = 'uuc' # i->c   k->u
	op.inp_protocals[area_i] = ['cu', 'uu']




def condense_for_TC_tile(op, TC_tile_sizes, TC_k_notsorted, reorder_k_by_nnz = True):

	# 
	# store the new row ids given max bucket size
	k_axis = 2
	if op.op_type in ['spmm', 'sddmm']:
		row_num = len(op.idx_values_list[0][0])
		blk_i_size = TC_tile_sizes[0][1]
		row_blk_num = math.ceil(row_num / blk_i_size)

		idx_values = op.idx_values_list[0]
		csr = op.inps[0][idx_values[0],:][:, idx_values[k_axis]]


		if TC_k_notsorted:

			k_vals = np.arange( len(idx_values[k_axis]) )
			op.k_vals = [k_vals for i in range(row_blk_num)]
			return

		if reorder_k_by_nnz:
			k_vals = [ np.argsort( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)*-1, kind='stable' )  \
							for i in range(row_blk_num)]
		else:
			k_vals = [np.argsort( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0)<=0, kind='stable' ) \
							for i in range(row_blk_num)]
		op.k_vals = k_vals




def change_to_dense(op, area_i, TC_tile_sizes, TC_k_notsorted, reorder_k_by_nnz = True):
	'''
	change the operator from sparse to dense. Once all this, the area can be implemented using dense kernels or tensor cores.
	'''
	op.loop_protocals[area_i] = 'u'*len(op.idxs)
	op.inp_protocals[area_i] = ['u'*len(protocal) for protocal in op.inp_protocals[area_i]]
	if op.op_type == 'sddmm':
		op.inp_protocals[area_i] = 'uuu'
	condense_for_TC_tile(op, TC_tile_sizes, TC_k_notsorted, reorder_k_by_nnz = reorder_k_by_nnz)







def pad_irregular_idx_slow(op, area_i, idx_i, max_bucket_size):
	# 
	op.loop_protocals[area_i] = op.loop_protocals[area_i][:idx_i]+'p'+op.loop_protocals[area_i][idx_i+1:]
	# store the new row ids given max bucket size
	if op.op_type == 'spmm':
		op.hyb_new_rows = [list()]
		op.hyb_short_begin_row = None
		op.hyb_getnnz_minus = list()
		op.hyb_ori_row_ends = list()
		gen_position_space_for_area(op, area_i)
		# indptr = [0]
		print("finish process position space")
		for i in range(len(op.idx_values_list[area_i][0])):
			# get the nnz of row i in sparse input 1 
			# indices = get_nnz_from_dict(op.position_space[area_i], [i])
			indices_num = op.inp_getnnz[0][i]
			
			if (op.hyb_short_begin_row == None) and (indices_num <= max_bucket_size):
				op.hyb_short_begin_row = len(op.hyb_new_rows[0])

			# print(i, len(indices))


			# indptr = indptr + \
			# 	list(range(indptr[-1], indptr[-1] + indices_num, max_bucket_size))[1:] + [indptr[-1] + indices_num,]
			
			# op.hyb_ori_row_ends.append(len(indptr) - 2)

			for sub_row_i in range(math.ceil(indices_num/max_bucket_size)):

				op.hyb_new_rows[0].append(i)
				# op.hyb_new_rows[1].append(max_bucket_size*sub_row_i)
				# op.hyb_new_rows[2].append(max_bucket_size*(sub_row_i+1))

			if indices_num == 0:
				op.hyb_new_rows[0].append(i)

			op.hyb_ori_row_ends.append(len(op.hyb_new_rows[0]) - 1)


		indptr = np.empty(len(op.hyb_new_rows[0]) + 1)
		indptr[0] = 0
		start_idx = 0
		for i in range(len(op.idx_values_list[area_i][0])):
			indices_num = op.inp_getnnz[0][i]
			if indices_num == 0:
				indptr[start_idx+1] = indptr[start_idx]
			else:
				idx = slice(start_idx, op.hyb_ori_row_ends[i] + 1)
				start_indptr = indptr[start_idx]
				indptr[idx] = range(int(start_indptr), int(start_indptr + indices_num), max_bucket_size)
				indptr[op.hyb_ori_row_ends[i] + 1] = start_indptr + indices_num
			start_idx = op.hyb_ori_row_ends[i] + 1

				


		# print(op.position_space[area_i].shape, 
		# 	len(op.position_space[area_i].data), len(op.position_space[area_i].indices), 
		# 	len(indptr),
		# 	len(op.hyb_new_rows[0]), op.position_space[area_i].get_shape()[1])

		print("finish preparation")
		op.position_space[area_i] = scipy.sparse.csr_matrix(
			(op.position_space[area_i].data, op.position_space[area_i].indices, indptr), 
			shape=(len(op.hyb_new_rows[0]), op.position_space[area_i].get_shape()[1]))
		op.inp_getnnz = [op.position_space[area_i].getnnz(axis = 1)]
		
		if op.hyb_short_begin_row == None:

			op.hyb_short_begin_row = len(op.hyb_new_rows[0])
		op.hyb_getnnz_minus = -op.inp_getnnz[0][op.hyb_short_begin_row:]

		op.hyb_ori_row_ends = np.array(op.hyb_ori_row_ends)

		op.position_space_update = op.position_space[area_i].copy()






def pad_irregular_idx(op, area_i, idx_i, max_bucket_size):

	# 
	op.loop_protocals[area_i] = op.loop_protocals[area_i][:idx_i]+'p'+op.loop_protocals[area_i][idx_i+1:]
	# store the new row ids given max bucket size
	if op.op_type == 'spmm':
		gen_position_space_for_area(op, area_i)
		print("finish process position space")

		new_row_nums = np.maximum(np.ceil(op.inp_getnnz[0] / max_bucket_size), 1).astype('int')
		op.hyb_new_rows = [ np.repeat(np.arange(len(op.idx_values_list[area_i][0])), new_row_nums) ]
		op.hyb_ori_row_ends = np.cumsum(new_row_nums) - 1
		short_begin_row_ori = np.searchsorted(-op.inp_getnnz[0], -max_bucket_size)
		if short_begin_row_ori == len(op.inp_getnnz[0]):
			op.hyb_short_begin_row = len(op.hyb_new_rows[0])
		else:
			op.hyb_short_begin_row = op.hyb_ori_row_ends[short_begin_row_ori]


		indptr = np.empty(len(op.hyb_new_rows[0]) + 1)
		indptr[0] = 0
		start_idx = 0
		for i in range(len(op.idx_values_list[area_i][0])):
			indices_num = op.inp_getnnz[0][i]
			if indices_num == 0:
				indptr[start_idx+1] = indptr[start_idx]
			else:
				idx = slice(start_idx, op.hyb_ori_row_ends[i] + 1)
				start_indptr = indptr[start_idx]
				indptr[idx] = range(int(start_indptr), int(start_indptr + indices_num), max_bucket_size)
				indptr[op.hyb_ori_row_ends[i] + 1] = start_indptr + indices_num
			start_idx = op.hyb_ori_row_ends[i] + 1


		# print(op.position_space[area_i].shape, 
		# 	len(op.position_space[area_i].data), len(op.position_space[area_i].indices), 
		# 	len(indptr),
		# 	len(op.hyb_new_rows[0]), op.position_space[area_i].get_shape()[1])

		print("finish preparation")
		op.position_space[area_i] = scipy.sparse.csr_matrix(
			(op.position_space[area_i].data, op.position_space[area_i].indices, indptr), 
			shape=(len(op.hyb_new_rows[0]), op.position_space[area_i].get_shape()[1]))
		op.inp_getnnz = [op.position_space[area_i].getnnz(axis = 1)]
		



		op.hyb_getnnz_minus = -op.inp_getnnz[0][op.hyb_short_begin_row:]

		op.hyb_ori_row_ends = np.array(op.hyb_ori_row_ends)

		op.position_space_update = op.position_space[area_i].copy()






def get_1D_position_space(op, max_bucket_size):
	# get (1) position_space_update, (2) the ori-row-range of each 'new' row.

	assert op.op_type == 'sddmm'

	csr = op.inps[0][op.idx_values_list[0][0],:][:,op.idx_values_list[0][2]]

	csr.has_sorted_indices = False
	csr.sort_indices()
	op.position_space[0] = csr
	op.position_space_update = csr.copy()

	op.nnz_data_1D = csr.data.copy()
	op.nnz_update_1D = np.add.reduceat(
		op.nnz_data_1D, 
		np.arange(0, len(op.nnz_data_1D), op.max_bucket_size)
		)

	return







def get_factors(val):
    ret = list()
    for i in range(1, val+1):
        if val%i == 0:
            ret.append(i)
    return ret


def get_tiles(val, split_n, base):
    if split_n == 2:
        ret = [base+(i,) for i in get_factors(val)]
        return ret
    ret1 = [get_tiles(val // i, split_n-1, base+(i, )) for i in get_factors(val)]
    ret = list()
    for i in ret1:
        ret = ret+i
    return ret
    






def tile_size_heuristics(op_type, tile_sizes_list, max_bucket_size, is_tb_tile = False):
	'''
	Prune the tile sizes candidates using the following heuristics:
		1. the number of threads per thread block should be the multiples of 128;
		2. the number of total threads per thread block <= 1024;
	'''
	if op_type == 'spmm':
		ret = list()
		THREAD_num = 256
		WORKLOAD = max_bucket_size * 8
		for tile_sizes in tile_sizes_list:
			if is_tb_tile:
				thread_num = tile_sizes[0][1]*tile_sizes[1][1]
				if thread_num%128 != 0:
					continue
				# if thread_num>1024:
				# 	continue


				if thread_num<THREAD_num:
					continue


				if tile_sizes[0][1]*tile_sizes[2][1]!=WORKLOAD:
					continue

				ret.append(tile_sizes)
			else:
				thread_num = tile_sizes[0][2]*tile_sizes[1][2]
				# if thread_num%128 != 0:
				# 	continue

				if thread_num%32 != 0:
					continue


				if thread_num>1024:
					continue


				if (tile_sizes[0][2]!=1) or (tile_sizes[1][2]!=32):
					continue


				ret.append(tile_sizes)
	else:
		assert False, f"We do not support {op_type} now!"
	return ret






def get_TC_tile_sizes():
	# return [((None, 32), (None, 32), (None, 32)), #((None, 16), (None, 32), (None, 16)), 
	# 		((None, 16), (None, 32), (None, 32)), #((None, 8), (None, 32), (None, 16)), ((None, 32), (None, 32), (None, 16))
	# 		]

	# FOR DEBUG
	# return [((None, 16), (None, 16), (None, 16)), #((None, 32), (None, 32), (None, 32)), #((None, 16), (None, 32), (None, 16)), 
	# 		#((None, 16), (None, 32), (None, 32)), #((None, 8), (None, 32), (None, 16)), ((None, 32), (None, 32), (None, 16))
	# 		]

	# FOR BENCHMARK SDDMM TC tile
	if 'TC_tile_sizes' in os.environ:
		ret = [json.loads(os.environ['TC_tile_sizes'])]
		ret = [tuple([tuple(j) for j in i]) for i in ret]
		return ret

	# if os.environ['TC_k_notsorted'] == 'True':
	# 	return [((None, 16), (None, 16*i), (None, 16*(2**j))) for i in range(1, 33) for j in range(6)]

	if os.environ['TC_k_notsorted'] == 'True':
		return [((None, 16), (None, 32), (None, 16*i)) for i in range(1, 2)]
	ret = [((None, 16), (None, 32), (None, 16*i)) for i in range(160//16, 0, -1)]  #[((None, 16), (None, 32), (None, 16*i)) for i in range(160//16, 0, -1)]  # ((None, 16), (None, 32), (None, 144))
	
	# for SDDMM TC tile benchmark
	if os.environ['op_type'] == 'sddmm':
		# ret = [((None, 16), (None, 16*(2**j)), (None, 16*i)) for i in range(1, 9) for j in range(3)]
		ret = [((None, 16), (None, 16*(2**j)), (None, 16*i)) for i in range(1, 2) for j in range(3)]
	return ret




def select_the_densest_tensor_for_TC_tiles(TC_ops):
	'''
	Select the TC-type cand op with the highest density (only consider non-empty TC blocks).
	We only consider 16x16x16 blocks here.
	'''
	best_density = 0
	best_TC_op = None
	if TC_ops[0].op_type in ['spmm', 'sddmm']:
		for op in TC_ops:
			gen_position_space_for_area(op, 0)

			row_num = len(op.idx_values_list[0][0])
			blk_i_size = 16
			blk_k_size = 16
			row_blk_num = math.ceil(row_num / blk_i_size)
			csr = op.position_space_update
			tot = sum([ math.ceil(np.count_nonzero( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0) ) / blk_k_size)  \
					for i in range(row_blk_num)]) * blk_i_size * blk_k_size
			density = csr.nnz / tot
			print("density: ", density, "op: ", op.get_key())
			if density > best_density:
				best_TC_op = op
				best_density = density
	return best_TC_op



def set_init_non_empty_blk_num(op):
	'''
	Set the initial non-empty block number per row for the given TC op.
	'''
	k_axis = 2

	tile_sizes_list = get_TC_tile_sizes()
	csr = op.position_space_update
	for tile_sizes in tile_sizes_list:
		row_num = len(op.idx_values_list[0][0])
		blk_i_size = tile_sizes[0][1]
		blk_k_size = tile_sizes[k_axis][1]
		row_blk_num = math.ceil(row_num / blk_i_size)
		op.blk_nums[tile_sizes] = np.asarray([ math.ceil(np.count_nonzero( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0) ) / blk_k_size)  \
					for i in range(row_blk_num)])
		if op.TC_k_notsorted:

			op.blk_nums[tile_sizes] = np.asarray([ np.count_nonzero(
				np.add.reduceat( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0), np.arange(0, csr.shape[1], blk_k_size))
				) for i in range(row_blk_num)])
			op.blk_ks[tile_sizes] = [ np.nonzero(
				np.add.reduceat( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0), np.arange(0, csr.shape[1], blk_k_size))
				)[0] for i in range(row_blk_num)]




def gen_candidate_formats_only_condense(op, op_id_start):
	# store the new row ids given max bucket size
	# this function will generate a new op based on the given op
	if op.op_type == 'spmm':
		sub_op = copy.copy(op)
		sub_op.idx_reordered = copy.deepcopy(op.idx_reordered)
		sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)

		csr = op.position_space_update
		vals = op.idx_values_list[0][0][np.argsort( csr.getnnz(axis=1)<=0, kind='stable' )]
		sub_op.idx_values_list[0][0] = vals
		sub_op.idx_reordered[0][0] = True

		sub_op.op_id = op_id_start
		op_id_start += 1

		return [sub_op], op_id_start




def gen_candidate_TC_formats(op, TC_k_notsorted, reorder_k_by_nnz = True):
	'''
	Generate candidate TC ops, called by gen_candidate_formats.
	'''
	# 1. before we do kernel tile, we need to reorder the rows first to deal with the possible strided sparse pattern.
	last_cand_ops = [op]
	cand_ops = list()
	group_sizes = [2**i for i in range(int(math.log(32*32, 2)))]
	# =====================================

	for op in last_cand_ops:
		# for group_size in group_sizes:
		# 	sub_op = copy.copy(op)
		# 	sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
		# 	regroup_for_strided_pattern(sub_op, group_size)
		# 	cand_ops.append(sub_op)
		# 
		sub_op = copy.copy(op)
		sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
		reorder_for_strided_pattern(sub_op)

		cand_ops.append(op)
	print(f"\n1. regroup rows for strided sparse pattern----cand_ops size: {len(cand_ops)}")

	# 2. do kernel tile
	kernel_tile_sizes = [(1, 1, 1)]
	# max_kernel_tile_Ns = [loop_tile_num[0], 1, loop_tile_num[0]]
	ref_nnz_dicts = dict()
	
	last_cand_ops = cand_ops
	# last_cand_ops = [op]
	cand_ops = list()
	# print(len(loop_tile_num)**3)
	for op in last_cand_ops:
		print(op.get_key())
		for tile_Ns in kernel_tile_sizes:
			to_add = kernel_tile(op, tile_Ns)
			# print(len(to_add))
			cand_ops = cand_ops + to_add
	print(f"\n2. kernel tile----cand_ops size: {len(cand_ops)}")



	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		sub_op = copy.copy(op)
		sub_op.idx_reordered = copy.deepcopy(op.idx_reordered)
		sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
		sub_op.nnz_dict = dict()
		area_i = sub_op.this_area_i
		for idx_i in op.idxs:
			# print(area_i, idx_i)
			# return sub_op, area_i, idx_i
			if op.op_type in ['spmm', 'sddmm']:
				# if (idx_i == 0) or (sub_op.kernel_tile_Ns[idx_i] == 1):
					# reorder_idx(sub_op, area_i, idx_i)

				if (idx_i == 0) or (sub_op.kernel_tile_Ns[idx_i] == 1):
					reorder_idx(sub_op, area_i, idx_i)
		cand_ops.append(sub_op)
		nnz_dict_ref = sub_op.nnz_dict


		if (op.op_type in ['spmm', 'sddmm']) and (op.kernel_tile_Ns[0] == 1) and (op.kernel_tile_Ns[2] == 1):

			cand_ops.append(op)
		# ========================================================================


		if ((op.op_type in ['spmm', 'sddmm']) and (op.kernel_tile_Ns[2] == 1)):
			sub_op = copy.copy(op)
			sub_op.idx_reordered = copy.deepcopy(op.idx_reordered)
			sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
			sub_op.nnz_dict = nnz_dict_ref
			area_i = sub_op.this_area_i
			idx_i = 0
			reorder_idx(sub_op, area_i, idx_i)
			cand_ops.append(sub_op)		
		# -------------------------------
	print(f"\n3. index reorder----cand_ops size: {len(cand_ops)}")
	# 
	# 

	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		if (op.op_type in ['spmm', 'sddmm']) and (not((op.idx_reordered[0][2]) and (op.kernel_tile_Ns[2] == 1))):
			cand_ops.append(op)	
	print(f"\n3.5. kernel tile again----cand_ops size: {len(cand_ops)}")	
	# 
	# 3.7 index reorder again, used for the case where index k has been reordered
	for count, op in enumerate(cand_ops):
		if (op.op_type in ['spmm', 'sddmm']) and op.idx_reordered[0][0] and op.idx_reordered[0][2]:

			reorder_idx(op, 0, 0)		
	print(f"\n3.7. index reorder again----cand_ops size: {len(cand_ops)}")	
	# 
	# 4. apply algorithms from the asymptotic cost model
	if op.op_type == 'spmm':
		for sub_op in cand_ops:
			SPMM_Gustavson_product(sub_op, sub_op.this_area_i) # use this order by default
	elif op.op_type == 'sddmm':
		for sub_op in cand_ops:
			SDDMM_Gustavson_product(sub_op, 0) # use this order by default

	print(f"\n4. algorithm----cand_ops size: {len(cand_ops)}")

	# ============================
	# FOR DEBUG
	# return cand_ops
	# ============================

	# 
	# 5. change compressed to uncompressed
	# dense kernels can only be used when loops are not compressed
	# we only consider dense kernels for an uncompressed tile?
	last_cand_ops = cand_ops
	cand_ops = list()

	ori_TC_tile_sizes_list = get_TC_tile_sizes()
	TC_tile_sizes_list = list()
	TC_is = list()
	for TC_tile_sizes in ori_TC_tile_sizes_list:
		if TC_tile_sizes[0][1] not in TC_is:
			TC_tile_sizes_list.append(TC_tile_sizes)
			TC_is.append(TC_tile_sizes[0][1])

	for sub_op in last_cand_ops:

		# FOR DEBUG===========================================================================================
		if (sum(sub_op.kernel_tile_Ns) == len(sub_op.kernel_tile_Ns)): # and (True not in sub_op.idx_reordered[0]):

			for TC_tile_sizes in TC_tile_sizes_list: # get_TC_tile_sizes():
				new_sub_op = copy.copy(sub_op)
				new_sub_op.loop_protocals = copy.deepcopy(sub_op.loop_protocals)
				new_sub_op.inp_protocals = copy.deepcopy(sub_op.inp_protocals)
				new_sub_op.k_vals = list()
				# change_to_dense(new_sub_op, new_sub_op.this_area_i)
				change_to_dense(new_sub_op, new_sub_op.this_area_i, TC_tile_sizes, TC_k_notsorted, reorder_k_by_nnz = reorder_k_by_nnz)
				cand_ops.append(new_sub_op)
				print(sub_op.get_key())


			# cand_ops.append(sub_op)
		# else:
		# 	cand_ops.append(sub_op)
		# print(new_sub_op.loop_protocals[new_sub_op.this_area_i], new_sub_op.idx_reordered[new_sub_op.this_area_i], sub_op.loop_protocals[sub_op.this_area_i], sub_op.idx_reordered[sub_op.this_area_i])
	print(f"\n5. compressed to uncompressed----cand_ops size: {len(cand_ops)}")
	
	for sub_op in cand_ops:
		assert sub_op.position_space == dict()
		sub_op.position_space = dict()
		sub_op.TC_k_notsorted = TC_k_notsorted

	return cand_ops




'''接下来写怎么一步步生成format。'''
def gen_candidate_formats_BSPMM(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
	reorder_k_by_nnz = True, op_id_start = 0, gen_TC_formats = True):

	TC_ops = list()
	if gen_TC_formats:
		TC_ops = gen_candidate_TC_formats(op, TC_k_notsorted, reorder_k_by_nnz = reorder_k_by_nnz)


	# 1. before we do kernel tile, we need to reorder the rows first to deal with the possible strided sparse pattern.
	last_cand_ops = [op]
	cand_ops = list()
	group_sizes = [2**i for i in range(int(math.log(32*32, 2)))]
	# FOR DEBUG============================
	# group_sizes = [2**i for i in range(1)]
	# cand_ops = [op]
	# =====================================
	for op in last_cand_ops:
		# for group_size in group_sizes:
		# 	sub_op = copy.copy(op)
		# 	sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
		# 	regroup_for_strided_pattern(sub_op, group_size)
		# 	cand_ops.append(sub_op)
		# 
		sub_op = copy.copy(op)
		sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
		reorder_for_strided_pattern(sub_op)

		cand_ops.append(op)
	print(f"\n1. regroup rows for strided sparse pattern----cand_ops size: {len(cand_ops)}")

	# 2. do kernel tile
	loop_tile_num = [2**i for i in [0]] #[2**i for i in [1, 0]] #[2**i for i in [0, 1, 2, 3, 4]]
	kernel_tile_sizes = itertools.product(loop_tile_num, repeat=len(op.idxs))
	kernel_tile_sizes = itertools.product(*([1], [1], loop_tile_num))
	kernel_tile_sizes = itertools.product(*kernel_tile_size_options)
	kernel_tile_sizes = list(kernel_tile_sizes)
	# max_kernel_tile_Ns = [loop_tile_num[0], 1, loop_tile_num[0]]
	ref_nnz_dicts = dict()
	
	last_cand_ops = cand_ops
	# last_cand_ops = [op]
	cand_ops = list()
	# print(len(loop_tile_num)**3)
	for op in last_cand_ops:
		print(op.get_key())
		for tile_Ns in kernel_tile_sizes:
			# print(tile_Ns)
			# new_op = copy.copy(op)
			# if kernel_tile(new_op, tile_Ns):
			# 	cand_ops.append(new_op)
			to_add = kernel_tile(op, tile_Ns)
			# print(len(to_add))
			cand_ops = cand_ops + to_add
	print(f"\n2. kernel tile----cand_ops size: {len(cand_ops)}")



			

	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:

		if (op.kernel_tile_Ns[0] > 1) and (op.kernel_tile_Ns[2] > 1):
			continue
		cand_ops.append(op)



	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		sub_op = copy.copy(op)
		sub_op.idx_reordered = copy.deepcopy(op.idx_reordered)
		sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
		sub_op.nnz_dict = dict()
		area_i = sub_op.this_area_i
		for idx_i in op.idxs:
			# print(area_i, idx_i)
			# return sub_op, area_i, idx_i
			if op.op_type == 'spmm':

				if (idx_i == 0) or (sub_op.kernel_tile_Ns[idx_i] == 1):
					reorder_idx(sub_op, area_i, idx_i)
		cand_ops.append(sub_op)
		nnz_dict_ref = sub_op.nnz_dict


		if (op.kernel_tile_Ns[0] == 1) and (op.kernel_tile_Ns[2] == 1):

			cand_ops.append(op)
		# ========================================================================


		if (op.kernel_tile_Ns[2] == 1):
			sub_op = copy.copy(op)
			sub_op.idx_reordered = copy.deepcopy(op.idx_reordered)
			sub_op.idx_values_list = copy.deepcopy(op.idx_values_list)
			sub_op.nnz_dict = nnz_dict_ref
			area_i = sub_op.this_area_i
			idx_i = 0
			reorder_idx(sub_op, area_i, idx_i)
			cand_ops.append(sub_op)		
		# -------------------------------
	print(f"\n3. index reorder----cand_ops size: {len(cand_ops)}")
	# 

	# -----------------------------------------------

	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		# kernel_tile_sizes = itertools.product(*(loop_tile_num, [1], loop_tile_num))
		print(op.get_key())
		for tile_Ns in kernel_tile_sizes:
			print(tile_Ns)
			if len(tile_Ns) == sum(tile_Ns):
				continue

			if tile_Ns[2] == 1 and op.idx_reordered[0][2]:
				continue
			# 
			to_add = kernel_tile(op, tile_Ns)
			# print(len(to_add))
			cand_ops = cand_ops + to_add
		

		if op.idx_reordered[0][2]:

			continue
		cand_ops.append(op)
	print(f"\n3.5. kernel tile again----cand_ops size: {len(cand_ops)}")	
	# 

	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		if not((op.idx_reordered[0][2]) and (op.kernel_tile_Ns[2] == 1)):
			cand_ops.append(op)
	print(f"\n3.5. kernel tile again----cand_ops size: {len(cand_ops)}")	
	# 
	# 3.7 index reorder again, used for the case where index k has been reordered
	for count, op in enumerate(cand_ops):
		if op.idx_reordered[0][0] and op.idx_reordered[0][2]:

			reorder_idx(op, 0, 0)
	print(f"\n3.7. index reorder again----cand_ops size: {len(cand_ops)}")	
	# 
	# 4. apply algorithms from the asymptotic cost model
	if op.op_type == 'spmm':
		for sub_op in cand_ops:
			SPMM_Gustavson_product(sub_op, sub_op.this_area_i) # use this order by default
	print(f"\n4. algorithm----cand_ops size: {len(cand_ops)}")
	# 



	last_cand_ops = cand_ops
	cand_ops = list()
	pad_idx = 2
	for sub_op in last_cand_ops:
		assert sub_op.position_space == dict()
		sub_op.position_space = dict()

		# condition
		if sub_op.op_type == 'spmm':
			if not sub_op.idx_reordered[sub_op.this_area_i][0]:
				cand_ops.append(sub_op)
				continue
		# 
		if sub_op.loop_protocals[sub_op.this_area_i][pad_idx]=='c':
			print(sub_op.get_key())
			# the other case will be dealt with later
			new_sub_op = copy.copy(sub_op)
			new_sub_op.loop_protocals = copy.deepcopy(sub_op.loop_protocals)
			
			# print((new_sub_op.hyb_new_rows))
			new_sub_op.position_space = dict()
			pad_irregular_idx(new_sub_op, new_sub_op.this_area_i, pad_idx, max_bucket_size)
			# print(len(new_sub_op.idx_values_list[0][0]), len(new_sub_op.hyb_new_rows[0]))

			cand_ops.append(new_sub_op)
			# print(f"padding set : {new_sub_op.loop_protocals[new_sub_op.this_area_i]} {new_sub_op.idx_reordered[new_sub_op.this_area_i]}")
		cand_ops.append(sub_op)
		# print(f"ori set : {sub_op.loop_protocals[sub_op.this_area_i]} {sub_op.idx_reordered[sub_op.this_area_i]}")

	print(f"\n6. padding----cand_ops size: {len(cand_ops)}")


	cand_ops = cand_ops + TC_ops


	# make the position space of each sub_op independent of each other
	ret_cand_ops = list()
	TC_idx_is = list()
	TC_ops = list()
	for sub_op in cand_ops:
		print(sub_op.get_key())
		# prune some sub_ops-------------------
		if sub_op.loop_protocals[0][2] in ['p', 'c']:
			if not sub_op.idx_reordered[0][0]:
				# index i should be reordered-----------------
				continue
		# elif sub_op.loop_protocals[0] == 'uuu':

		elif sub_op.loop_protocals[0] == 'uuu':

			if list(sub_op.idx_values_list[0][0]) in TC_idx_is:
				continue

			TC_idx_is.append(list(sub_op.idx_values_list[0][0]))


			sub_op.idx_reordered = copy.deepcopy(sub_op.idx_reordered)
			if not np.array_equal(sub_op.idx_values_list[0][0], np.sort(sub_op.idx_values_list[0][0])):
				sub_op.idx_reordered[0][0] = True
			else:
				sub_op.idx_reordered[0][0] = False


			TC_ops.append(sub_op)
			continue

		print("SUCCESS")
		# sub_op.position_space = dict()
		# compute the complete idx values for the sub_op as metadata，不需要这个值，因为complete_idx_values应该是由tile决定
		# comp_complete_idx_values(sub_op)

		sub_op.op_id = op_id_start
		op_id_start += 1
		ret_cand_ops.append(sub_op)




	if len(TC_ops) > 0:
		TC_op = select_the_densest_tensor_for_TC_tiles(TC_ops)
		set_init_non_empty_blk_num(TC_op)
		print("best TC op : ", TC_op.get_key())
		TC_op.op_id = op_id_start
		op_id_start += 1
		ret_cand_ops.append(TC_op)

	# first print the information of operators
	# print("PRINT SUB-OP INFOR----------------------------------")
	# for sub_op_id, sub_op in enumerate(cand_ops):
	# 	print(f"op id: {sub_op_id}, op key: {sub_op.get_key()}")

	return ret_cand_ops, op_id_start



def gen_candidate_formats_BSDDMM(op, max_bucket_size, TC_k_notsorted,
	reorder_k_by_nnz = True, op_id_start = 0, gen_TC_formats = True):
	TC_ops = list()
	if gen_TC_formats:
		TC_ops = gen_candidate_TC_formats(op, TC_k_notsorted, reorder_k_by_nnz = reorder_k_by_nnz)


	op_1Dtile = copy.copy(op)
	SDDMM_Gustavson_product(op_1Dtile, 0)
	op_1Dtile.max_bucket_size = max_bucket_size

	op_1Dtile.position_space = dict()
	get_1D_position_space(op_1Dtile, max_bucket_size)

	# op_1Dtile.position_space = dict()

	cand_ops = [op_1Dtile] + TC_ops

	# make the position space of each sub_op independent of each other
	ret_cand_ops = list()
	TC_idx_is = list()
	TC_ops = list()
	for sub_op in cand_ops:
		print(sub_op.get_key())
		# prune some sub_ops-------------------
		if sub_op.loop_protocals[0] == 'uuu':

			if list(sub_op.idx_values_list[0][0]) in TC_idx_is:
				continue

			TC_idx_is.append(list(sub_op.idx_values_list[0][0]))


			sub_op.idx_reordered = copy.deepcopy(sub_op.idx_reordered)
			if not np.array_equal(sub_op.idx_values_list[0][0], np.sort(sub_op.idx_values_list[0][0])):
				sub_op.idx_reordered[0][0] = True
			else:
				sub_op.idx_reordered[0][0] = False


			TC_ops.append(sub_op)
			continue

		print("SUCCESS")

		sub_op.op_id = op_id_start
		op_id_start += 1
		ret_cand_ops.append(sub_op)



	if len(TC_ops) > 0:
		TC_op = select_the_densest_tensor_for_TC_tiles(TC_ops)
		set_init_non_empty_blk_num(TC_op)
		print("best TC op : ", TC_op.get_key())
		TC_op.op_id = op_id_start
		op_id_start += 1
		ret_cand_ops.append(TC_op)


	return ret_cand_ops, op_id_start



def gen_candidate_formats(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
	reorder_k_by_nnz = True, op_id_start = 0, gen_TC_formats = True):

	if op.op_type in ['spmm', 'batched_spmm']:
		return gen_candidate_formats_BSPMM(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
				reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = op_id_start, gen_TC_formats = gen_TC_formats)
	elif op.op_type in ['sddmm']:
		return gen_candidate_formats_BSDDMM(op, max_bucket_size, TC_k_notsorted, 
				reorder_k_by_nnz = reorder_k_by_nnz, op_id_start = op_id_start, gen_TC_formats = gen_TC_formats)







def get_tile_sizes_idx(sub_op, max_bucket_size, is_tb_tile = False):
	'''
	is_tb_tile = False: the tile sizes are generated for tiles with concrete implementation details.
	           = True: the tile sizes are generated for tiles for thread blocks, without concrete thread numbers.
	'''
	area_i = sub_op.this_area_i
	tile_sizes_idx = None
	
	tile_level_num = 3
	if is_tb_tile:
		tile_level_num = 2
	
	if sub_op.op_type == 'spmm':
		


		# ------------------------------------------------------
		if sub_op.loop_protocals[area_i][2] == 'c':
			tile_sizes_idx = [get_tiles(len(sub_op.idx_values_list[area_i][idx_i]), tile_level_num, (None,)) \
							for idx_i in [0,1]] + [[(1, len(sub_op.idx_values_list[area_i][2]))]]
		elif sub_op.loop_protocals[area_i][2] == 'p':
			tot_lens = [math.ceil(len(sub_op.hyb_new_rows[0])/128)*128, len(sub_op.idx_values_list[area_i][1])]
			tile_sizes_idx = [get_tiles(tot_lens[idx_i], tile_level_num, (None,)) \
							for idx_i in [0,1]] + [[(1, 2**tmp) for tmp in range(int(math.log(max_bucket_size, 2)) + 1) ]]
			

			# tile_sizes_idx[0] = [(None, 32*8//(2**tmp)) for tmp in range(int(math.log(max_bucket_size, 2)) + 1)]
			tile_sizes_idx[0] = [(None, 2**tmp) for tmp in range(int(math.log(len(sub_op.hyb_new_rows[0]), 2)) + 1)]

		elif sub_op.loop_protocals[area_i] == 'uuu':

			tile_sizes_idx = get_TC_tile_sizes()
			# [((None, 32), (None, 32), (None, 32)), #((None, 16), (None, 32), (None, 16)), 
			# ((None, 16), (None, 32), (None, 32)), #((None, 8), (None, 32), (None, 16)), ((None, 32), (None, 32), (None, 16))
			# ]

	elif sub_op.op_type == 'sddmm':
		if get_template_str(sub_op) == 'TC_sddmm':
			tile_sizes_idx = get_TC_tile_sizes()
			tile_j = set([tile_sizes[1][1] for tile_sizes in tile_sizes_idx])
			tile_j = max([j for j in tile_j if j <= sub_op.idx_lens[1]])
			tile_sizes_idx = [tile_sizes for tile_sizes in tile_sizes_idx if tile_sizes[1][1] == tile_j]
			# tile_sizes_idx = [((None, 16), (None, 16), (None, 16*i)) for i in [1]]
		elif get_template_str(sub_op) == '1D_sddmm':
			tile_sizes_idx = [(max_bucket_size,)] 
	return tile_sizes_idx





def get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile = False):
	'''
	is_tb_tile = False: the tile sizes are generated for tiles with concrete implementation details.
	           = True: the tile sizes are generated for tiles for thread blocks, without concrete thread numbers.
	'''
	area_i = sub_op.this_area_i
	sub_op_id = sub_op.op_id
	if sub_op.op_type == 'spmm':
		if sub_op.loop_protocals[area_i] != 'uuu':
			tile_sizes_list = itertools.product(*tile_sizes_idx)
	# 		tile_pos_idx = [[math.ceil(len(idx_values) / math.prod(sizes[1:])) for sizes in tile_sizes_idx[idx_i]] \
	# 							for idx_i, idx_values in enumerate(sub_op.idx_values_list[area_i][:2])]
			tile_sizes_list = list(tile_sizes_list)
			# print(f"sub_op {sub_op_id}, possible tile sizes: {len(tile_sizes_list)}")

			tile_sizes_list = tile_size_heuristics(sub_op.op_type, tile_sizes_list, max_bucket_size, is_tb_tile = is_tb_tile)
		else:
			tile_sizes_list = tile_sizes_idx
		# print(f"possible tile sizes after prune: {len(tile_sizes_list)}")
		return tile_sizes_list
	elif sub_op.op_type == 'sddmm':
		return tile_sizes_idx




def get_tile_pos_list(sub_op, tile_sizes):
	area_i = sub_op.this_area_i
	tile_poses_idx = None
	if sub_op.op_type == 'spmm':
		# tile_poses_idx = [list(range(math.ceil(len(idx_values) / math.prod(tile_sizes[idx_i][1:])))) \
		# 			for idx_i, idx_values in enumerate(sub_op.idx_values_list[area_i][:2])] + [[0]]
		if sub_op.loop_protocals[area_i][2] == 'c':
			tile_poses_idx = [list(range(math.ceil(len(idx_values) / math.prod(tile_sizes[idx_i][1:])))) \
						for idx_i, idx_values in enumerate(sub_op.idx_values_list[area_i][:1])] + [[0], [0]]
		elif sub_op.loop_protocals[area_i][2] == 'p':
			tile_poses_idx = [list(range(math.ceil(len(sub_op.hyb_new_rows[0]) / math.prod(tile_sizes[0][1:])))) ] + [[0], [0]]
		elif sub_op.loop_protocals[area_i] == 'uuu':
			# tile_poses_idx = [(0, 0, 0)]
			tile_poses_idx = [list(range(math.ceil(len(idx_values) / math.prod(tile_sizes[idx_i][1:])))) \
						for idx_i, idx_values in enumerate(sub_op.idx_values_list[area_i][:3])]
			tile_poses_idx[1] = [0]

		tile_pos_list = itertools.product(*tile_poses_idx)
		# if sub_op.loop_protocals[area_i] != 'uuu':
		# 	tile_pos_list = itertools.product(*tile_poses_idx)
		# else:
		# 	tile_pos_list = tile_poses_idx

		return tile_pos_list



def get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile = False):
	params_list = list()
	area_i = sub_op.this_area_i

	use_implicit_unroll_values = [True, False]
	if is_tb_tile:
		use_implicit_unroll_values = [True]

	if sub_op.op_type == 'spmm':
		for use_implicit_unroll in use_implicit_unroll_values: #[True, False]:
			params = {'use_implicit_unroll': use_implicit_unroll}
			if sub_op.loop_protocals[area_i][2] == 'p':
				params['last_bucket'] = False
				if math.prod(tile_sizes[2][1:]) == max_bucket_size:
					params['last_bucket'] = True
			elif sub_op.loop_protocals[area_i] == 'uuu':
				if not params['use_implicit_unroll']:
					continue

				if tile_sizes == ((None, 8), (None, 32), (None, 16)):
					params['mma_shape_str'] = "m8n32k16"
				else:
					params['mma_shape_str'] = "m16n16k16"
			params_list.append(params)
	elif sub_op.op_type == 'sddmm':
		if get_template_str(sub_op) == '1D_sddmm':
			params = {'tx': 8, 'ty': 4, 'vec_size': 4, 'group_size': 8, 'max_bucket_size': max_bucket_size}
			params_list.append(params)
		elif get_template_str(sub_op) == 'TC_sddmm':
			params = {'mma_shape_str': "m16n16k16", 'warp_num': 1, 'vec_size': 4}
			params_list.append(params)

	return params_list




def get_template_str(op):
	if op.op_type == 'spmm':
		if op.loop_protocals[op.this_area_i][2] == 'c':
			return "sparse_template"
		elif op.loop_protocals[op.this_area_i][2] == 'p':
			return "sparse_template_ell"
		elif op.loop_protocals[op.this_area_i] == 'uuu':
			return "TensorCore_template"
	elif op.op_type == 'sddmm':
		if op.loop_protocals[0] == 'uuu':
			return "TC_sddmm"
		else:
			return "1D_sddmm"





def gen_position_space_for_tile(tile):
	'''
	Generate the initial position space (i.e., no element has been covered) for the given tile.
	'''
	op = tile.op
	tile_sizes = tile.tile_sizes
	tile_pos = tile.tile_pos

	area_i = op.this_area_i
	idx_values = op.idx_values_list[area_i]
	# gen_position_space_for_area(op, area_i)

	if op.op_type == 'spmm':
		# v2idx = get_val_2_ind(idx_values[2])
		template_str = get_template_str(op)

		tile_rngs = [(int(tile_pos[i]*math.prod(tile_sizes[i][1:])), int((tile_pos[i]+1)*math.prod(tile_sizes[i][1:])))
					for i in op.idxs]
		
		# tile.uncovered_position_space['j'] = set(idx_values[1][tile_rngs[1][0] : tile_rngs[1][1]])
		# # store the value range for index k, here is idx_values[2] because we do not tile on index k, but we do not use this for the last bucket of ELL
		# # tile.uncovered_position_space['k'] = set(idx_values[2])
		# tile.uncovered_position_space['in1'] = dict() # key: i index, value: corr. k values.

		if template_str == "sparse_template":


			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]

		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			
			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(op.hyb_new_rows[0]))),:]


		elif template_str == "TensorCore_template":

			ks = tile.op.k_vals[tile_pos[0]][tile_rngs[2][0] : min(tile_rngs[2][1], len(idx_values[2]))]
			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]\
				[:, ks]
	elif op.op_type == 'sddmm':
		template_str = get_template_str(op)
		if template_str == '1D_sddmm':

			tile.uncovered_position_space = (op.position_space[0].data)[ tile_pos[0]*tile_sizes[0]:(tile_pos[0]+1)*tile_sizes[0] ]
		elif template_str == 'TC_sddmm':
			ks = tile.op.k_vals[tile_pos[0]][tile.tile_k_rng[0] : tile.tile_k_rng[1]+1]
			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile.tile_i_rng[0], tile.tile_i_rng[1]+1),:]\
				[:, ks]



def gen_updated_position_space_for_tile(tile):
	'''
	Generate the updated position space for the given tile: tile.uncovered_position_space.
	'''
	op = tile.op
	tile_sizes = tile.tile_sizes
	tile_pos = tile.tile_pos
	idx_values = op.idx_values_list[0]

	if op.op_type == 'spmm':
		# v2idx = get_val_2_ind(idx_values[2])
		template_str = get_template_str(op)

		tile_rngs = [( int(tile_pos[i]*math.prod(tile_sizes[i][1:])), int((tile_pos[i]+1)*math.prod(tile_sizes[i][1:]))  )
					for i in op.idxs]
		
		# tile.uncovered_position_space['j'] = set(idx_values[1][tile_rngs[1][0] : tile_rngs[1][1]])
		# # store the value range for index k, here is idx_values[2] because we do not tile on index k, but we do not use this for the last bucket of ELL
		# # tile.uncovered_position_space['k'] = set(idx_values[2])
		# tile.uncovered_position_space['in1'] = dict() # key: i index, value: corr. k values.

		if template_str == "sparse_template":
			# we do not tile on index k, but the computation below still deals with the case which tiles on k
			# there is no padding in this template


			tile.uncovered_position_space = op.position_space_update\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]

		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			
			tile.uncovered_position_space = op.position_space_update\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(op.hyb_new_rows[0]))),:]


		elif template_str == "TensorCore_template":

			ks = tile.op.k_vals[tile_pos[0]][tile_rngs[2][0] : min(tile_rngs[2][1], len(idx_values[2]))]
			tile.uncovered_position_space = op.position_space_update\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]\
				[:, ks]			
	elif op.op_type == 'sddmm':
		template_str = get_template_str(op)
		if template_str == '1D_sddmm':

			tile.uncovered_position_space = op.nnz_data_1D[ tile_pos[0]*tile_sizes[0]:(tile_pos[0]+1)*tile_sizes[0] ]


			# tile.uncovered_position_space = scipy.sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=( len(indptr)-1, op.position_space_update.shape[1] ))
		elif template_str == 'TC_sddmm':
			ks = tile.op.k_vals[tile_pos[0]][tile.tile_k_rng[0] : tile.tile_k_rng[1]+1]
			tile.uncovered_position_space = op.position_space_update\
				[range(tile.tile_i_rng[0], tile.tile_i_rng[1]+1),:]\
				[:, ks]


def comp_complete_idx_values_given_tile(tile_op, tile_idx_values):

	if tile_op.op_type == 'spmm':
		complete_idx_values = [None, None, None]
		for i in [0, 2]:
			complete_idx_values[i] = np.concatenate(
				(tile_idx_values[i], np.array( list( set(range(tile_op.idx_lens[i])).difference(tile_idx_values[i]) ) ) )
				)
			
		return complete_idx_values

 
def transform_covered_position_space_to_ori_space(selected_tile):
	op = selected_tile.op
	if op.op_type == 'spmm':
		covered_csr = selected_tile.position_space_when_selected

		tile_idx_values = [None, None, None]


		unfold_covered_csr = None
		if get_template_str(op) == 'sparse_template_ell':

			tile_sizes = selected_tile.tile_sizes
			# tile_i_rng = math.prod(tile_sizes[0][1:]) * tile_pos[0], \
			# 	min(math.prod(tile_sizes[0][1:]) * (tile_pos[0]+1), len(op.hyb_new_rows[0])) - 1

			tile_i_rng = selected_tile.tile_i_rng

			row_rng_before_fold = [op.hyb_new_rows[0][i] for i in tile_i_rng]
			
			row_end_offset = op.hyb_ori_row_ends[ row_rng_before_fold[0]:row_rng_before_fold[1]+1 ] - tile_i_rng[0]


			row_end_offset[-1] = min(row_end_offset[-1], math.prod(tile_sizes[0][1:])-1 )

			indptr = np.concatenate( ([0], covered_csr.indptr[ row_end_offset+1 ]) )

			unfold_covered_csr = scipy.sparse.csr_matrix(
				(covered_csr.data, covered_csr.indices, indptr), 
				shape=(row_rng_before_fold[1] - row_rng_before_fold[0] + 1, covered_csr.get_shape()[1]))

			tile_idx_values[0] = op.idx_values_list[0][0][row_rng_before_fold[0]:row_rng_before_fold[1]+1]

		else:
			unfold_covered_csr = covered_csr.copy()
			tile_idx_values[0] = op.idx_values_list[0][0][selected_tile.tile_i_rng[0]:selected_tile.tile_i_rng[1]+1]

	
		unfold_covered_csr.resize(op.inps[0].shape)


		tile_idx_values[2] = op.idx_values_list[0][2] # [selected_tile.tile_k_rng[0]:selected_tile.tile_k_rng[1]+1]
		if get_template_str(op) == 'TensorCore_template':
			tile_idx_values[2] = op.idx_values_list[0][2][\
				op.k_vals[selected_tile.tile_pos[0]][selected_tile.tile_k_rng[0]:selected_tile.tile_k_rng[1]+1] ]

		complete_idx_values = comp_complete_idx_values_given_tile(op, tile_idx_values)

		ind_i = np.argsort(complete_idx_values[0])
		ind_k = np.argsort(complete_idx_values[2])


		unfold_covered_csr = unfold_covered_csr[ind_i,:][:,ind_k]
	return unfold_covered_csr





def set_j_len(tile):
	# compute the number of j values covered by this tile
	if tile.op.op_type == 'spmm':
		tile_j_rng = (tile.tile_pos[1]*math.prod(tile.tile_sizes[1][1:]), (tile.tile_pos[1]+1)*math.prod(tile.tile_sizes[1][1:]))
		
		idx_values = tile.op.idx_values_list[0]
		# self.j_num = len(idx_values[1][tile_j_rng[0] : tile_j_rng[1]])
		tile.j_num = min(tile_j_rng[1], len(idx_values[1])) - tile_j_rng[0]
	elif tile.op.op_type == 'sddmm':
		tile.j_num = tile.op.idx_lens[1]

def init_nnz(tile):

	if get_template_str(tile.op) in ["TensorCore_template", 'TC_sddmm']:
		gen_position_space_for_tile(tile)
		tile.nnz = tile.uncovered_position_space.getnnz()
	elif get_template_str(tile.op) == 'sparse_template_ell':
		# tile_i_rng = math.prod(tile_sizes[0][1:]) * tile_pos[0], \
		# 		min(math.prod(tile_sizes[0][1:]) * (tile_pos[0]+1), self.op.position_space[0].shape[0] ) - 1

		tile.nnz = tile.op.position_space[0].indptr[tile.tile_i_rng[1]+1] - \
			tile.op.position_space[0].indptr[tile.tile_i_rng[0]]
	elif get_template_str(tile.op) == '1D_sddmm':
		# tile.nnz = tile.op.position_space[0].indptr[tile.tile_pos[0]+1] - tile.op.position_space[0].indptr[tile.tile_pos[0]]
		tile.nnz = min((tile.tile_pos[0]+1)*tile.tile_sizes[0], tile.op.position_space[0].nnz) - tile.tile_pos[0]*tile.tile_sizes[0]


class ComputeTile(object):
	"""docstring for ComputeTile"""
	def __init__(self, op, tile_sizes, tile_pos, params):
		super(ComputeTile, self).__init__()
		self.op = op
		self.tile_sizes = tile_sizes
		self.tile_pos = tile_pos
		self.params = params
		self.nnz = 0 #count_tile_nnz_iter_space(op, tile_sizes, tile_pos)
		self.cost = None
		self.pred_cost = None
		self.j_num = None

		self.nnz_uncovered = 0
		self.avg_cost = None # self.avg_cost = self.cost / self.nnz_uncovered. Every time when self.nnz_uncovered is updated, update self.avg_cost.
		self.pred_avg_cost = None
		self.uncovered_position_space = dict() # this variable will be updated constantly

		self.best_tile_sizes = None # when the compute tile is a thread block tile, this variable stores the best 3 level tile size setting for it.
		self.best_params = None

		self.position_space_when_selected = None
		self.nnz_when_selected = None

		self.tile_i_rng = None # should be a tuple, both ends are included
		self.tile_k_rng = None # should be a tuple, both ends are included  NOTE: seems only used for TC tiles
		self.k_vals = None # should be a list of k vals, when this is not None, self.tile_k_rng should be None. Used in postprocess.
		self.i_vals = None # similar as self.k_vals

		self.is_atomic_tile = None # is related to how the pred cost of this tile is computed

		self.remap_1d = None


		# initialize self.nnz
		if (op.op_type == 'spmm') or (get_template_str(op) == 'TC_sddmm'):

			self.tile_i_rng = math.prod(tile_sizes[0][1:]) * tile_pos[0], \
					min(math.prod(tile_sizes[0][1:]) * (tile_pos[0]+1), op.position_space_update.shape[0] ) - 1

			self.tile_k_rng = math.prod(tile_sizes[2][1:]) * tile_pos[2], \
					min(math.prod(tile_sizes[2][1:]) * (tile_pos[2]+1), op.position_space_update.shape[1] ) - 1

			self.tile_k_rng = int(self.tile_k_rng[0]), int(self.tile_k_rng[1])

			set_j_len(self) 
			init_nnz(self)
			self.nnz_uncovered = self.nnz * self.j_num
		elif get_template_str(op) == '1D_sddmm':
			self.tile_i_rng = None #tile_pos[0], tile_pos[0]
			self.tile_k_rng = None # 0, tile_sizes[0]-1
			set_j_len(self) 
			init_nnz(self)
			self.nnz_uncovered = self.nnz * self.j_num			

	def get_k_vals(self):

		if get_template_str(self.op) == 'TensorCore_template':
			self.k_vals = self.op.idx_values_list[0][2][ self.tile_k_rng[0]:self.tile_k_rng[1]+1 ]

	def check_atomic(self, ori_op, max_bucket_size):

		if get_template_str(self.op) == "sparse_template_ell":
			if self.tile_i_rng[0] < self.op.hyb_short_begin_row:
				self.is_atomic_tile = True
				return

		if get_template_str(self.op) == "sparse_template_ell":
			rows = np.array(list(set(self.op.idx_values_list[0][0][self.op.hyb_new_rows[0][self.tile_i_rng[0]:self.tile_i_rng[1]+1]])))
		else:
			rows = self.op.idx_values_list[0][0][self.tile_i_rng[0]:self.tile_i_rng[1]+1]

		ori_nnz = sum(ori_op.inps[0].getnnz(axis=1)[rows])
		self.is_atomic_tile = (ori_nnz > self.nnz)


	def update_nnz_uncovered(self):

		if get_template_str(self.op) in ["TensorCore_template", "TC_sddmm"]:
			gen_updated_position_space_for_tile(self)
			self.nnz_uncovered = self.j_num * self.uncovered_position_space.getnnz()
		elif get_template_str(self.op) == '1D_sddmm':
			self.nnz_uncovered = self.j_num * self.op.nnz_update_1D[ self.tile_pos[0] ]
		else:
			# tile_i_rng = math.prod(tile_sizes[0][1:]) * tile_pos[0], \
			# 		min(math.prod(tile_sizes[0][1:]) * (tile_pos[0]+1), self.op.position_space_update.shape[0] ) - 1

			self.nnz_uncovered = self.j_num * \
				(self.op.position_space_update.indptr[self.tile_i_rng[1]+1] - \
					self.op.position_space_update.indptr[self.tile_i_rng[0]])

	def set_avg_cost(self):
		if self.cost < 0:

			assert False, f"the cost of this tile is {self.cost} < 0"
			self.cost = float('inf')

		if self.nnz_uncovered == 0:
			self.avg_cost = float('inf')
		else:
			self.avg_cost = self.cost / self.nnz_uncovered
	
	def set_pred_avg_cost(self):
		if self.pred_cost == None:

			assert 'mma_shape_str' in self.params, "Should be a TC tile"
			return
		

		self.pred_avg_cost = self.pred_cost / np.float32(self.nnz_uncovered)



	def get_key(self, as_dict_key=False):
		if as_dict_key:
			return (self.op.op_id, tuple(self.tile_sizes), tuple(self.tile_pos), json.dumps(self.params), self.is_atomic_tile)
		else:
			return (self.op.op_id, self.tile_sizes, self.tile_pos, self.params, self.is_atomic_tile)




def is_valid_tile(sub_op, tile_sizes, tile_pos, params):
	'''
	Return True if the tile is valid.
	'''
	area_i = 0
	if sub_op.op_type == 'spmm':
		if sub_op.loop_protocals[area_i][2] == 'p':

			
			new_row_i = tile_pos[0]*math.prod(tile_sizes[0][1:])
			if new_row_i < sub_op.hyb_short_begin_row:
				# should be the last bucket
				if (not params['last_bucket']):
					return False
			else:
				ori_row_i = sub_op.hyb_new_rows[0][new_row_i] # idx_values[0][new_row_i]
				indices = get_nnz_from_dict(sub_op.position_space[area_i], [ori_row_i])

				if (len(indices) <= (math.prod(tile_sizes[2][1:])//2)) or (len(indices) > math.prod(tile_sizes[2][1:])):
					# print("too much padding")
					# return -1
					return False
	return True



def gen_tiles_given_op(sub_op, max_bucket_size, is_tb_tile = False):
	'''
		Generate all the possible tiles for sub_op.
	'''	
	all_tiles = list()

	sub_op_id = sub_op.op_id
	area_i = sub_op.this_area_i

	tile_sizes_idx = get_tile_sizes_idx(sub_op, max_bucket_size, is_tb_tile=is_tb_tile)
	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, is_tb_tile=is_tb_tile)
	for tile_sizes in tile_sizes_list:
		tile_pos_list = get_tile_pos_list(sub_op, tile_sizes)
		for tile_pos in tile_pos_list:
			params_list = get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile=is_tb_tile)
			for params in params_list:

				# if list(tile_sizes)[:2] == [128, 128]:
				# 	print(is_valid_tile(sub_op, tile_sizes, tile_pos, params))

				if not is_valid_tile(sub_op, tile_sizes, tile_pos, params):
					continue
				tile = ComputeTile(sub_op, tile_sizes, tile_pos, params)
				if tile.nnz > 0:
					all_tiles.append(tile)

	return all_tiles



def gen_tile_sizes_given_tile_tb(tile_tb, max_bucket_size):
	'''
	Generate concrete tile sizes with concrete thread numbers given the thread block tile workload.
	Require:
		the input tile_tb has only 2 tile levels for each loop.
	'''	
	tile_sizes_idx = None
	
	sub_op = tile_tb.op
	op_type = sub_op.op_type
	loop_protocals = sub_op.loop_protocals
	tile_sizes_tb = tile_tb.tile_sizes
	area_i = sub_op.this_area_i

	if op_type == 'spmm':	
		if loop_protocals[area_i] != 'uuu':
			tile_sizes_idx = [ [ [tile_size[0], tile_size[1] // v , v ] for v in get_factors(tile_size[1])] for tile_size in tile_sizes_tb[:-1] ] + [[tile_sizes_tb[-1]]]
		else:
			tile_sizes_idx = [tile_sizes_tb]

	tile_sizes_list = get_tile_sizes_list(sub_op, tile_sizes_idx, max_bucket_size, is_tb_tile = False)

	return tile_sizes_list




				


def simple_test():
	op_type = 'spmm'
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [128, 128, 128] # [16, 16, 16] # [128, 128, 128]
	m, n, k = idx_lens
	idx_graph = None
	A = scipy.sparse.rand(m, k, 0.2, 'csr', 'float32', random_state=0)
	A_val = tuple( np.array([1]*len(A.indices)).astype("float32") )
	A = scipy.sparse.csr_matrix((A_val, A.indices, A.indptr), shape=A.shape)

	A_csc = A.tocsc()

	rng = np.random.default_rng(seed=0)
	# B = np.random.rand(k,n).astype("float32")
	B = rng.random((k,n)).astype("float32")
	# print("B:")
	# print(B.tolist())

	inps = [A, DenseTensor(B)]
	sparse_inps_transposed = [A_csc]
	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)




def csr_to_dcsr(op_type, data):
	if op_type == 'spmm':
		indptr = data
		new_row_is, new_ptr = list(), [0]
		for i in range(len(indptr)-1):
			if indptr[i] < indptr[i+1]:
				new_row_is.append(i)
				new_ptr.append(indptr[i+1])
		return new_row_is, new_ptr





def dcsr_to_csr(op_type, data, idx_lens):
	if op_type == 'spmm':
		row_is, ind_ptr = data
		new_ind_prt = [0]
		ptr = 0
		for i in range(idx_lens[0]):
			if (ptr >= len(row_is)) or (i < row_is[ptr]):
				new_ind_prt.append(new_ind_prt[-1])
			elif i == row_is[ptr]:
				new_ind_prt.append(ind_ptr[ ptr+1 ])
				ptr += 1

		return new_ind_prt







def test_real_op(op_type, name = "arxiv", feat_size = 32, pad=True):
	'''
	if pad is True, we will pad the shape of the sparse matrix to multiples of 128.
	'''
	from utils import get_dataset

	# op_type = 'spmm'
	# name = "arxiv"
	etype=None
	if type(name) == tuple:
		name, etype = name
	g = get_dataset(name)

	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = None

	indptr, indices, _ = g.adj_tensors("csr", etype=etype) 
	if pad:
		if op_type == 'spmm':

			idx_lens = [math.ceil( (len(indptr)-1) /128)*128, feat_size, math.ceil( (max(indices)+1) /128)*128]
		elif op_type == 'sddmm':
			idx_lens = [math.ceil(g.num_dst_nodes()/128)*128, feat_size, math.ceil(g.num_src_nodes()/160)*160]
	else:
		idx_lens = [g.num_dst_nodes(), feat_size, g.num_src_nodes()]
	idx_graph = None
	
	# x = th.rand((g.num_src_nodes(), feat_size))
	rng = np.random.default_rng(seed=0)
	# B = np.random.rand(k,n).astype("float32")
	Bs = list()
	if op_type == 'spmm':
		Bs = [rng.random((idx_lens[2],feat_size)).astype("float32")]
	elif op_type == 'sddmm':
		Bs = [rng.random((idx_lens[0],feat_size)).astype("float32"), rng.random((idx_lens[2],feat_size)).astype("float32")]
	# B = rng.random((idx_lens[2],feat_size)).astype("float32")

	# indptr, indices, _ = g.adj_sparse("csr")
	indptr, indices, _ = g.adj_tensors("csr", etype=etype) 
	# new_row_is, new_ptr = csr_to_dcsr(op_type, indptr)

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	i_pad_num = 0
	if pad:
		# i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 - g.num_dst_nodes()

		# i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 + 1 - len(indptr)
		i_pad_num = idx_lens[0] + 1 - len(indptr)
	# indptr = list(indptr) + [indptr[-1] for tmp in range(i_pad_num)]
	indptr = np.concatenate( [indptr, np.full(i_pad_num, indptr[-1])] )

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	# A_val = tuple( np.array([1]*len(indices)).astype("float32") )
	A_val = np.full(len(indices), 1, dtype="float32")

	A_csr = scipy.sparse.csr_matrix((A_val, indices, indptr), shape=( idx_lens[0], idx_lens[2] ))


	# return A_data, A_raw_poses
	# inps = [A_csr, DenseTensor(B)]
	inps = [A_csr] + [DenseTensor(B) for B in Bs]
	sparse_inps_transposed = [ A_csr.tocsc() ]


	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)



# test the spmm operator from pruned-bert
def test_real_op_pruned_bert(op_type, data_i, feat_size = 32, print_infor=True):
	from transformers import AutoTokenizer, AutoModelForQuestionAnswering

	tokenizer = AutoTokenizer.from_pretrained(
		"madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
	)

	model = AutoModelForQuestionAnswering.from_pretrained(
		"madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
	)

	csr_weight, x = None, None
	count = 0
	for name, param in model.named_parameters():
		if (
			name.endswith("key.weight")
			or name.endswith("value.weight")
			or name.endswith("query.weight")
			or name.endswith("dense.weight")
		):
			# 
			# bsr_weight = sp.bsr_matrix(
				# param.detach().numpy(), shape=param.shape, blocksize=(32, 32)
			# )
			# 
			csr_weight = scipy.sparse.csr_matrix(param.detach().numpy())
			if print_infor:
				print(count, ": ", csr_weight.nnz / param.numel(), name, type(param))
			x = torch.rand(csr_weight.shape[1], feat_size).half()
			# 
			if count == data_i:
				break
			count+=1


	A_val = tuple( np.array([1]*len(csr_weight.indices)).astype("float32") )
	A_csr = scipy.sparse.csr_matrix((A_val, csr_weight.indices, csr_weight.indptr), shape=csr_weight.shape)


	# return A_data, A_raw_poses
	inps = [A_csr]
	# inps = [A_csr, DenseTensor(x.numpy())]
	if op_type == 'spmm':
		inps = inps + [DenseTensor(x.numpy())]
	if op_type == 'sddmm':
		x1 = torch.rand(A_csr.shape[0], feat_size).half()
		x2 = torch.rand(A_csr.shape[1], feat_size).half()
		inps = inps + [DenseTensor(x1.numpy()), DenseTensor(x2.numpy())]
	sparse_inps_transposed = [ A_csr.tocsc() ]


	# op_type = 'spmm'
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [A_csr.shape[0], feat_size, A_csr.shape[1]]
	idx_graph = None

	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed), count




# test the spmm operator from pruned-bert
def test_real_op_pruned_bert_unstructured(op_type, data_i, feat_size = 32, print_infor=True):
	from transformers import AutoTokenizer, AutoModelForQuestionAnswering

	tokenizer = AutoTokenizer.from_pretrained(
		"huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad")

	model = AutoModelForQuestionAnswering.from_pretrained(
		"huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad")

    # W * X: W: (64, 32) X: (32, 128)
	nnz_params = 0
	ttl_params = 0
	for name, param in model.named_parameters():
		if name.endswith("dense.weight") or name.endswith(
				"key.weight") or name.endswith(
					"value.weight") or name.endswith("query.weight"):
			ttl_params += param.numel()
			nnz_params += len(param.nonzero())

	if print_infor:
		print(nnz_params / ttl_params)

	csr_weight, x = None, None
	count = 0
	for name, param in model.named_parameters():
		if name.endswith("dense.weight") or name.endswith(
				"key.weight") or name.endswith(
					"value.weight") or name.endswith("query.weight"):
			csr_weight = scipy.sparse.csr_matrix(param.detach().numpy())
			if print_infor:
				print(count, ": ", csr_weight.nnz / param.numel(), name, type(param), param.shape)
			x = torch.rand(csr_weight.shape[1], feat_size).half()
			# 
			if count == data_i:
				break
			count+=1


	A_val = tuple( np.array([1]*len(csr_weight.indices)).astype("float32") )
	A_csr = scipy.sparse.csr_matrix((A_val, csr_weight.indices, csr_weight.indptr), shape=csr_weight.shape)


	# return A_data, A_raw_poses
	inps = [A_csr, DenseTensor(x.numpy())]
	sparse_inps_transposed = [ A_csr.tocsc() ]

	if op_type == 'sddmm':
		x1 = torch.rand(A_csr.shape[0], feat_size).half()
		x2 = torch.rand(A_csr.shape[1], feat_size).half()
		inps = [A_csr, DenseTensor(x1.numpy()), DenseTensor(x2.numpy())]


	# op_type = 'spmm'
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [A_csr.shape[0], feat_size, A_csr.shape[1]]
	idx_graph = None

	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed), count




# test the spmm operator from sparse attention: banded sparse pattern, butterfly sparse pattern
# or hybrid sparse pattern, e.g., combining the global sparse and the window sparse, or the one from ``big bird''.
def test_real_op_sparse_attention_csr(pattern, feat_size = 32):


	from utils import create_pixelfly, create_longformer

	block_size = 16
	mb = 256
	nb = 256
	# feat_size = 64
	num_heads = 12
	m = mb * block_size
	n = nb * block_size

	if pattern == "pixelfly":
		csr = create_pixelfly(1, 4096 // 16, fmt="csr", block_size=16)
	elif pattern == "longformer":
		csr = create_longformer(1, 4096 // 16, 256 // 16, fmt="csr", block_size=16)
	else:
		raise KeyError("Sparse pattern {} not recongized.".format(pattern))
	indptr = csr.indptr
	indices = csr.indices
	nnzb = csr.nnz
	np.random.seed(0)
	# data = np.random.rand(num_heads, nnzb, block_size, block_size)
	# x = np.random.rand(num_heads, n, feat_size).astype("float16")
	x = np.random.rand(n, feat_size).astype("float32")


	A_val = tuple( np.array([1]*len(indices)).astype("float32") )
	A_csr = scipy.sparse.csr_matrix((A_val, indices, indptr), shape=csr.shape)


	# return A_data, A_raw_poses
	inps = [A_csr, DenseTensor(x)]
	sparse_inps_transposed = [ A_csr.tocsc() ]


	op_type = 'spmm'
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [A_csr.shape[0], feat_size, A_csr.shape[1]]
	idx_graph = None

	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)




def test_real_op_sparse_attention_bsr(pattern, feat_size = 32):


	from utils import create_pixelfly, create_longformer

	block_size = 16
	mb = 256
	nb = 256
	# feat_size = 64
	num_heads = 12
	m = mb * block_size
	n = nb * block_size

	if pattern == "pixelfly":
		A_block = create_pixelfly(1, mb, fmt="bsr")
	elif pattern == "longformer":
		A_block = create_longformer(1, mb, 256 // block_size, fmt="bsr")
	else:
		raise KeyError("Sparse pattern {} not recongized.".format(pattern))
	indptr = A_block.indptr
	indices = A_block.indices
	nnzb = A_block.nnz
	np.random.seed(0)
	# data = np.random.rand(nnzb, block_size, block_size)
	data = np.ones((nnzb, block_size, block_size), dtype="float32")
	x = np.random.rand(n, feat_size).astype("float32")

	A = scipy.sparse.bsr_matrix((data, indices, indptr), shape=(m, n))
	A_csr = A.tocsr()

	# return A_data, A_raw_poses
	inps = [A_csr, DenseTensor(x)]
	sparse_inps_transposed = [ A_csr.tocsc() ]


	op_type = 'spmm'
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [A_csr.shape[0], feat_size, A_csr.shape[1]]
	idx_graph = None

	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)




def test_multi_level(a, b, c, TC_K, r, feat_size = 32):

	rows = 108*64*16
	# cols = max((108*64-1)*16+TC_K*a, TC_K*a+b*16 )
	cols = (108*64-1)*16+TC_K*a+c
	block_indices = np.concatenate([ np.concatenate([np.arange(TC_K*a)+i*16 for j in range(16)]) for i in range(108*64)])
	block_indptr = np.arange(0, (108*64*16+1)*TC_K*a, TC_K*a)
	# block_data = np.full(len(block_indices), 1, dtype='float32')
	block_data = (np.random.rand(len(block_indices))<r).astype('float32')
	blk_A = scipy.sparse.csr_matrix((block_data, block_indices, block_indptr), shape=(rows, cols))
	blk_A.eliminate_zeros()
	# 

	# diag_indptr = np.arange(0, (108*64*16+1)*b, b)
	diag_indices = np.concatenate([ np.arange(TC_K*a+i*16, TC_K*a+i*16+c) for i in range(108*64)])
	diag_indptr = np.cumsum(np.concatenate([[0]]+[[c] + [0 for j in range(15)] for i in range(108*64)]))
	diag_data = np.full(len(diag_indices), 1, dtype='float32')
	diag_A = scipy.sparse.csr_matrix((diag_data, diag_indices, diag_indptr), shape=(rows, cols))
	# 
	A_csr = blk_A + diag_A
	x = np.random.rand(cols, feat_size).astype("float32")
	# 
	inps = [A_csr, DenseTensor(x)]
	sparse_inps_transposed = [ A_csr.tocsc() ]
	# 
	op_type = 'spmm'
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [A_csr.shape[0], feat_size, A_csr.shape[1]]
	idx_graph = None
	# 
	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)



def test_real_op_big_bird(feat_size = 32):


	m = 4096 
	k = m #
	block_size = 64
	r = 3*block_size # 
	g = 2*block_size # 
	w = 3*block_size # 
	
	random_indptr_A = [r * i for i in range(m+1)]
	random_indices_A = np.concatenate(np.random.randint(k, size=(m, r)))
	random_val_A = [1 for i in range(m*r)]
	random_A = scipy.sparse.csr_matrix((random_val_A, random_indices_A, random_indptr_A), shape=(m, k))

	global_indptr_A1 = [k * i for i in range(g+1)] + [k*g for i in range(m-g)]
	global_indices_A1 = [i for j in range(g) for i in range(k)]
	global_val_A1 = [1 for i in range(g*k)]
	global_A1 = scipy.sparse.csr_matrix((global_val_A1, global_indices_A1, global_indptr_A1), shape=(m, k))

	global_indptr_A2 = [g * i for i in range(m+1)]
	global_indices_A2 = [i for j in range(m) for i in range(g)]
	global_val_A2 = [1 for i in range(g*m)]
	global_A2 = scipy.sparse.csr_matrix((global_val_A2, global_indices_A2, global_indptr_A2), shape=(m, k))


	window_indptr_A = np.concatenate( ([0], np.cumsum([ min(w, k-i) for i in range(m) ])) )
	window_indices_A = np.concatenate([i for j in range(m) for i in range(j, min(j+w, k) ) ])
	window_val_A = [1 for i in range(len(window_indices_A))]
	window_A = scipy.sparse.csr_matrix((window_val_A, window_indices_A, window_indptr_A), shape=(m, k))

	A_csr = random_A + global_A1 + global_A2 + window_A
	x = np.random.rand(k, feat_size).astype("float32")

	# return A_data, A_raw_poses
	inps = [A_csr, DenseTensor(x)]
	sparse_inps_transposed = [ A_csr.tocsc() ]


	op_type = 'spmm'
	# feat_size = 32
	sidxs = [0,1]
	ridxs = [2]
	idx_lens = [A_csr.shape[0], feat_size, A_csr.shape[1]]
	idx_graph = None

	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)





def bench_cusparse(csr: Any, X: torch.Tensor, dtype: str, cuda_i: int):
	# W and X are of the same date type
	dtype_th = None
	if dtype == 'float16':

		dtype_th = torch.float16 # torch.float32 # torch.float16
	elif dtype == 'float32':
		dtype_th = torch.float32
	# 
	W = torch.sparse_csr_tensor(
		csr.indptr, csr.indices, csr.data, size=csr.shape, dtype=dtype_th
	).to(cuda_i)
	X = torch.from_numpy(X)
	print(type(W))
	with torch.no_grad():
		W = W.to(dtype_th).to(cuda_i)
		X = X.to(dtype_th).to(cuda_i)			
		measure = profile_pytorch_ms(lambda: W @ X)
		# 
		print("cusparse time: \t{:.5f}ms".format(measure))
		return measure



def bench_cublas(csr: Any, X: torch.Tensor, dtype: str, cuda_i: int):
	# W and X are of the same date type
	dtype_th = None
	if dtype == 'float16':

		dtype_th = torch.float16 # torch.float32 # torch.float16
	elif dtype == 'float32':
		dtype_th = torch.float32
	# 
	W = torch.from_numpy(csr.toarray())
	X = torch.from_numpy(X)
	with torch.no_grad():
		W = W.to(dtype_th).to(0)
		X = X.to(dtype_th).to(0)

		measure = profile_pytorch_ms(lambda: W @ X)

		print("cublas time: \t{:.5f}ms".format(measure))
		return measure





def bench_dgl(csr: Any, X: torch.Tensor, dtype: str, cuda_i: int):
	import dgl.sparse as dglsp
	# W and X are of the same date type
	dtype_th = None
	if dtype == 'float16':

		dtype_th = torch.float16 # torch.float32 # torch.float16
	elif dtype == 'float32':
		dtype_th = torch.float32
	# 
	print("-----Create from CSR format-----")
	W = dglsp.from_csr(torch.from_numpy(csr.indptr), torch.from_numpy(csr.indices), 
		val=torch.from_numpy(csr.data).to(dtype_th), shape=csr.shape)
	X = torch.from_numpy(X).to(dtype_th)
	with torch.no_grad():
		W = W.to(0)
		X = X.to(0)
		# 
		measure = profile_pytorch_ms(lambda: W @ X)
		# 
		print("dglsp time: \t{:.5f}ms".format(measure))
		return measure



	print("A @ X:")
	print(A @ X)

	print("D @ X:")
	print(D @ X)




  