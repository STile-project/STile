# this file defines the rules used to generate candidate single formats.


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
		


class SparseTensor(object):
	"""
	SparseTensor
	The data structure is like a tree, and it is a multi-level dict, each level is for an index.
	Like the DCSR format.
	"""
	def __init__(self, data, coo_indices, ND = 2):
		super(SparseTensor, self).__init__()
		# the data is in dcsr format
		# the data is in the order of self.row_is, self.col_poses, self.col_is, self.vals
		self.data = tuple([tuple(np.array(i).tolist()) for i in data])
		self.data_type = None
		self.ND = ND
		if ND == 2:
			self.data_type = ['idx', 'ptr', 'idx', 'val']
		else:
			assert False, "We only support 2D sparse tensors now!"
		
		self.coo_indices = tuple([tuple(np.array(i).tolist()) for i in coo_indices]) # raw_poses = set(zip(coo_indices[0], coo_indices[2]))

		self.raw_poses = set(zip(*coo_indices)) # set of tuple of positions
		# self.raw_poses = set(raw_poses) # set of tuple of positions

		# self.data = data
		# self.val_data = val_data
	# 
	def store_to_file(self, log_file):
		to_write = {"data":self.data, "coo_indices":self.coo_indices, "ND":self.ND}
		with open(log_file, 'w') as f:
			f.write(json.dumps(to_write))
	# 
	def is_nnz(self, pos):
		'''
			Check the value of a given position is non-zero or not.
		'''
		# data = self.data
		# for i in pos:
		# 	if i in data:
		# 		data = data[i]
		# 	else:
		# 		return False
		# return True
		# print(pos)
		start, end = 0, len(self.data[0])
		idx_found = None
		dim_i = 0
		for count in range(len(self.data)):
			data_type = self.data_type[count]
			# print(count, data_type)
			if data_type == 'idx':
				# print(start, end)
				# we directly try to find the position of dim_i in the selected range i
				data = self.data[count][start:end]
				idx_found = np.searchsorted(data, pos[dim_i])
				# print(idx_found)
				if (idx_found >= len(data)) or (data[idx_found] != pos[dim_i]):
					return False
				dim_i += 1
			elif data_type == 'ptr':
				start, end = self.data[count][start+idx_found], self.data[count][start+idx_found+1]
				# end = self.data[count][start+idx_found+1]
		return True
	# 
	def num_nnz_givenlist(self, prefix_pos, poses):
		'''
			Check the value of a given position is non-zero or not.
			poses: list of index ids after prefix_pos.
		'''
		# data = self.data
		# for i in pos:
		# 	if i in data:
		# 		data = data[i]
		# 	else:
		# 		return False
		# return True
		# print(pos)
		nnzs = self.get_nnz(prefix_pos)
		return len(nnzs.intersection(poses))
	# 
	def num_nnz_poses(self, poses):
		return len(self.raw_poses.intersection(poses))
	# 
	def nnz_num(self, prefix_pos):
		'''
		Get the number of NNZs in the direct childs of position prefix_pos in the tree.
		'''
		start, end = 0, len(self.data[0])
		idx_found = None
		dim_i = 0
		for count in range(len(self.data)):
			data_type = self.data_type[count]
			if data_type == 'idx':
				# we directly try to find the position of dim_i in the selected range i
				data = self.data[count][start:end]
				idx_found = np.searchsorted(data, prefix_pos[dim_i])
				if (idx_found >= len(data)) or (data[idx_found] != prefix_pos[dim_i]):
					return 0
				dim_i += 1
			elif data_type == 'ptr':
				start, end = self.data[count][start+idx_found], self.data[count][start+idx_found+1]
				# end = self.data[count][start+idx_found+1]
				if dim_i == len(prefix_pos):
					return end-start
		return True
		# data = self.data
		# for i in prefix_pos:
		# 	if i not in data:
		# 		return 0
		# 	data = data[i]
		# return len(data)
	# 
	def get_nnz(self, prefix_pos, return_set=True):
		'''
		Get the NNZs in the direct childs of position prefix_pos in the tree.
		'''
		# data = self.data
		# for i in prefix_pos:
		# 	if i not in data:
		# 		return list()
		# 	data = data[i]
		# return data.keys()
		start, end = 0, len(self.data[0])
		idx_found = None
		dim_i = 0
		for count in range(len(self.data)):
			data_type = self.data_type[count]
			if data_type == 'idx':
				# we directly try to find the position of dim_i in the selected range i
				data = self.data[count][start:end]
				idx_found = np.searchsorted(data, prefix_pos[dim_i])
				if (idx_found >= len(data)) or (data[idx_found] != prefix_pos[dim_i]):
					if return_set:
						return set()
					else:
						return list()
				dim_i += 1
			elif data_type == 'ptr':
				# print(self.data[count][start+idx_found], self.data[count][start+idx_found+1])
				start, end = (self.data[count][start+idx_found], self.data[count][start+idx_found+1])
				# print(start, end)
				# end = self.data[count][start+idx_found+1]
				if dim_i == len(prefix_pos):
					if return_set:
						return set(self.data[count+1][start:end])
					else:
						return self.data[count+1][start:end]
	# 
	def get_vals(self, prefix_pos):
		'''
		Get the values of the NNZs in the direct childs of position prefix_pos in the tree.
		Return list.
		NOTE: require len(prefex_pos) == self.ND-1
		'''
		# data = self.data
		# for i in prefix_pos:
		# 	if i not in data:
		# 		return list()
		# 	data = data[i]
		# return data.keys()
		assert len(prefix_pos) == (self.ND - 1)
		start, end = 0, len(self.data[0])
		idx_found = None
		dim_i = 0
		for count in range(len(self.data)):
			data_type = self.data_type[count]
			if data_type == 'idx':
				# we directly try to find the position of dim_i in the selected range i
				data = self.data[count][start:end]
				idx_found = np.searchsorted(data, prefix_pos[dim_i])
				if (idx_found >= len(data)) or (data[idx_found] != prefix_pos[dim_i]):
					return list()
				dim_i += 1
			elif data_type == 'ptr':
				# print(self.data[count][start+idx_found], self.data[count][start+idx_found+1])
				start, end = (self.data[count][start+idx_found], self.data[count][start+idx_found+1])
				# print(start, end)
				# end = self.data[count][start+idx_found+1]
				if dim_i == len(prefix_pos):
					return self.data[count+2][start:end]
	# 
	def get_val(self, pos):
		'''
			Return the value of a given position.
		'''
		# data = self.data
		# for i in pos:
		# 	if i in data:
		# 		data = data[i]
		# 	else:
		# 		assert False, "This element is zero!"
		# return data
		start, end = 0, len(self.data[0])
		idx_found = None
		dim_i = 0
		for count in range(len(self.data)):
			data_type = self.data_type[count]
			if data_type == 'idx':
				# we directly try to find the position of dim_i in the selected range i
				data = self.data[count][start:end]
				idx_found = np.searchsorted(data, pos[dim_i])
				if (idx_found >= len(data)) or (data[idx_found] != pos[dim_i]):
					return 0
					assert False, "This element is zero!"
				dim_i += 1
			elif data_type == 'ptr':
				start, end = self.data[count][start+idx_found], self.data[count][start+idx_found+1]
				# end = self.data[count][start+idx_found+1]
			elif data_type == 'val':
				return self.data[count][start+idx_found]
	# 










def get_nnz_public(tot_data, data_rngs, data_types, prefix_pos):
	'''
	Get the NNZs in the direct childs of position prefix_pos in the tree.
	此处的tot_data为包含了to_check之类的无关data的array。此处的data_rng也是原始的data_rng而没有做额外的offset处理。
	'''
	start, end = 0, data_rngs[0][1]-data_rngs[0][0]
	idx_found = None
	dim_i = 0
	# print(data_rngs, data_types)
	for count in range(len(data_rngs)):
		data_type = data_types[count]
		if data_type == 'idx':
			# we directly try to find the position of dim_i in the selected range i
			# data = itertools.islice(tot_data, min(data_rngs[count][0]+start, data_rngs[count][1]), min(data_rngs[count][0]+end, data_rngs[count][1]))
			# data = tot_data[data_rngs[count][0]:data_rngs[count][1]][start:end]
			data = tot_data[min(data_rngs[count][0]+start,data_rngs[count][1]):min(data_rngs[count][0]+end,data_rngs[count][1])]
			idx_found = np.searchsorted(data, prefix_pos[dim_i])
			if (idx_found >= len(data)) or (data[idx_found] != prefix_pos[dim_i]):
				return set()
			dim_i += 1
		elif data_type == 'ptr':
			# start = tot_data[data_rngs[count][0]:data_rngs[count][1]][start+idx_found]
			# end = tot_data[data_rngs[count][0]:data_rngs[count][1]][start+idx_found+1]
			# if dim_i == len(prefix_pos):
			# 	return set(tot_data[data_rngs[count+1][0]:data_rngs[count+1][1]][start:end])
			# 
			start, end = tot_data[data_rngs[count][0]+start+idx_found], tot_data[data_rngs[count][0]+start+idx_found+1]
			# end = tot_data[data_rngs[count][0]+start+idx_found+1]
			if dim_i == len(prefix_pos):
				return set(tot_data[min(data_rngs[count+1][0]+start,data_rngs[count+1][1]):min(data_rngs[count+1][0]+end,data_rngs[count+1][1])])


def num_nnz_givenlist_public(data, data_rngs, data_type, prefix_pos, poses):
	'''
		Check the value of a given position is non-zero or not.
		poses: list of index ids after prefix_pos.
	'''
	# data = self.data
	# for i in pos:
	# 	if i in data:
	# 		data = data[i]
	# 	else:
	# 		return False
	# return True
	# print(pos)
	nnzs = get_nnz_public(data, data_rngs, data_type, prefix_pos)
	return len(nnzs.intersection(poses))
# 


def num_nnz_poses_public(raw_poses, poses):
	return len(set([tuple(i) for i in raw_poses]).intersection(poses))










def _summarize_nnz_iter_space(first_end_id, common_params):
	# only one shared memory here
	# print(f"{first_end_id}", flush=True)
	end_id_num, idx_i, op_type, data_type_list, shm_name, shm_shape, shm_dtype, sub_names, sub_shapes = common_params
	# we need first recover the shared data
	# to_check, rngs_list, data_list, raw_poses_list = None, list(), list(), list()
	cur_in, cur_raw = None, None
	# print(shm_name, shm_shape, shm_dtype, sub_names, sub_shapes)
	shm = shared_memory.SharedMemory(name=shm_name)
	tot_data = np.ndarray(shape=shm_shape, dtype=shm_dtype, buffer=shm.buf)
	# 
	to_check_rng, rngs_rng_list, data_rng_list, raw_poses_rng_list = None, list(), list(), list()
	last_end = 0
	raw_poses_shape = None
	ret = list()
	for name, shape in zip(sub_names, sub_shapes):
		if "to_check" in name:
			to_check_rng = [last_end, last_end + math.prod(shape)]
			last_end = last_end + math.prod(shape)
		elif "rng" in name:
			rngs_rng_list.append([last_end, last_end + math.prod(shape)])
			last_end = last_end + math.prod(shape)
		elif "data" in name:
			if (cur_in == None) or (name[:7] != cur_in):
				cur_in = name[:7]
				data_rng_list.append(list())
			data_rng_list[-1].append([last_end, last_end + math.prod(shape)])
			last_end = last_end + math.prod(shape)
		elif "raw" in name:
			raw_poses_rng_list.append([last_end, last_end + math.prod(shape)])
			last_end = last_end + math.prod(shape)
			raw_poses_shape = shape
	# 
	for v in tot_data[to_check_rng[0]:to_check_rng[1]][first_end_id:first_end_id + end_id_num]: 
		if op_type == 'spmm':
			# data, data_type, raw_poses = data_list[0], data_type_list[0], raw_poses_list[0]
			nnz = 0
			i_rngs = [v] if idx_i == 0 else tot_data[rngs_rng_list[0][0]:rngs_rng_list[0][1]]
			j_rngs = [v] if idx_i == 1 else tot_data[rngs_rng_list[1][0]:rngs_rng_list[1][1]]
			k_rngs = [v] if idx_i == 2 else tot_data[rngs_rng_list[2][0]:rngs_rng_list[2][1]]
			# 
			if idx_i == 0:
				# print((data, data_type, (v, ), k_rngs)) ||  data, data_rngs, data_type, prefix_pos, poses
				# print(list(tot_data[data_rng_list[0][0][0]:data_rng_list[0][-1][-1]]), list(data_rng_list[0]), list(data_type_list[0]), (v, ), k_rngs)
				ret.append((v, num_nnz_givenlist_public(tot_data, data_rng_list[0], data_type_list[0], (v, ), k_rngs) * len(j_rngs)))
			elif idx_i == 2:
				# use the csr format in1
				ret.append((v, num_nnz_givenlist_public(tot_data, data_rng_list[0], data_type_list[0], (v, ), i_rngs) * len(j_rngs)))
			else:
				# print((raw_poses, i_rngs,k_rngs))
				ret.append((v, num_nnz_poses_public(np.reshape(tot_data[raw_poses_rng_list[0][0]:raw_poses_rng_list[0][1]], raw_poses_shape), itertools.product(i_rngs,k_rngs)) * len(j_rngs)))
			# print(ret[-1])
	# print("happy")
	return ret









def _summarize_nnz_iter_space_use_file_worker(first_end_id, common_params):
	end_id_num, op_type, idx_i, idx_lens, kernel_tile_Ns, area_pos = common_params
	
	ret = list()
	if op_type == 'spmm':
		# print("begin1", flush=True)
		in1_ref_file = "log_hub/in1.json" # self.inps[0]
		if idx_i == 2:
			in1_ref_file = "log_hub/in1_T.json" # self.sparse_inps_transposed[0]
		data = None
		with open(in1_ref_file, 'r') as f:
			data = json.load(f)

		in1_ref = SparseTensor(data["data"], data["coo_indices"], ND = data["ND"])

		tile_sizes = [math.ceil(v_num / n) for v_num, n in zip(idx_lens, kernel_tile_Ns)]
		idx_values = [range(v_num) for v_num in idx_lens]
		idx_values = [idx_values[i][area_pos[i]*tile_sizes[i] : (area_pos[i]+1)*tile_sizes[i]] \
						for i in range(len(idx_lens))]

		to_check = idx_values[idx_i]

		# print("begin2", flush=True)
		ret = [first_end_id]
		for v in to_check[first_end_id:first_end_id + end_id_num]:
			nnz = 0
			i_rngs = [v] if idx_i == 0 else idx_values[0]
			j_rngs = [v] if idx_i == 1 else idx_values[1]
			k_rngs = [v] if idx_i == 2 else idx_values[2]
			# 
			if idx_i == 0:
				ret.append(in1_ref.num_nnz_givenlist((v, ), k_rngs) * len(j_rngs))
			elif idx_i == 2:
				ret.append(in1_ref.num_nnz_givenlist((v, ), i_rngs) * len(j_rngs))
			else:
				assert False, "do not need to compute nnz for idx j"
				ret.append(in1_ref.num_nnz_poses(itertools.product(i_rngs,k_rngs)) * len(j_rngs)	)
	return ret




summarize_nnz_sharedata_id = 0



def gen_nnz_id_matrix(op):
	'''
	Generate a index sparse matrix for the non-zeros in op's sparse input. The index is of the original 1D compressed nonzero array.
	NOTE: this method will only be called for the original SDDMM operator.
	'''
	if op.op_type == 'sddmm':
		op.nnz_id_matrix = scipy.sparse.csr_matrix((np.arange(1, op.inps[0].nnz+1), op.inps[0].indices, op.inps[0].indptr), shape=op.inps[0].shape)
	else:
		assert False, "Only support SDDMM now."




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
		self.hyb_ori_row_ends = None # 记录每一行在折叠之后生成的新行们的最后一行的行号
		# ------------
		# self.hyb_row_rng = None # 记录1D op的nna平铺折叠之后每一行对应的原行范围
		self.ori_row_ids_1d = None # 记录1D op的nna平铺之后每个non-zero对应的行号，是原始的行号
		self.ori_col_ids_1d = None # 记录1D op的nna平铺之后每个non-zero对应的列号，是原始的行号
		self.row_nums_1d = list() # 记录1D op的每个1D tile对应的行数
		self.col_nums_1d = list() # 记录1D op的每个1D tile对应的列数
		# self.row_matrix_1d = None # 记录1D op中每个non-zero的折叠之前的行号。这个矩阵不会在选择的过程中被更新
		self.nnz_id_matrix = None # 记录每个non-zero在原始matrix的1D压缩数组中的index，用于SDDMM
		self.nnz_update_1D = None # 只用于1D SDDMM，在每次更新了position space update之后更新，存的是每个1D tile的最新nnz
		self.nnz_data_1D = None # 只用于1D SDDMM，在每次更新了position space update 之后更新，存的是每个1D tile的最新的data (包含原来非零但更新之后为零的元素，和现在依然非零的元素)
		self.max_bucket_size = None #目前只用于1D SDDMM，理论上也可以用于ELL SPMM。
		# ------------
		self.k_vals = None # 用来存储为了TC tile对position space进行condense或者k的reorder操作后，每tile_size行对应的原始k vals
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
	#
	# 
	def _prepare_shared_data_summarize_nnz(self, area_i, idx_i, to_check):
		'''
		to_check, rngs_dict, inps.data, inps.raw_poses are to be shared.
		'''
		arrays_to_deal = list()
		shm_names, shm_shapes = list(), list()
		# to_check------------------
		arrays_to_deal.append(np.array(list(to_check), dtype=np.int32))
		shm_names.append("to_check_shared")
		shm_shapes.append(arrays_to_deal[-1].shape)
		# rngs_dict------------------
		for idx_i in self.idxs:
			arrays_to_deal.append(np.array(self.idx_values_list[area_i][idx_i], dtype=np.int32))
			shm_names.append(f"rng{idx_i}_shared")
			shm_shapes.append(arrays_to_deal[-1].shape)
		# inps data and raw_poses------------------
		if self.op_type=='spmm':
			inp_refer = self.inps[0]
			if idx_i == 2:
				# use the csc format in1
				inp_refer = self.sparse_inps_transposed[0]
			# 
			for count, level in enumerate(inp_refer.data[:-1]):
				arrays_to_deal.append(np.array(level, dtype=np.int32))
				shm_names.append(f"in0data{count}_shared")
				shm_shapes.append(arrays_to_deal[-1].shape)
			arrays_to_deal.append(np.array(list(inp_refer.raw_poses), dtype=np.int32))
			shm_names.append("in0raw_poses_shared")
			shm_shapes.append(arrays_to_deal[-1].shape)
		else:
			assert False, "We only support spmm now!"
		# 
		global summarize_nnz_sharedata_id
		shm_name = f"tot_data{summarize_nnz_sharedata_id}" # add "summarize_nnz_sharedata_id" here to avoid "data already exists error"
		summarize_nnz_sharedata_id += 1

		data = np.concatenate(arrays_to_deal, axis=None)
		d_size = data.itemsize * np.prod(data.shape)
		shm = shared_memory.SharedMemory(create=True, size=d_size, name=shm_name)
		# numpy array on shared memory buffer
		dst = np.ndarray(shape=data.shape, dtype=data.dtype, buffer=shm.buf)
		dst[:] = data[:]
		# 
		return shm_name, data.shape, data.dtype, shm_names, shm_shapes
	#
	#
	def _release_shared_data_summarize_nnz(self, shm_names):
		for shm_name in shm_names:
			shm = shared_memory.SharedMemory(name=shm_name)
			shm.close()
			shm.unlink() # free and release the shared memory block
	# 
	def summarize_nnz_iter_space(self, area_i, idx_i, use_file, do_parallel):
		'''
		Compute the NNZs along the given index in the iteration space. The NNZs are from positions with only idx_i fixed.
		暂时按照iteration space来算, 而不是按照input tensor 来算？可能还是得具体算子具体分析？还是得看已有的
		方法中是怎么处理多sparse tensor的情况的, 暂时先只考虑单个sparse tensor的情况。
		This function is only used to generate formats, and we do not consider anything about the final costs here.
		Output:
			No output, save the summary of nnz infor in self.nnz_dict.
		'''
		if self.op_type != 'spmm':
			assert False, f"We do not support {self.op_type} now!"

		idx_values = self.idx_values_list[area_i][idx_i]
		to_check = None
		if (area_i, idx_i) not in self.nnz_dict:
			self.nnz_dict[(area_i, idx_i)] = dict()
			to_check = idx_values
		else:
			# to_check = [v for v in idx_values if v not in self.nnz_dict[(area_i, idx_i)]]
			to_check = set(idx_values).difference(self.nnz_dict[(area_i, idx_i)].keys())
		
		if len(to_check) == 0:
			return
		# focus on this area
		# inps = self.inps
		# if self.op_type == 'spmm':
		# 	# i, j, k are index 0, 1, 2
		# 	# as long as the first input is non-zero, the iteration point is valid
		# 	for v in to_check:
		# 		nnz = 0
		# 		i_rngs = [v] if idx_i == 0 else self.idx_values_list[area_i][0]
		# 		j_rngs = [v] if idx_i == 1 else self.idx_values_list[area_i][1]
		# 		k_rngs = [v] if idx_i == 2 else self.idx_values_list[area_i][2]
		# 		# for i in i_rngs:
		# 		# 	for k in k_rngs:
		# 		# 		if self.inps[0].is_nnz((i,k)):
		# 		# 			nnz += 1
		# 		# nnz = nnz * len(j_rngs)
		# 		# self.nnz_dict[(area_i, idx_i)][v] = nnz
		# 		# 
		# 		if idx_i == 0:
		# 			self.nnz_dict[(area_i, idx_i)][v] = self.inps[0].num_nnz_givenlist((v, ), k_rngs) * len(j_rngs)
		# 		else:
		# 			self.nnz_dict[(area_i, idx_i)][v] = self.inps[0].num_nnz_poses(itertools.product(i_rngs,k_rngs)) * len(j_rngs)
		# use_file = True
		if use_file:
			# 我们已经提前把input的信息存储在文件里面了，此时只需要让多个进程并行从文件中读取结果
			tot_task_num = len(to_check)
			print(f"tot_task_num:{tot_task_num}", flush=True)
			workerNum = 240
			end_id_num = math.ceil(tot_task_num/workerNum)
			first_end_ids = list(range(0, tot_task_num, end_id_num))			

			common_params = None
			if self.op_type == 'spmm':
				common_params = end_id_num, self.op_type, idx_i, self.idx_lens, self.kernel_tile_Ns, self.area_pos[0] # 此处假定idx reorder只在一次kernel tile之后调用

			with multiprocessing.Pool(processes=workerNum) as pool:
				nnz_nums_list = pool.starmap(_summarize_nnz_iter_space_use_file_worker, zip(first_end_ids, itertools.repeat(common_params)))

			print("finish")
			for nnz_nums in nnz_nums_list:
				first_end_id = nnz_nums[0]
				for count, num in enumerate(nnz_nums[1:]):
					self.nnz_dict[(area_i, idx_i)][to_check[first_end_id+count]] = num

			return


		# do_parallel = True #False
		if do_parallel:
			# to_check = to_check[:1]
			tot_task_num = len(to_check)
			print(f"tot_task_num:{tot_task_num}", flush=True)
			workerNum = 240
			end_id_num = math.ceil(tot_task_num/workerNum)
			first_end_ids = list(range(0, tot_task_num, end_id_num))
			
			# common_params = (end_id_num, to_check, self.idx_values_list[area_i], idx_i, self.op_type, self.inps)
			# shm_names, shm_shapes, shm_dtypes = self._prepare_shared_data_summarize_nnz(area_i, idx_i, to_check)
			shm_name, shm_shape, shm_dtype, sub_names, sub_shapes = self._prepare_shared_data_summarize_nnz(area_i, idx_i, to_check)
			common_params = None
			if self.op_type == 'spmm':
				# common_params = (end_id_num, set(to_check), [set(self.idx_values_list[area_i][idx_i]) for idx_i in self.idxs], idx_i, self.op_type, \
				# 	[self.inps[0].data])#, [self.inps[0].data_type], [self.inps[0].raw_poses])
				# common_params = (end_id_num, idx_i, self.op_type, [self.inps[0].data_type], shm_names, shm_shapes, shm_dtypes)
				# 
				# end_id_num, idx_i, op_type, data_type_list, shm_name, shm_shape, shm_dtype, sub_names, sub_shapes
				common_params = (end_id_num, idx_i, self.op_type, [self.inps[0].data_type[:-1]], shm_name, shm_shape, shm_dtype, sub_names, sub_shapes)
			with multiprocessing.Pool(processes=workerNum) as pool:
				nnz_nums_list = pool.starmap(_summarize_nnz_iter_space, zip(first_end_ids, itertools.repeat(common_params)))

			# tot_task_num = len(to_check)
			# print(f"tot_task_num:{tot_task_num}", flush=True)
			# workerNum = 240
			# end_id_num = math.ceil(tot_task_num/workerNum)
			# first_end_ids = list(range(0, tot_task_num, end_id_num))
			# # 
			# to_check_ref, rngs_dict_ref, inps_ref = ray.put(to_check), ray.put(self.idx_values_list[area_i]), ray.put(self.inps)
			# results = [_summarize_nnz_iter_space.remote(start, start+end_id_num, to_check_ref, rngs_dict_ref, inps_ref, area_i, idx_i, self.op_type)\
			# 					for start in first_end_ids]
			# nnz_nums_list = ray.get(results)
			print("finish")
			for nnz_nums in nnz_nums_list:
				for v, num in nnz_nums:
					self.nnz_dict[(area_i, idx_i)][v] = num

			self._release_shared_data_summarize_nnz([shm_name])
		else:
			# compute in sequential
			if self.op_type == 'spmm':
				in1_ref = self.inps[0]
				if idx_i == 2:
					in1_ref = self.sparse_inps_transposed[0]
				for v in to_check:
					nnz = 0
					i_rngs = [v] if idx_i == 0 else self.idx_values_list[area_i][0]
					j_rngs = [v] if idx_i == 1 else self.idx_values_list[area_i][1]
					k_rngs = [v] if idx_i == 2 else self.idx_values_list[area_i][2]
					# 
					if idx_i == 0:
						self.nnz_dict[(area_i, idx_i)][v] = in1_ref.num_nnz_givenlist((v, ), k_rngs) * len(j_rngs)
					elif idx_i == 2:
						self.nnz_dict[(area_i, idx_i)][v] = in1_ref.num_nnz_givenlist((v, ), i_rngs) * len(j_rngs)
					else:
						assert False, "do not need to compute nnz for idx j"
						self.nnz_dict[(area_i, idx_i)][v] = in1_ref.num_nnz_poses(itertools.product(i_rngs,k_rngs)) * len(j_rngs)					




		# if self.op_type != 'spmm':
		# 	assert False, f"We do not support {self.op_type} now!"
		# self._release_shared_data_summarize_nnz([shm_name])
	# 
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
			# 对于spmm而言，index i之前没被tile但是不管有没有reorder过，我们都不会再进行第二次kernel tile，因为没有必要。
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
	
	# 此处有一个假设，即原op.idx_values_list中只有一个area的idx_values
	for idx_values, area_pos in zip(idx_values_list, area_poses):
		# 生成对应的新算子
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

	# op.idx_values_list = idx_values_list
	# # 
	# idx_reordered = op.idx_reordered[0]
	# op.idx_reordered = dict()
	# for area_i in range(len(op.idx_values_list)):
	# 	op.idx_reordered[area_i] = copy.deepcopy(idx_reordered)
	# # 
	# # update op.kernel_tile_Ns
	# op.kernel_tile_Ns = [i * j for i, j in zip(op.kernel_tile_Ns, tile_Ns)]
	# op.kernel_tile_times = op.kernel_tile_times + 1

	# return True








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
	
	
	# 求该op的position space
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

	# 以下是原本的写法
	# op.summarize_nnz_iter_space(area_i, idx_i)
	# op.idx_values_list[area_i][idx_i].sort(key=lambda v: op.nnz_dict[(area_i, idx_i)][v], reverse=True)
	
	# op.idx_reordered[area_i][idx_i] = True





def reorder_for_strided_pattern(op):
	'''
	直接把每一行按照第一个非零元素的位置排序。
	NOTE: 只修改了op.idx_values_list, 并没有修改相应的position space
	'''
	if op.op_type == 'spmm':
		# we only regroup the rows
		# 先找出从哪一行起所有行都为空行
		pos = np.searchsorted(op.inps[0].indptr, op.inps[0].indptr[-1])
		poses = [op.inps[0].indices[i] for i in op.inps[0].indptr[:pos]] + [float('inf') for i in range(op.inps[0].shape[0]-pos)]
		order = np.argsort( poses, kind='stable' )
		op.idx_values_list[0][0] = order
		
		# 我们不认为这个操作是reorder了i，因为并没有把i按照nnz来排序。
		# if not np.array_equal(order, np.sort(order)):
		# 	op.idx_reordered[0][0] = True





















def gen_position_space_for_area(op, area_i):
	'''
	Generate the corresponding tree structure for the area_i, according to the setting in op.loop_order, op.loop_protocals.
	It seems that op.loop_protocals will not be used here.
	INPUT:
		area_i:		int.	The id of the area.
	OUTPUT:
		No output, we directly add the position space into op.position_space.
	要不然, 对于position space来说, 如果某一个dimension是dense的, 那我们就直接忽略这个维度, 不把相关的value存进去。
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




# TODO: 在使用这个函数的时候，要注意prefix应该是reorder之后的？？？？？？？再仔细想想
def get_nnz_from_dict(val_dict, prefix_pos):
	'''
	Get the NNZs in the direct childs of position prefix_pos in the tree.
	重写了这个函数, input里的val_dict目前是scipy.sparse.csr_matrix的数据结构
	'''
	if len(prefix_pos) == 1:
		# 返回的是该行在k轴的有效坐标值
		return val_dict.getrow(prefix_pos[0]).indices
	# for i in prefix_pos:
	# 	if i not in val_dict:
	# 		return list()
	# 	val_dict = val_dict[i]
	# if isinstance(val_dict, dict):
	# 	return val_dict.keys()
	# else:
	# 	return val_dict











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





def condense_for_TC_tile(op, TC_tile_sizes, TC_k_notsorted, reorder_k_by_nnz = True):
	'''
	对给定的sub_op按照TC tile的workload, 对没tile_size行内的所有列进行reorder。
	可以只进行condense, 或者按照nnz的数量进行reorder。
	理论上, 不管i有没有被reorder, 我们都可以利用tensor core来加速, 不过目前暂时还不允许reorder i。
	我们只需要修改每tile_size行对应的 idx_val_list 列表即可。
	INPUT:
		TC_tile_sizes: (tile_i, tile_j, tile_k)
	'''
	# 
	# store the new row ids given max bucket size
	k_axis = 2
	if op.op_type in ['spmm', 'sddmm']:
		row_num = len(op.idx_values_list[0][0])
		blk_i_size = TC_tile_sizes[0][1]
		row_blk_num = math.ceil(row_num / blk_i_size)

		idx_values = op.idx_values_list[0]
		csr = op.inps[0][idx_values[0],:][:, idx_values[k_axis]]

		# csr = op.position_space_update # 此处的position_space_update还没有更新，所以不能使用
		if TC_k_notsorted:
			# 我们不需要对k进行reorder，通常用在block-wise structed sparse pattern上
			# op.k_vals = [idx_values[k_axis] for i in range(row_blk_num)]
			# 此处之前貌似有个bug，因为我们的k_vals应该是基于idx_values_list的reorder的基础上的。
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










# 这个版本我们做了更多的内存预分配，希望能更快一点
# 但是从测试的时间来看，并没有变快。无所谓用slow版本还是这个版本都ok。
def pad_irregular_idx(op, area_i, idx_i, max_bucket_size):
	'''
	padding the irregular length index such that its length is the multiple of 32.
	NOTE: 在此处, 最终得到的position space并不会删除我们遇到的nnz=0的行, hyb_new_rows中也会为nnz为0的行记录相应的信息。
	之前的写法是空行不存, 现在的写法是空行依然会存, 本质上都可以运行, 但是结果可能不用, 因为和tile size的candidate生成有关。
	'''
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
		
		# if op.hyb_short_begin_row == None:
		# 	# 没有行的长度比max bucket size 短
		# 	op.hyb_short_begin_row = len(op.hyb_new_rows[0])
		op.hyb_getnnz_minus = -op.inp_getnnz[0][op.hyb_short_begin_row:]

		op.hyb_ori_row_ends = np.array(op.hyb_ori_row_ends)

		op.position_space_update = op.position_space[area_i].copy()






def get_1D_position_space(op, max_bucket_size):
	# get (1) position_space_update, (2) the ori-row-range of each 'new' row.

	assert op.op_type == 'sddmm'
	# 虽然现在默认不对1D op做reorder，但是要为之后可能得reorder做准备
	csr = op.inps[0][op.idx_values_list[0][0],:][:,op.idx_values_list[0][2]]

	# 不再在position space对csr进行折叠
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

	# ========================================================================










# ------------------------------------------------------------------------------------

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
    







# 这个版本根据 max_bucket_size 自动 pruning
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

				# FOR DEBUG：假设我们会加上thread num的控制
				if thread_num<THREAD_num:
					continue

				# 我们现在直接加上最严格的限制,即要求ELL tile的size都一样
				if tile_sizes[0][1]*tile_sizes[2][1]!=WORKLOAD:
					continue

				ret.append(tile_sizes)
			else:
				thread_num = tile_sizes[0][2]*tile_sizes[1][2]
				# if thread_num%128 != 0:
				# 	continue
				# FOR DEBUG: 假设放松thread num的限制
				if thread_num%32 != 0:
					continue


				if thread_num>1024:
					continue

				# # FOR DEBUG: 换成和sparsetir完全一样的限制条件
				# if (tile_sizes[0][2]!=8) or (tile_sizes[1][2]!=32):
				# 	continue
				# FOR DEBUG: 换成和线程数为64
				# if (tile_sizes[0][2]!=2) or (tile_sizes[1][2]!=32):
				# 	continue
				if (tile_sizes[0][2]!=1) or (tile_sizes[1][2]!=32):
					continue
				# FOR DEBUG: 测试要求所有tile的线程数量都相同
				# if (tile_sizes[0][2]*tile_sizes[1][2]!=256):
				# 	continue

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
	# try to reduce possible TC tile options
	# ret = [((None, 16), (None, 32), (None, 16*i)) for i in range(7, 0, -1)]  #[((None, 16), (None, 32), (None, 16*i)) for i in range(160//16, 0, -1)]  # ((None, 16), (None, 32), (None, 144))

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
			# 如果我们没有对这个TC op的k按TC row进行reorder，那就重新算blk_num,并且更新blk_ks
			op.blk_nums[tile_sizes] = np.asarray([ np.count_nonzero(
				np.add.reduceat( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0), np.arange(0, csr.shape[1], blk_k_size))
				) for i in range(row_blk_num)])
			op.blk_ks[tile_sizes] = [ np.nonzero(
				np.add.reduceat( csr[i*blk_i_size:(i+1)*blk_i_size, :].getnnz(axis=0), np.arange(0, csr.shape[1], blk_k_size))
				)[0] for i in range(row_blk_num)]





def gen_candidate_TC_formats(op, TC_k_notsorted, reorder_k_by_nnz = True):
	'''
	Generate candidate TC ops, called by gen_candidate_formats.
	'''
	# 1. before we do kernel tile, we need to reorder the rows first to deal with the possible strided sparse pattern.
	last_cand_ops = [op]
	cand_ops = list()
	group_sizes = [2**i for i in range(int(math.log(32*32, 2)))]
	# =====================================
	# 其实step 1 可以跳过
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
		# FOR DEBUG 看看不考虑reorder效果如何===============
		# if sub_op.idx_reordered[0][0]: # sub_op和op不一样了
		# 暂时先直接删掉这种reorder，因为TC tile也是block wise的。要更彻底的reorder，可以考虑在一开始做一些类似similarity based的reorder。
		# if not np.array_equal(sub_op.idx_values_list[0][0], op.idx_values_list[0][0]): # sub_op和op不一样了
		# 	cand_ops.append(sub_op)
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


	# 3. reorder index again
	# 此处选择reorder的index的可能性可能很多，因为还要考虑到reorder多个index的情况，是不是就安装常见的implementation中的reorder的情况来？还是说我们就默认要么全部都reorder，要么就全部都不reorder？
	# 先暂时选成要么就全都reorder,要么就全部不reorder.
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
				# 对于只kernel tile了i轴的情况，我们不同时reorder i和k，而是先reorder k，然后tile k，再reorder i。
				if (idx_i == 0) or (sub_op.kernel_tile_Ns[idx_i] == 1):
					reorder_idx(sub_op, area_i, idx_i)
		cand_ops.append(sub_op)
		nnz_dict_ref = sub_op.nnz_dict

		# FOR DEBUG 暂时删掉不reorder任何轴的op，因为我们默认一定会reorder i=============
		if (op.op_type in ['spmm', 'sddmm']) and (op.kernel_tile_Ns[0] == 1) and (op.kernel_tile_Ns[2] == 1):
			# 这个op是为了tensor core template准备的
			cand_ops.append(op)
		# ========================================================================

		# 为每个op再加一个只reorder index i的
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
	# 3.5 删掉reorder了k但是不在k上tile的op
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
			# 如果同时reorder了i和k，那说明在第二次kernel tiling之后，每个新得到的sub_op中的i的顺序不对了，需要再次reorder。
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
	# 获取tile_i不同的TC_tile_sizes
	ori_TC_tile_sizes_list = get_TC_tile_sizes()
	TC_tile_sizes_list = list()
	TC_is = list()
	for TC_tile_sizes in ori_TC_tile_sizes_list:
		if TC_tile_sizes[0][1] not in TC_is:
			TC_tile_sizes_list.append(TC_tile_sizes)
			TC_is.append(TC_tile_sizes[0][1])

	for sub_op in last_cand_ops:
		# if (sum(sub_op.kernel_tile_Ns) == len(sub_op.kernel_tile_Ns)) and (True not in sub_op.idx_reordered[0]):
		# 	# 只用于tensor core based template，即: 没有被kernel tile过，也没有被reorder过
		# FOR DEBUG===========================================================================================
		if (sum(sub_op.kernel_tile_Ns) == len(sub_op.kernel_tile_Ns)): # and (True not in sub_op.idx_reordered[0]):
			# 只用于tensor core based template，即: 没有被kernel tile过，但是我们会reorder i，是否要把同时也考虑没有reorder i的？
		# 	# 不需要也考虑没有reorder i的，因为我们在reorder的时候做的是stable sort。
			for TC_tile_sizes in TC_tile_sizes_list: # get_TC_tile_sizes():
				new_sub_op = copy.copy(sub_op)
				new_sub_op.loop_protocals = copy.deepcopy(sub_op.loop_protocals)
				new_sub_op.inp_protocals = copy.deepcopy(sub_op.inp_protocals)
				new_sub_op.k_vals = list()
				# change_to_dense(new_sub_op, new_sub_op.this_area_i)
				change_to_dense(new_sub_op, new_sub_op.this_area_i, TC_tile_sizes, TC_k_notsorted, reorder_k_by_nnz = reorder_k_by_nnz)
				cand_ops.append(new_sub_op)
				print(sub_op.get_key())

			# 我们还是要为这个sub_op保留变成ELL 或者CSR 类型的可能
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





def gen_candidate_formats_BSPMM(op, max_bucket_size, kernel_tile_size_options, TC_k_notsorted,
	reorder_k_by_nnz = True, op_id_start = 0, gen_TC_formats = True):
	'''
	Generate candidate formats, including kernel tiling and format decomposition.
	'''
	# index reorder, kernel tile, loop reorder, loop fusion, compressed to uncompressed
	'''	kernel tile 的tile size可以搞成一个annotation然后tune。
	index reorder 和 kernel tile 的应用顺序是可以互换的但是只会用一次？
	kernel tile 和 index reorder 的顺序是可以交换的
	所以整体变成index reorder -> kernel tile -> index reorder ==> 现在变成了 kernel_tile -> index reorder
	'''
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
		# FOR DEBUG 看看不考虑reorder效果如何===============
		# if sub_op.idx_reordered[0][0]: # sub_op和op不一样了
		# 暂时先直接删掉这种reorder，因为TC tile也是block wise的。要更彻底的reorder，可以考虑在一开始做一些类似similarity based的reorder。
		# if not np.array_equal(sub_op.idx_values_list[0][0], op.idx_values_list[0][0]): # sub_op和op不一样了
		# 	cand_ops.append(sub_op)
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


	# 因为我们打算删掉以下kernel tile方案，所以在此处先暂时准备好之后的idx_reorder所需要的ref_nnz_dicts
	# for op in cand_ops:
	# 	if op.kernel_tile_Ns == max_kernel_tile_Ns:
	# 		for idx_i in [0,2]:
	# 			op.summarize_nnz_iter_space(0, idx_i, *run_params["summarize_nnz_iter_space"])
	# 			ref_nnz_dicts[op.area_pos[0]] = op.nnz_dict
			
	# 现在删掉一些我们认为不需要的tiling方案
	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		# 对于spmm，我们要求在reorder之前最多只能进行一维的kernel tile
		if (op.kernel_tile_Ns[0] > 1) and (op.kernel_tile_Ns[2] > 1):
			continue
		cand_ops.append(op)


	# 3. reorder index again
	# 此处选择reorder的index的可能性可能很多，因为还要考虑到reorder多个index的情况，是不是就安装常见的implementation中的reorder的情况来？还是说我们就默认要么全部都reorder，要么就全部都不reorder？
	# 先暂时选成要么就全都reorder,要么就全部不reorder.
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
				# if (idx_i == 0) or (sub_op.kernel_tile_Ns[idx_i] == 1):
					# reorder_idx(sub_op, area_i, idx_i)
				# 对于只kernel tile了i轴的情况，我们不同时reorder i和k，而是先reorder k，然后tile k，再reorder i。
				if (idx_i == 0) or (sub_op.kernel_tile_Ns[idx_i] == 1):
					reorder_idx(sub_op, area_i, idx_i)
		cand_ops.append(sub_op)
		nnz_dict_ref = sub_op.nnz_dict

		# FOR DEBUG 暂时删掉不reorder任何轴的op，因为我们默认一定会reorder i=============
		if (op.kernel_tile_Ns[0] == 1) and (op.kernel_tile_Ns[2] == 1):
			# 这个op是为了tensor core template准备的
			cand_ops.append(op)
		# ========================================================================

		# 为每个op再加一个只reorder index i的
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
	# 先暂时删掉 “do kernel level tiling again”，因为好像没什么用，还妨碍了我们在col partition的同时保留TC op
	# 算了，还是保留吧，改成单独用一个函数生成TC op
	# 
	# 3.5 do kernel level tiling again
	last_cand_ops = cand_ops
	cand_ops = list()
	for op in last_cand_ops:
		# kernel_tile_sizes = itertools.product(*(loop_tile_num, [1], loop_tile_num))
		print(op.get_key())
		for tile_Ns in kernel_tile_sizes:
			print(tile_Ns)
			if len(tile_Ns) == sum(tile_Ns):
				continue
			# 把后面的“删掉reorder了k但是不在k上tile的op”往前挪
			if tile_Ns[2] == 1 and op.idx_reordered[0][2]:
				continue
			# 
			to_add = kernel_tile(op, tile_Ns)
			# print(len(to_add))
			cand_ops = cand_ops + to_add
		
		# 把后面的“删掉reorder了k但是不在k上tile的op”往前挪
		if op.idx_reordered[0][2]:
			# 因为 这个op本身也相当于reorder了k但是不在此基础上切分。
			continue
		cand_ops.append(op)
	print(f"\n3.5. kernel tile again----cand_ops size: {len(cand_ops)}")	
	# 
	# 3.5 删掉reorder了k但是不在k上tile的op
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
			# 如果同时reorder了i和k，那说明在第二次kernel tiling之后，每个新得到的sub_op中的i的顺序不对了，需要再次reorder。
			reorder_idx(op, 0, 0)
	print(f"\n3.7. index reorder again----cand_ops size: {len(cand_ops)}")	
	# 
	# 4. apply algorithms from the asymptotic cost model
	if op.op_type == 'spmm':
		for sub_op in cand_ops:
			SPMM_Gustavson_product(sub_op, sub_op.this_area_i) # use this order by default
	print(f"\n4. algorithm----cand_ops size: {len(cand_ops)}")
	# 

	# 我们会通过 gen_candidate_TC_formats(op, reorder_k_by_nnz = True) 来生成TC op，而不是在同一个流程里生成ELL op和TC op
	# # 5. change compressed to uncompressed
	# # dense kernels can only be used when loops are not compressed
	# # we only consider dense kernels for an uncompressed tile?
	# last_cand_ops = cand_ops
	# cand_ops = list()
	# # 获取tile_i不同的TC_tile_sizes
	# ori_TC_tile_sizes_list = get_TC_tile_sizes()
	# TC_tile_sizes_list = list()
	# TC_is = list()
	# for TC_tile_sizes in ori_TC_tile_sizes_list:
	# 	if TC_tile_sizes[0][1] not in TC_is:
	# 		TC_tile_sizes_list.append(TC_tile_sizes)
	# 		TC_is.append(TC_tile_sizes[0][1])

	# for sub_op in last_cand_ops:
	# 	if not gen_TC_formats:
	# 		# 在生成next level的op的时候，为了节约时间，我们可能直接不生成TC ops
	# 		cand_ops.append(sub_op)
	# 		continue

	# 	# if (sum(sub_op.kernel_tile_Ns) == len(sub_op.kernel_tile_Ns)) and (True not in sub_op.idx_reordered[0]):
	# 	# 	# 只用于tensor core based template，即: 没有被kernel tile过，也没有被reorder过
	# 	# FOR DEBUG===========================================================================================
	# 	if (sum(sub_op.kernel_tile_Ns) == len(sub_op.kernel_tile_Ns)): # and (True not in sub_op.idx_reordered[0]):
	# 		# 只用于tensor core based template，即: 没有被kernel tile过，但是我们会reorder i，是否要把同时也考虑没有reorder i的？
	# 	# 	# 不需要也考虑没有reorder i的，因为我们在reorder的时候做的是stable sort。
	# 		for TC_tile_sizes in TC_tile_sizes_list: # get_TC_tile_sizes():
	# 			new_sub_op = copy.copy(sub_op)
	# 			new_sub_op.loop_protocals = copy.deepcopy(sub_op.loop_protocals)
	# 			new_sub_op.inp_protocals = copy.deepcopy(sub_op.inp_protocals)
	# 			new_sub_op.k_vals = list()
	# 			# change_to_dense(new_sub_op, new_sub_op.this_area_i)
	# 			change_to_dense(new_sub_op, new_sub_op.this_area_i, TC_tile_sizes, reorder_k_by_nnz = reorder_k_by_nnz)
	# 			cand_ops.append(new_sub_op)
	# 			print(sub_op.get_key())

	# 		# 我们还是要为这个sub_op保留变成ELL 或者CSR 类型的可能
	# 		cand_ops.append(sub_op)
	# 	else:
	# 		cand_ops.append(sub_op)
	# 	# print(new_sub_op.loop_protocals[new_sub_op.this_area_i], new_sub_op.idx_reordered[new_sub_op.this_area_i], sub_op.loop_protocals[sub_op.this_area_i], sub_op.idx_reordered[sub_op.this_area_i])
	# print(f"\n5. compressed to uncompressed----cand_ops size: {len(cand_ops)}")
	
	# now we have generated all the possible formats for dependence level 1
	'''接下来需要判断哪些tile不适合现有的format, 从而将其组织起来, 得到dependence level 2 的format。
				TODO: 还需要加一个转换, 即例如对于SPMM那样的2D input, 是否需要对irregular shape进行padding,
				使其长度变成32的倍数---------------------------------------------------------------------'''
	# 6. padding the irregular axis to regular length, 暂时先公用sparse template，但是事实上针对这种情况，可以有别的scheduling（利用了shared memory的
	# max_bucket_size = 256
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
			# if sub_op.idx_reordered[0][0]:
			# FOR DEBUG ===================我们默认要支持tensor core tile的reorder i了。
			# if not sub_op.idx_reordered[0][0]:
			# 	continue
		elif sub_op.loop_protocals[0] == 'uuu':
			# 先判断这个sub_op是否可能是redundant的
			if list(sub_op.idx_values_list[0][0]) in TC_idx_is:
				continue

			TC_idx_is.append(list(sub_op.idx_values_list[0][0]))

			# 要重新修正idx_reordered,因为在kernel tile之前做了reorder，但是没有记录
			sub_op.idx_reordered = copy.deepcopy(sub_op.idx_reordered)
			if not np.array_equal(sub_op.idx_values_list[0][0], np.sort(sub_op.idx_values_list[0][0])):
				sub_op.idx_reordered[0][0] = True
			else:
				sub_op.idx_reordered[0][0] = False

			# 先暂时存到TC_ops里面，之后再从里面挑一个最好的
			TC_ops.append(sub_op)
			continue

		print("SUCCESS")
		# sub_op.position_space = dict()
		# compute the complete idx values for the sub_op as metadata，不需要这个值，因为complete_idx_values应该是由tile决定
		# comp_complete_idx_values(sub_op)

		sub_op.op_id = op_id_start
		op_id_start += 1
		ret_cand_ops.append(sub_op)



	# 最后只挑选一个密度最高的TC tensor
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

	# 我们只需要生成一个给1D tile使用的op
	# 这个op需要哪些数据结构呢？感觉也不需要啥特殊的结构，不需要任何reorder，tiling啥的
	op_1Dtile = copy.copy(op)
	SDDMM_Gustavson_product(op_1Dtile, 0)
	op_1Dtile.max_bucket_size = max_bucket_size
	# 还需要为这个1D op 生成折叠之后的position space
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
			# 先判断这个sub_op是否可能是redundant的
			if list(sub_op.idx_values_list[0][0]) in TC_idx_is:
				continue

			TC_idx_is.append(list(sub_op.idx_values_list[0][0]))

			# 要重新修正idx_reordered,因为在kernel tile之前做了reorder，但是没有记录
			sub_op.idx_reordered = copy.deepcopy(sub_op.idx_reordered)
			if not np.array_equal(sub_op.idx_values_list[0][0], np.sort(sub_op.idx_values_list[0][0])):
				sub_op.idx_reordered[0][0] = True
			else:
				sub_op.idx_reordered[0][0] = False

			# 先暂时存到TC_ops里面，之后再从里面挑一个最好的
			TC_ops.append(sub_op)
			continue

		print("SUCCESS")

		sub_op.op_id = op_id_start
		op_id_start += 1
		ret_cand_ops.append(sub_op)


	# 最后只挑选一个密度最高的TC tensor
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
	'''
	Generate candidate formats, including kernel tiling and format decomposition.
	'''
	# index reorder, kernel tile, loop reorder, loop fusion, compressed to uncompressed
	'''	kernel tile 的tile size可以搞成一个annotation然后tune。
	index reorder 和 kernel tile 的应用顺序是可以互换的但是只会用一次？
	kernel tile 和 index reorder 的顺序是可以交换的
	所以整体变成index reorder -> kernel tile -> index reorder ==> 现在变成了 kernel_tile -> index reorder
	'''
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
		

		# print(sub_op.loop_protocals[area_i], sub_op.idx_reordered[area_i])
		# if sub_op.loop_protocals[area_i] != 'uuu' or sub_op.idx_reordered[area_i][0]:
		# 	continue

		# ------------------------------------------------------
		if sub_op.loop_protocals[area_i][2] == 'c':
			tile_sizes_idx = [get_tiles(len(sub_op.idx_values_list[area_i][idx_i]), tile_level_num, (None,)) \
							for idx_i in [0,1]] + [[(1, len(sub_op.idx_values_list[area_i][2]))]]
		elif sub_op.loop_protocals[area_i][2] == 'p':
			tot_lens = [math.ceil(len(sub_op.hyb_new_rows[0])/128)*128, len(sub_op.idx_values_list[area_i][1])]
			tile_sizes_idx = [get_tiles(tot_lens[idx_i], tile_level_num, (None,)) \
							for idx_i in [0,1]] + [[(1, 2**tmp) for tmp in range(int(math.log(max_bucket_size, 2)) + 1) ]]
			
			# FOR DEBUG 看看完全和sparsetir一样的参数效果会怎么样
			# tile_sizes_idx[0] = [(None, 32*8//(2**tmp)) for tmp in range(int(math.log(max_bucket_size, 2)) + 1)]
			tile_sizes_idx[0] = [(None, 2**tmp) for tmp in range(int(math.log(len(sub_op.hyb_new_rows[0]), 2)) + 1)]

		elif sub_op.loop_protocals[area_i] == 'uuu':
			# tile_sizes_idx = [((None, 8), (None, 32), (None, 16)), ((None, 16), (None, 16), (None, 16))]
			# tile_sizes_idx = [((None, 16), (None, 16), (None, 16))]
			# tile_sizes_idx = [((None, 32), (None, 32), (None, 768))]
			# 这些tile sizes的设置规则就是把wmma的tile size放大四倍。GPU还支持32x8x16的TC tile，但是似乎我们这里不太需要？之后也可以试试
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
			tile_sizes_idx = [(max_bucket_size,)] # 但是这个地方好像用不到 TODO SDDMM
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
			# we do not tile on index k, but the computation below still deals with the case which tiles on k
			# there is no padding in this template
			# 我们目前暂时不支持tile之间在k轴也有workload的划分！！！！TODO，似乎没这个必要？？因为ELL中有折叠过长的行的操作

			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]

		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			
			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(op.hyb_new_rows[0]))),:]


		elif template_str == "TensorCore_template":
			# print("This recored is about tensor cores, we do not need to estimate cost.")
			# 我们依然需要设置tile.uncovered_position_space
			# tensor core based的template的loop protocol一定是 'uuu'

			# tile.uncovered_position_space = op.position_space[area_i]\
			# 	[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]\
			# 	[:, range(tile_rngs[2][0], min(tile_rngs[2][1], len(idx_values[2])))]

			# 我们需要考虑condense之类的模板每个区域对k的reorder不一样
			ks = tile.op.k_vals[tile_pos[0]][tile_rngs[2][0] : min(tile_rngs[2][1], len(idx_values[2]))]
			tile.uncovered_position_space = op.position_space[area_i]\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]\
				[:, ks]
	elif op.op_type == 'sddmm':
		template_str = get_template_str(op)
		if template_str == '1D_sddmm':
			# tile.uncovered_position_space = op.position_space[area_i][tile_pos[0]]
			# NOTE: 对于1D tile，其uncovered position space不是csr matrix，而是1维数组
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
			# 我们目前暂时不支持tile之间在k轴也有workload的划分！！！！TODO，似乎没这个必要？？因为ELL中有折叠过长的行的操作

			tile.uncovered_position_space = op.position_space_update\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]

		elif template_str == "sparse_template_ell":
			# the inputs are padded if necessary (on both index i, j, k)
			
			tile.uncovered_position_space = op.position_space_update\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(op.hyb_new_rows[0]))),:]


		elif template_str == "TensorCore_template":
			# print("This recored is about tensor cores, we do not need to estimate cost.")
			# 我们依然需要设置tile.uncovered_position_space
			# tensor core based的template的loop protocol一定是 'uuu'

			# tile.uncovered_position_space = op.position_space_update\
			# 	[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]\
			# 	[:, range(tile_rngs[2][0], min(tile_rngs[2][1], len(idx_values[2])))]

			# 我们需要考虑condense之类的模板每个区域对k的reorder不一样
			ks = tile.op.k_vals[tile_pos[0]][tile_rngs[2][0] : min(tile_rngs[2][1], len(idx_values[2]))]
			tile.uncovered_position_space = op.position_space_update\
				[range(tile_rngs[0][0], min(tile_rngs[0][1], len(idx_values[0]))),:]\
				[:, ks]			
	elif op.op_type == 'sddmm':
		template_str = get_template_str(op)
		if template_str == '1D_sddmm':
			# TODO SDDMM: 此处暂时还没想要需要存什么样的数据结构
			# tile.uncovered_position_space = op.position_space_update[tile_pos[0]]
			# NOTE: 对于1D tile而言，其uncovered position space不再是csr matrix，而只是单纯的1维数组
			tile.uncovered_position_space = op.nnz_data_1D[ tile_pos[0]*tile_sizes[0]:(tile_pos[0]+1)*tile_sizes[0] ]

			# indices = op.position_space_update.indices[tile_pos[0]*tile_sizes[0]:(tile_pos[0]+1)*tile_sizes[0]]
			# row0 = np.searchsorted(op.position_space_update.indptr, tile_pos[0]*tile_sizes[0], side='right') - 1
			# row1 = np.searchsorted(op.position_space_update.indptr, min((tile_pos[0]+1)*tile_sizes[0], op.position_space_update.nnz)-1, side='right') - 1
			# indptr = [tile_pos[0]*tile_sizes[0]] + op.position_space_update.indptr[row0+1 : row1] + [min((tile_pos[0]+1)*tile_sizes[0], op.position_space_update.nnz)]
			# indptr = indptr - tile_pos[0]*tile_sizes[0]
			# tile.uncovered_position_space = scipy.sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=( len(indptr)-1, op.position_space_update.shape[1] ))
		elif template_str == 'TC_sddmm':
			ks = tile.op.k_vals[tile_pos[0]][tile.tile_k_rng[0] : tile.tile_k_rng[1]+1]
			tile.uncovered_position_space = op.position_space_update\
				[range(tile.tile_i_rng[0], tile.tile_i_rng[1]+1),:]\
				[:, ks]


def comp_complete_idx_values_given_tile(tile_op, tile_idx_values):
	'''
	将这个sub_op的idx_values补充完整。
	'''
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

		# 对于ELL，先要将其cover的范围恢复成折叠行之前的样子
		unfold_covered_csr = None
		if get_template_str(op) == 'sparse_template_ell':
			# 将ELL的covered_csr转化为没有折叠过长行时的状态 data不变，indices不变，变的只有indptr
			tile_sizes = selected_tile.tile_sizes
			# tile_i_rng = math.prod(tile_sizes[0][1:]) * tile_pos[0], \
			# 	min(math.prod(tile_sizes[0][1:]) * (tile_pos[0]+1), len(op.hyb_new_rows[0])) - 1

			tile_i_rng = selected_tile.tile_i_rng

			row_rng_before_fold = [op.hyb_new_rows[0][i] for i in tile_i_rng]
			
			row_end_offset = op.hyb_ori_row_ends[ row_rng_before_fold[0]:row_rng_before_fold[1]+1 ] - tile_i_rng[0]

			# tile中包含的原始行各自在tile中的最末新行的行号
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

		# 计算resize之后的矩阵的i和k对应的原始op的input矩阵中的i和k
		# 我们并不考虑csr和ELL在k轴的tile workload划分
		tile_idx_values[2] = op.idx_values_list[0][2] # [selected_tile.tile_k_rng[0]:selected_tile.tile_k_rng[1]+1]
		if get_template_str(op) == 'TensorCore_template':
			tile_idx_values[2] = op.idx_values_list[0][2][\
				op.k_vals[selected_tile.tile_pos[0]][selected_tile.tile_k_rng[0]:selected_tile.tile_k_rng[1]+1] ]

		complete_idx_values = comp_complete_idx_values_given_tile(op, tile_idx_values)

		ind_i = np.argsort(complete_idx_values[0])
		ind_k = np.argsort(complete_idx_values[2])

		# 把unfold_covered_csr还原回原始op的in1的矩阵坐标
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
	# require the ori position space of self.op is set
	# 假定tile之间在op的k轴workload上没有分工
	# 此处计算的nnz是根据op的原始position space来计算的，没有考虑有别的tile被选择了
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
	def __init__(self, op, tile_sizes, tile_pos, params, not_init_nnz=False):
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

		# gen_position_space_for_tile(self)
		# # initialize self.nnz
		# if op.op_type == 'spmm':
		# 	# # self.nnz = len(self.uncovered_position_space['j']) * sum([len(v) for v in self.uncovered_position_space['in1'].values()])
		# 	# self.nnz = sum([len(v) for v in self.uncovered_position_space['in1'].values()])
		# 	# self.nnz_uncovered = self.nnz * len(self.uncovered_position_space['j'])

		# 	# 此处uncovered_position_space 被换成了scipy sparse csr_matrix
		# 	self.nnz = self.uncovered_position_space['in1'].getnnz()
		# 	self.nnz_uncovered = self.nnz * len(self.uncovered_position_space['j'])

		# 在初始化的时候不再计算tile的uncovered_position_space
		# initialize self.nnz
		if (op.op_type == 'spmm') or (get_template_str(op) == 'TC_sddmm'):
			# 此处uncovered_position_space 被换成了scipy sparse csr_matrix
			self.tile_i_rng = math.prod(tile_sizes[0][1:]) * tile_pos[0], \
					min(math.prod(tile_sizes[0][1:]) * (tile_pos[0]+1), op.position_space_update.shape[0] ) - 1

			self.tile_k_rng = math.prod(tile_sizes[2][1:]) * tile_pos[2], \
					min(math.prod(tile_sizes[2][1:]) * (tile_pos[2]+1), op.position_space_update.shape[1] ) - 1

			self.tile_k_rng = int(self.tile_k_rng[0]), int(self.tile_k_rng[1])

			set_j_len(self) 

			# <jingzhi>@revision: can choose to not generate position space for nnz initial
			if not_init_nnz:
				return

			init_nnz(self)
			self.nnz_uncovered = self.nnz * self.j_num
		elif get_template_str(op) == '1D_sddmm':
			self.tile_i_rng = None #tile_pos[0], tile_pos[0]
			self.tile_k_rng = None # 0, tile_sizes[0]-1
			set_j_len(self)

			# <jingzhi>@revision: can choose to not generate position space for nnz initial
			if not_init_nnz:
				return

			init_nnz(self)
			self.nnz_uncovered = self.nnz * self.j_num			

	def get_k_vals(self):
		# 计算 这个tile的 k_vals；目前只对TC tile这样做
		if get_template_str(self.op) == 'TensorCore_template':
			self.k_vals = self.op.idx_values_list[0][2][ self.tile_k_rng[0]:self.tile_k_rng[1]+1 ]

	def check_atomic(self, ori_op, max_bucket_size):
		# ori_op是没有经过任何tile，没有经过任何reorder的op
		# NOTE： 这个函数不会被用于TC tile
		# 这个函数会直接设置 is_atomic_tile
		# 首先要得到这个tile涉及的i
		# 这里好像有点问题，因为对于last bucket的ELL tile，不用看nnz是否完整覆盖，就应该设成atomic的。
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
		# require the updated position space of self.op is set
		# 假定tile之间在op的k轴workload上没有分工
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
			# 一定是出了什么错
			assert False, f"the cost of this tile is {self.cost} < 0"
			self.cost = float('inf')

		if self.nnz_uncovered == 0:
			self.avg_cost = float('inf')
		else:
			self.avg_cost = self.cost / self.nnz_uncovered
	
	def set_pred_avg_cost(self):
		if self.pred_cost == None:
			# 应该是TC tile
			assert 'mma_shape_str' in self.params, "Should be a TC tile"
			return
		
		# 因为加入了withdraw的支持，所以pred cost可能是负数，也可能是inf级别的，nnz_uncovered也可能暂时为0
		self.pred_avg_cost = self.pred_cost / np.float32(self.nnz_uncovered)

		# 以下为原有写法
		# if self.pred_cost < 0:
		# 	# 一定是出了什么错
		# 	assert False, f"the pred_cost of this tile is {self.pred_cost} < 0"
		# 	self.pred_cost = float('inf') 

		# if self.nnz_uncovered == 0:
		# 	self.pred_avg_cost = float('inf')
		# else:
		# 	self.pred_avg_cost = self.pred_cost / self.nnz_uncovered

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
			# 加下来是对ELL类的format进行schedule

			# v2idx = get_val_2_ind(idx_values[2])

			# check the first row (its #nnz should be the largest in the tile) of the sparse tensor in the tile range has #nnz < tile_sizes[2] (i.e., tile_sizes.k)-----------
			# or this tile is in the last bucket---------------
			
			new_row_i = tile_pos[0]*math.prod(tile_sizes[0][1:])
			if new_row_i < sub_op.hyb_short_begin_row:
				# should be the last bucket
				if (not params['last_bucket']):
					return False
			else:
				ori_row_i = sub_op.hyb_new_rows[0][new_row_i] # idx_values[0][new_row_i]
				indices = get_nnz_from_dict(sub_op.position_space[area_i], [ori_row_i])

				# 如果需要pad太多，或者超出了tile k的长度限制，也不采取这个format
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







def load_snap_graph(op_type, name = "arxiv", feat_size = 32, pad=True, delimiter=None):
	a = None
	if '.txt' in name:
		a = np.loadtxt(name, dtype='int')
	elif '.csv' in name:
		a = np.loadtxt(name, dtype='int', delimiter=",")
	else:
		a = np.loadtxt(name, dtype='float', delimiter=delimiter)
	a = a[:,:2].astype('int')
	a = np.unique(a, axis=0).flatten()
	_, indices = np.unique(a, return_inverse=True)
	a = np.reshape(indices, (-1, 2))
	starts, ends = a.T
	node_num = max(max(starts), max(ends)) + 1
	coo = scipy.sparse.coo_matrix((np.full(len(starts), 1, dtype="float32"), (starts, ends)), shape=(node_num, node_num))
	A_csr = coo.tocsr()
	indptr = A_csr.indptr
	indices = A_csr.indices

	sidxs = [0,1]
	ridxs = [2]
	idx_lens = None

	if pad:
		if op_type == 'spmm':
			# idx_lens = [math.ceil(g.num_dst_nodes()/128)*128, feat_size, math.ceil(g.num_src_nodes()/128)*128]
			# 这样改是因为有的一构图总node数太大了，内存不够
			idx_lens = [math.ceil( (len(indptr)-1) /128)*128, feat_size, math.ceil( (max(indices)+1) /128)*128]
		elif op_type == 'sddmm':
			idx_lens = [math.ceil(node_num/128)*128, feat_size, math.ceil(node_num/160)*160]
	else:
		idx_lens = [node_num, feat_size, node_num]
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
	# 没有必要再获取一次indptr, indices信息
	# indptr, indices, _ = g.adj_tensors("csr", etype=etype) # 更新了dgl版本之后使用这个函数名。
	# new_row_is, new_ptr = csr_to_dcsr(op_type, indptr)

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	i_pad_num = 0
	if pad:
		# i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 - g.num_dst_nodes()
		# 对于多张异构体在一起的g，这种写法就不对了
		# i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 + 1 - len(indptr)
		i_pad_num = idx_lens[0] + 1 - len(indptr)
	# indptr = list(indptr) + [indptr[-1] for tmp in range(i_pad_num)]
	indptr = np.concatenate( [indptr, np.full(i_pad_num, indptr[-1])] )

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	# A_val = tuple( np.array([1]*len(indices)).astype("float32") )
	A_val = np.full(len(indices), 1, dtype="float32")

	A_csr = scipy.sparse.csr_matrix((A_val, indices, indptr), shape=( idx_lens[0], idx_lens[2] ))
	A_csr.sum_duplicates()
	A_csr = scipy.sparse.csr_matrix((np.full(len(A_csr.data), 1, dtype="float32"), A_csr.indices, A_csr.indptr), shape=( idx_lens[0], idx_lens[2] ))


	# return A_data, A_raw_poses
	# inps = [A_csr, DenseTensor(B)]
	inps = [A_csr] + [DenseTensor(B) for B in Bs]
	sparse_inps_transposed = [ A_csr.tocsc() ]


	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)



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

	if etype == None:
		g = dgl.to_homogeneous(g)
	# if etype == None:
	# 	indptr, indices = combine_subgraphs(g)
	# else:
	indptr, indices, _ = g.adj_tensors("csr", etype=etype) # 更新了dgl版本之后使用这个函数名。

	if pad:
		if op_type == 'spmm':
			# idx_lens = [math.ceil(g.num_dst_nodes()/128)*128, feat_size, math.ceil(g.num_src_nodes()/128)*128]
			# 这样改是因为有的一构图总node数太大了，内存不够
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
	# 没有必要再获取一次indptr, indices信息
	# indptr, indices, _ = g.adj_tensors("csr", etype=etype) # 更新了dgl版本之后使用这个函数名。
	# new_row_is, new_ptr = csr_to_dcsr(op_type, indptr)

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	i_pad_num = 0
	if pad:
		# i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 - g.num_dst_nodes()
		# 对于多张异构体在一起的g，这种写法就不对了
		# i_pad_num = math.ceil(g.num_dst_nodes()/128)*128 + 1 - len(indptr)
		i_pad_num = idx_lens[0] + 1 - len(indptr)
	# indptr = list(indptr) + [indptr[-1] for tmp in range(i_pad_num)]
	indptr = np.concatenate( [indptr, np.full(i_pad_num, indptr[-1])] )

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	# A_val = tuple( np.array([1]*len(indices)).astype("float32") )
	A_val = np.full(len(indices), 1, dtype="float32")

	A_csr = scipy.sparse.csr_matrix((A_val, indices, indptr), shape=( idx_lens[0], idx_lens[2] ))
	A_csr.sum_duplicates()
	A_csr = scipy.sparse.csr_matrix((np.full(len(A_csr.data), 1, dtype="float32"), A_csr.indices, A_csr.indptr), shape=( idx_lens[0], idx_lens[2] ))


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

	# scipy 的数据结果不支持float16，所以等最后计算的时候在切换成float16
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

	# scipy 的数据结果不支持float16，所以等最后计算的时候在切换成float16
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










def test_LogSparse(op_type, m, n, feat_size = 32):
	'''
	测试 LogSparse 这个类型的sparse matrix。
	'''
	indices = list()
	for i in range(m):
		tmp = None
		if i == 0:
			tmp = [i] + [i+2**j for j in range(int(math.log(n-i-1, 2)))]
		elif i == n-1:
			tmp = [i-2**j for j in range( int(math.log(i, 2)), -1, -1 )] + [i]
		else:
			tmp = [i-2**j for j in range( int(math.log(i, 2)), -1, -1 )] + [i] + [i+2**j for j in range(int(math.log(n-i-1, 2)))]
		indices.append( tmp )
	indptr = [0] + [len(i) for i in indices]
	indptr = np.cumsum(indptr)
	indices = np.concatenate(indices)
	data = np.full(len(indices), 1, dtype="float32")
	A_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=( m, n ))

	idx_graph = None
	idx_lens = [m, feat_size, n]
	
	rng = np.random.default_rng(seed=0)
	Bs = list()
	if op_type == 'spmm':
		Bs = [rng.random((idx_lens[2],feat_size)).astype("float32")]
	elif op_type == 'sddmm':
		Bs = [rng.random((idx_lens[0],feat_size)).astype("float32"), rng.random((idx_lens[2],feat_size)).astype("float32")]

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	inps = [A_csr] + [DenseTensor(B) for B in Bs]
	sparse_inps_transposed = [ A_csr.tocsc() ]

	sidxs = [0,1]
	ridxs = [2]
	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)






def test_Strided(op_type, m, n, feat_size = 32):
	'''
	测试 strided 这个类型的sparse matrix。
	'''
	indices = list()
	s = int(n**0.5)
	for i in range(m):
		tmp = list(range(max(i-s+1, 0), min(i+s, n))) + [i+j*s for j in range( 1, int((n-1-i)/s) )] + [i-j*s for j in range( 1, int(i/s) )]
		tmp = np.unique(tmp)
		indices.append( tmp )
	indptr = [0] + [len(i) for i in indices]
	indptr = np.cumsum(indptr)
	indices = np.concatenate(indices)
	data = np.full(len(indices), 1, dtype="float32")
	A_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=( m, n ))

	idx_graph = None
	idx_lens = [m, feat_size, n]
	
	rng = np.random.default_rng(seed=0)
	Bs = list()
	if op_type == 'spmm':
		Bs = [rng.random((idx_lens[2],feat_size)).astype("float32")]
	elif op_type == 'sddmm':
		Bs = [rng.random((idx_lens[0],feat_size)).astype("float32"), rng.random((idx_lens[2],feat_size)).astype("float32")]

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	inps = [A_csr] + [DenseTensor(B) for B in Bs]
	sparse_inps_transposed = [ A_csr.tocsc() ]

	sidxs = [0,1]
	ridxs = [2]
	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)








def test_random_sample(op_type, m, n, feat_size = 32, patch_size = 2, mask_ratio = 0.75):
	'''
	测试 MAE 里面提到的 random sample 这个类型的sparse matrix。
	'''	
	# we use random shuffle to implement random sampling as MAE does.
	"""
	Perform per-sample random masking by per-sample shuffling.
	Per-sample shuffling is done by argsort random noise.
	x: [N, L, D], sequence
	"""

	L = (m//patch_size) * (n//patch_size)
	D = patch_size * patch_size
	assert (m % patch_size == 0) and (n % patch_size == 0)

	len_keep = int(L * (1 - mask_ratio))  
	noise = torch.rand(L)  # noise in [0, 1]

	# sort noise for each sample
	ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove

	# keep the first subset
	indices = np.sort(ids_shuffle[:len_keep])
	indices = [indices // (n//patch_size), indices % (n//patch_size)]
	_, indptr = np.unique(indices[0], return_counts=True)
	indptr = np.concatenate([[0], np.cumsum(indptr)])
	indices = indices[1]
	data = np.ones((len(indices), patch_size, patch_size), dtype="float32")

	A_bsr = scipy.sparse.bsr_array((data,indices,indptr), shape=(m, n))
	A_csr = A_bsr.tocsr()
	
	idx_graph = None
	idx_lens = [m, feat_size, n]
	
	rng = np.random.default_rng(seed=0)
	Bs = list()
	if op_type == 'spmm':
		Bs = [rng.random((idx_lens[2],feat_size)).astype("float32")]
	elif op_type == 'sddmm':
		Bs = [rng.random((idx_lens[0],feat_size)).astype("float32"), rng.random((idx_lens[2],feat_size)).astype("float32")]

	print(idx_lens, len(indptr), len(indices), indptr[-1])

	inps = [A_csr] + [DenseTensor(B) for B in Bs]
	sparse_inps_transposed = [ A_csr.tocsc() ]

	sidxs = [0,1]
	ridxs = [2]
	return Op(op_type, sidxs, ridxs, idx_lens, idx_graph, inps, sparse_inps_transposed)

















def bench_cusparse(csr: Any, X: torch.Tensor, dtype: str, cuda_i: int):
	# W and X are of the same date type
	dtype_th = None
	if dtype == 'float16':
		# TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
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

		torch.cuda.reset_peak_memory_stats()
		W @ X
		memory_usage = torch.cuda.max_memory_allocated() / (1024*1024)


		measure = profile_pytorch_ms(lambda: W @ X)
		# 
		print("cusparse time: \t{:.5f}ms".format(measure), f" mem: {memory_usage}")
		return measure, memory_usage



def bench_cublas(csr: Any, X: torch.Tensor, dtype: str, cuda_i: int):
	# W and X are of the same date type
	dtype_th = None
	if dtype == 'float16':
		# TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
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
		# TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
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



