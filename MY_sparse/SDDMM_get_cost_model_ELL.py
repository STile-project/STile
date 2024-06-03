# 用来获取1D tile所需要的cost model数据
from bench_sddmm_for_test import *
import math


name = 'arxiv'
g = get_dataset(name)
bit_num = 16

# 首先获得g的邻接矩阵
indptr, indices, _ = g.adj_tensors("csr") # g.adj_sparse("csr")
tot = math.ceil(len(indptr) / 32)

row_nums = dict()


def get_row_nums(tot, indptr, indices, row_nums):
	for idx in range(tot):
		row1 = np.searchsorted(indptr, idx*32, side='right') - 1
		row2 = min(np.searchsorted(indptr, (idx+1)*32-1, side='right'), len(indptr)-1) # exclusive
		num = int(row2-row1)
		if num not in row_nums:
			row_nums[num] = list()
		row_nums[num].append(idx)



def prepare_fake_graph(idx, indptr, indices):
	# idx is the index of the 1D tile
	# first get the row ids of this tile
	row1 = np.searchsorted(indptr, idx*32, side='right') - 1
	row2 = min(np.searchsorted(indptr, (idx+1)*32-1, side='right'), len(indptr)-1) # exclusive
	# get the nnzs per row
	# nnzs = np.asarray([ min(indptr[i+1], (idx+1)*32) - indptr[i] for i in range(row1, row2)], dtype='int32')
	# 
	nnzs = indptr[row1:row2+1]
	nnzs[-1] = min(nnzs[-1], (idx+1)*32)
	nnzs[0] = max(nnzs[0], idx*32)
	nnzs = np.diff(nnzs)
	print(row1, row2, nnzs, sum(nnzs))
	# 
	tmp_indices = indices[ idx*32 : (idx+1)*32 ]
	# tmp_indices的数据有问题，我们也应该对其平移，但是首先应该对其压缩
	_, tmp_indices = np.unique(tmp_indices, return_inverse=True)
	# 
	fake_indptr = np.tile(nnzs, 108*100)
	fake_indptr = np.concatenate([[0], np.cumsum(fake_indptr)])
	# 
	node_num = max(len(tmp_indices)*108*100, len(fake_indptr)-1)
	fake_indptr = np.concatenate([fake_indptr, [fake_indptr[-1] for i in range( node_num+1-len(fake_indptr) )]])
	# fake_indices = np.tile(tmp_indices, 108*100)
	step = len(tmp_indices)
	fake_indices = np.concatenate([tmp_indices+step*i for i in range(108*100)])
	# print(fake_indptr, len(fake_indptr), fake_indices, len(fake_indices))
	assert fake_indptr[-1] == len(fake_indices)
	g = dgl.graph(('csr', (fake_indptr, fake_indices, [])), idtype=th.int32, num_nodes=node_num)
	return g


get_row_nums(tot, indptr, indices, row_nums)
idxs = [np.random.choice(row_nums[k], min(50, len(row_nums[k])), replace=False) for k in sorted(row_nums.keys())]
print(idxs)
print(row_nums.keys())

with open(f"cost_model_fp{bit_num}_fulltune4.json", 'a') as f:
	f.write(f"\n\n\n\nnew round\n")

for feat_size in [32, 64, 128, 256, 512][::-1]:
	print("feat_size = ", feat_size)
	with open(f"cost_model_fp{bit_num}_fulltune4.json", 'a') as f:
		f.write(f"new op: {json.dumps((name, feat_size))}\n")
	# 
	# idxs = np.random.choice(tot, 1000, replace=False)
	for cnt, vs in enumerate(idxs):
		with open(f"cost_model_fp{bit_num}_fulltune4.json", 'a') as f:
			f.write(f"row num: {sorted(row_nums.keys())[cnt]}\n")		
		for idx in vs:
			g = prepare_fake_graph(idx, indptr, indices)
			# 
			try:
				bench_sddmm(g, feat_size, idx)
			except Exception as e:
				print("OOM")
				print(e, file=sys.stderr)




