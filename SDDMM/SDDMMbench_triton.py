import os
os.environ['CUDA_VISIBLE_DEVICES']='3'


import pytest
import torch
import argparse
import triton
# from utils import create_pixelfly, create_longformer
from sparsetir_artifact import profile_pytorch_ms

import my_search_formats
import math
import numpy as np
import scipy.sparse as sp


import json




# NOTE: should not pad here
def get_op(op_type, data_i, feat_size, name, m=4096, patch_size = 2, mask_ratio = 0.75):
    op = None
    if name == 'pruned_bert':
        op, _ = my_search_formats.test_real_op_pruned_bert(op_type, data_i, feat_size = feat_size, print_infor=False)
    elif name == 'pruned_bert_unstructured':
        op, _ = my_search_formats.test_real_op_pruned_bert_unstructured(op_type, data_i, feat_size = feat_size, print_infor=False)
    elif name == 'logsparse':
        op = my_search_formats.test_LogSparse(op_type, m, m, feat_size = feat_size)
    elif name == 'strided':
        op = my_search_formats.test_Strided(op_type, m, m, feat_size = feat_size)
    elif name == 'random':
        op = my_search_formats.test_random_sample(op_type, m, m, feat_size = feat_size, patch_size = patch_size, mask_ratio = mask_ratio)
    elif ('.txt' in name) or ('.csv' in name) or ('.' in name):
        # NOTE: do not do padding
        op = my_search_formats.load_snap_graph(op_type, name = name, feat_size = feat_size, pad=False)
    else:
        op = my_search_formats.test_real_op(op_type, name = name, feat_size = feat_size, pad=False)
    return op






def get_input(name, data_i, feat_size, block_size):
    op_type = 'sddmm'
    # op = None
    # if name == 'pruned_bert':
    #     op, _ = my_search_formats.test_real_op_pruned_bert(op_type, data_i, feat_size = feat_size, print_infor=False)
    # else:
    #     op, _ = my_search_formats.test_real_op_pruned_bert_unstructured(op_type, data_i, feat_size = feat_size, print_infor=False)
    op = get_op(op_type, data_i, feat_size, name, m=4096)
    # 
    # get the C_clock data from op with the given block_size
    mb = math.ceil(op.inps[0].shape[0] / block_size)
    nb = math.ceil(op.inps[0].shape[1] / block_size)
    csr = op.inps[0]
    indices = [ np.nonzero(
                np.add.reduceat( csr[i*block_size:(i+1)*block_size, :].getnnz(axis=0), np.arange(0, csr.shape[1], block_size))
                )[0] for i in range(mb)]
    indptr = np.cumsum([0] + [len(i) for i in indices])
    nnzb = indptr[-1]
    indices = np.concatenate(indices)

    blk_coo = sp.csr_matrix((np.ones(nnzb), indices, indptr)).tocoo()
    blk_index = torch.zeros(1, mb, nb, dtype=torch.int32)
    blk_index[:, blk_coo.row, blk_coo.col] = 1

    return op.inps[0].shape, mb, nb, indices, indptr, nnzb, blk_index




def test_matmul(
    MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, sp_pattern, Z=1, H=1, M=512, N=384, K=256, layout=None, data_i=0, 
    for_measure_memory = False
):
    seed = 0
    torch.manual_seed(seed)
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
    # create inputs
    # create op
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]

    # if sp_pattern == "pixelfly":
    #     layout = create_pixelfly(H, M // BLOCK, fmt="mask")
    # elif sp_pattern == "longformer":
    #     layout = create_longformer(H, shape[0] // BLOCK, 256 // BLOCK, fmt="mask")
    # else:
    #     raise KeyError("Sparse pattern {} not recongized.".format(args.pattern))




    # create data
    a_ref, a_tri = triton.testing.make_pair(a_shape, dtype=DTYPE, alpha=0.1)
    b_ref, b_tri = triton.testing.make_pair(b_shape, dtype=DTYPE, alpha=0.1)
    # compute [torch]
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    c_ref = torch.matmul(
        a_ref.transpose(2, 3) if TRANS_A else a_ref,
        b_ref.transpose(2, 3) if TRANS_B else b_ref,
    )
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    # triton result
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    # print(a_tri)
    op = triton.ops.blocksparse.matmul(
        layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda"
    )

    print(1.5, ' ',  torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

    if for_measure_memory:
        return op, a_tri, b_tri


    c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)
    # compare
    triton.testing.assert_almost_equal(c_ref, c_tri)

    measure = profile_pytorch_ms(lambda: op(a_tri, b_tri))
    print(f"{[(sp_pattern, data_i), K]}", "triton time: \t{:.5f} ms".format(measure))
    return measure







def measure_memory(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, sp_pattern, Z=1, H=1, M=512, N=384, K=256, layout=None, data_i=0):
    print(1, ' ',  torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    op, a_tri, b_tri = test_matmul(
        MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, sp_pattern, Z=Z, H=H, M=M, N=N, K=K, layout=layout, data_i=data_i,
        for_measure_memory = True
    )

    # measure memory usage
    torch.cuda.reset_peak_memory_stats()
    print(2, ' ',  torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

    op(a_tri, b_tri)
    memory_usage = torch.cuda.max_memory_allocated() / (1024*1024)

    print(3, ' ',  torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())

    return memory_usage




def get_pruned_bert_graphNum_and_getOpFunc(name):
    tot_num = None
    get_op = None

    if name == 'pruned_bert':
        # measure the latency on pruned-bert sparse matrices
        _, tot_num = my_search_formats.test_real_op_pruned_bert('sddmm', float('inf'), feat_size = 32, print_infor=True)
        get_op = my_search_formats.test_real_op_pruned_bert
    elif name == 'pruned_bert_unstructured':
        _, tot_num = my_search_formats.test_real_op_pruned_bert_unstructured('sddmm', float('inf'), feat_size = 32, print_infor=True)
        get_op = my_search_formats.test_real_op_pruned_bert_unstructured

    return tot_num, get_op


def do_profile(name, filename, feat_size):
    # get_op = None
    # # name = args.pattern
    # if name == 'pruned_bert':
    #     get_op = my_search_formats.test_real_op_pruned_bert
    # else:
    #     get_op = my_search_formats.test_real_op_pruned_bert_unstructured
    # # 
    # _, tot_num = get_op('sddmm', float('inf'), feat_size = 32, print_infor=True)

    tot_num = 1
    if 'pruned_bert' in name:
        tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)

    # feat_size = 512
    block_size = 16
    # block_size = 32 # try 32 for pruned-bert-structured
    for data_i in range(tot_num):
        (m, n), mb, nb, indices, indptr, nnzb, blk_index = get_input(name, data_i, feat_size, block_size)
        memory_usage = measure_memory(
            "sdd",
            False,
            True,
            block_size,
            torch.float16,
            name, #args.pattern,
            Z=1,
            H=1,
            M=m,
            N=n,
            K=feat_size,
            layout = blk_index,
            data_i=data_i
        )
        latency = test_matmul(
            "sdd",
            False,
            True,
            block_size,
            torch.float16,
            name, #args.pattern,
            Z=1,
            H=1,
            M=m,
            N=n,
            K=feat_size,
            layout = blk_index,
            data_i=data_i,
            for_measure_memory = False
        )

        with open(filename, 'a') as file:
            json.dump( ("triton", (name, data_i), feat_size, {'latency': latency, 'mem': memory_usage}), file )
            file.write('\n')





if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Triton sparse attention sddmm")
    # parser.add_argument(
    #     "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
    # )
    # args = parser.parse_args()

    filename = 'Mem_Baselines_fp16.json'
    filename = 'Baselines_res_extraDatasets_fp16.csv'
    filename = 'Baselines_res_ReRUN_fp16.csv' # set the block size to 32 and rerun on pruned-bert structured matrices
    with open(filename, 'a') as file:
        file.write("NEW ROUND-----------\n")

    names = ['pruned_bert', 'pruned_bert_unstructured']
    names = ['logsparse', 'strided'] #, 'out.web-NotreDame']
    names = ['pruned_bert']
    feat_sizes = [32, 64, 128, 256, 512]

    for name in names:
        for feat_size in feat_sizes:
            if ('pruned_bert' in name) and (feat_size != 512):
                continue
            do_profile(name, filename, feat_size)
