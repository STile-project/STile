import os
os.environ['CUDA_VISIBLE_DEVICES']='3'


import dgl
import sys
import tvm
import argparse
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from ogb.nodeproppred import DglNodePropPredDataset
from sparsetir_artifact import profile_tvm_ms
from utils import get_dataset

import my_search_formats
import json

import math


dtype = "float16"
thdtyoe = th.float16




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




# Note should not pad here
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
        # assert False
        op = my_search_formats.test_real_op(op_type, name = name, feat_size = feat_size, pad=False)
    return op




def get_graph(op_type, data_i, feat_size, name, m=4096, patch_size = 2, mask_ratio = 0.75):
    if name in ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer']:
        return get_dataset(name)

    op = get_op(op_type, data_i, feat_size, name, m=m, patch_size = patch_size, mask_ratio = mask_ratio)
    node_num = max(*(op.inps[0].shape))
    indptr = op.inps[0].indptr
    indptr = np.concatenate([indptr, np.full(node_num+1-len(indptr), indptr[-1]) ])
    g = dgl.graph(('csr', (indptr, op.inps[0].indices, [])), idtype=th.int32)
    return g




def get_prunedbert_graph(data_i, feat_size, get_op):
    op_type = 'sddmm'
    op, _ = get_op(op_type, data_i, feat_size = feat_size, print_infor=False)
    # 此处计算c_golden有问题，因为不一定是B比A更大，所以更为general的解决方案是直接把g pad成长宽一样的样子
    node_num = max(*(op.inps[0].shape))
    indptr = op.inps[0].indptr
    indptr = np.concatenate([indptr, np.full(node_num+1-len(indptr), indptr[-1]) ])
    g = dgl.graph(('csr', (indptr, op.inps[0].indices, [])), idtype=th.int32)
    return g



def get_fixpattern_graph(name, m):
    op_type = 'sddmm'
    op = None
    if name == 'logsparse':
        op = my_search_formats.test_LogSparse(op_type, m, m, feat_size = 32)
    elif name == 'strided':
        op = my_search_formats.test_Strided(op_type, m, m, feat_size = 32)

    indptr = op.inps[0].indptr
    g = dgl.graph(('csr', (indptr, op.inps[0].indices, [])), idtype=th.int32)
    return g



def sddmm(m: int, n: int, feat_size: int, nnz: int):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        indptr: T.handle,
        indices: T.handle,
    ) -> None:
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        I = T.dense_fixed(m)
        J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
        J_detach = T.dense_fixed(n)
        K = T.dense_fixed(feat_size)
        A = T.match_sparse_buffer(a, (I, K), dtype)
        B = T.match_sparse_buffer(b, (J_detach, K), dtype)
        C = T.match_sparse_buffer(c, (I, J), dtype)

        with T.iter([I, J, K], "SSR", "sddmm") as [i, j, k]:
            with T.init():
                C[i, j] = 0.0
            C[i, j] = C[i, j] + A[i, k] * B[j, k]

    return func




def get_memory_usage(args_nd):
    # return how many MB the given args_nd takes
    return sum([arg.numpy().nbytes for arg in args_nd])/(1024**2)




# def bench_sddmm(g: dgl.DGLGraph, feat_size: int, data_i=0, name='', filename="mem_log.json"):
def bench_sddmm(op, feat_size: int, data_i=0, name='', filename="mem_log.json"):
    global sddmm
    # indptr, indices, _ = g.adj_tensors("csr") # g.adj_sparse("csr")
    # m = g.num_src_nodes()
    # n = g.num_dst_nodes()
    # nnz = g.number_of_edges()

    indptr, indices = op.inps[0].indptr, op.inps[0].indices
    m, n = op.inps[0].shape
    nnz = op.inps[0].nnz

    a = th.rand(m, feat_size).to(thdtyoe)
    b = th.rand(n, feat_size).to(thdtyoe)
    c = th.zeros(nnz).to(thdtyoe)

    # dgl
    # a_gpu = a.to(0)
    # b_gpu = b.to(0)
    # g = g.to(0)
    # c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)
    
    a_dgl = th.rand(max(m, n), feat_size).to(thdtyoe)
    b_dgl = th.rand(max(m, n), feat_size).to(thdtyoe)
    g_dgl = get_graph(op.op_type, data_i, feat_size, name, m=4096)
    a_dgl = a_dgl.to(0)
    b_dgl = b_dgl.to(0)
    g_dgl = g_dgl.to(0)
    c_golden = dgl.ops.u_dot_v(g_dgl, a_dgl, b_dgl)[:m, :n]



    # tvm
    mod = tvm.IRModule.from_expr(sddmm(m, n, feat_size, nnz))
    sch = tir.Schedule(mod)
    sp_iteration = sch.get_sparse_iteration("sddmm")
    i, j, k = sch.get_sp_iters(sp_iteration)
    sch.sparse_fuse(sp_iteration, [i, j])
    mod = lower_sparse_iter(sch.mod)

    # split preprocess and compute
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    mod_sddmm = tvm.tir.transform.RemovePreprocess()(mod)

    # schedule preprocess
    sch = tir.Schedule(mod_preprocess)
    blk = sch.get_block("binary_search_block_0_0")
    (i,) = sch.get_loops(blk)
    io, ii = sch.split(i, [None, 32])
    sch.bind(ii, "threadIdx.x")
    sch.bind(io, "blockIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    preproc = tvm.build(mod["main"], target="cuda")

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.cuda())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.cuda())
    c_nd = tvm.nd.array(c.numpy(), tvm.cuda())
    # indptr_nd = tvm.nd.array(indptr.numpy(), tvm.cuda())
    # indices_nd = tvm.nd.array(indices.numpy(), tvm.cuda())
    indptr_nd = tvm.nd.array(indptr, tvm.cuda())
    indices_nd = tvm.nd.array(indices, tvm.cuda())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.cuda())

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)

    # can direct get the memory usage informationa as the input & output are ready now
    # NOTE: indptr_nd will not be used in SDDMM computation
    memory_usage = get_memory_usage([a_nd, b_nd, c_nd, indices_nd, mid_nd])
    # with open(filename, 'a') as file:
    #     json.dump( ("sparsetir", (name, data_i), feat_size, ('mem', memory_usage)), file )
    #     file.write('\n')

    # return 


    ty_candidates = [1, 2, 4, 8]
    tx_candidates = [8, 16, 32]
    vecsize_candidates = [1, 2, 4]
    groupsize_candidates = [1, 2, 4]

    best = 1e9
    best_config = None
    best_memory_usage = None
    for ty in ty_candidates:
        for tx in tx_candidates:
            for vec_size in vecsize_candidates:
                for group_size in groupsize_candidates:
                    if tx * vec_size > feat_size:
                        continue
                    # schedule compute
                    sch = tir.Schedule(mod_sddmm)
                    blk = sch.get_block("sddmm0")
                    j, k = sch.get_loops(blk)
                    ko, kio, kii = sch.split(k, [None, tx, vec_size])
                    rf_blk = sch.rfactor(kio, 2)
                    j = sch.get_loops(rf_blk)[0]
                    joo, joi, ji = sch.split(j, [None, ty, group_size])
                    sch.bind(joo, "blockIdx.x")
                    sch.bind(joi, "threadIdx.y")
                    sch.unroll(ji)
                    sch.reverse_compute_at(blk, joi, True)
                    sch.set_scope(rf_blk, 0, "local")
                    read_A = sch.cache_read(rf_blk, 0, "local")
                    read_B = sch.cache_read(rf_blk, 2, "local")
                    write_C = sch.cache_write(blk, 0, "local")
                    ko, kio, kii = sch.get_loops(rf_blk)[-3:]
                    sch.reorder(ko, ji)
                    # schedule read A
                    sch.compute_at(read_A, ji, True)
                    ax0, ax1 = sch.split(sch.get_loops(read_A)[-1], [tx, vec_size])
                    sch.bind(ax0, "threadIdx.x")
                    sch.vectorize(ax1)
                    # schedule read B
                    sch.compute_at(read_B, ji, True)
                    ax0, ax1 = sch.split(sch.get_loops(read_B)[-1], [tx, vec_size])
                    sch.bind(ax0, "threadIdx.x")
                    sch.vectorize(ax1)
                    # schedule write C
                    sch.reverse_compute_at(write_C, joi, True)
                    ax0, ax1 = sch.get_loops(write_C)[-2:]
                    sch.vectorize(ax1)
                    # schedule rf
                    sch.bind(kio, "threadIdx.x")
                    sch.unroll(kii)
                    sch.unroll(ko)
                    # schedule write back
                    ax0, ax1, ax2 = sch.get_loops(blk)[-3:]
                    sch.reorder(ax1, ax2, ax0)
                    sch.bind(ax0, "threadIdx.x")
                    sch.unroll(ax2)
                    sch.unroll(ax1)
                    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
                    f = tvm.build(mod["main"], target="cuda")

                    # check result
                    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd]
                    f(*args)
                    
                    try:
                        tvm.testing.assert_allclose(
                            c_nd.numpy(), c_golden.view(-1).cpu(), rtol=1e-2, atol=1e-2 # , rtol=1e-5
                        )
                    except Exception as e:
                        print(e)

                    # evaluate time
                    mean_time = profile_tvm_ms(f, args)
                    print(f"op:{(name, data_i)}, config: {(tx, ty, vec_size, group_size)}, mean_time: {mean_time}          best:{best} config: {best_config}")
                    # if feat_size == 512 and (tx, ty, vec_size, group_size) == (16, 2, 4, 4):
                    #     print(f.imported_modules[0].get_source())

                    # <jingzhi>@response: print the memory usage information
                    # print(f"Memory consumption: {(math.prod(a_nd.shape)+math.prod(b_nd.shape)+math.prod(c_nd.shape))*2+(math.prod(indptr_nd.shape)+math.prod(indices_nd.shape)+math.prod(mid_nd.shape))*4}")
                    print(f"Memory consumption: {(math.prod(a_nd.shape)+math.prod(b_nd.shape)+math.prod(c_nd.shape))*2+(math.prod(indices_nd.shape)+math.prod(mid_nd.shape))*4}")
                    assert memory_usage*1024*1024 == (math.prod(a_nd.shape)+math.prod(b_nd.shape)+math.prod(c_nd.shape))*2+(math.prod(indices_nd.shape)+math.prod(mid_nd.shape))*4, f"wrong memory usage:{memory_usage*1024*1024, (math.prod(a_nd.shape)+math.prod(b_nd.shape)+math.prod(c_nd.shape))*2+(math.prod(indices_nd.shape)+math.prod(mid_nd.shape))*4}"

                    if mean_time < best:
                        best = mean_time
                        best_config = (tx, ty, vec_size, group_size)
                        best_memory_usage = (math.prod(a_nd.shape)+math.prod(b_nd.shape)+math.prod(c_nd.shape))*2\
                                        +(math.prod(indices_nd.shape)+math.prod(mid_nd.shape))*4
                        best_memory_usage = best_memory_usage / (1024*1024)
                        assert best_memory_usage == memory_usage, f'wrong memory usage: {best_memory_usage, memory_usage*1024*1024}'
    print(f"{json.dumps((name, data_i))}, ", "sparse tir:\t{:.5f} ms".format(best), f"config: {best_config}", f"memory usage: {best_memory_usage}")
    # with open('sparsetir_res.json', 'a') as file:
    with open(filename, 'a') as file:
        file.write(json.dumps(('sparsetir', (name, data_i), feat_size, best, best_config, best_memory_usage)))
        file.write('\n')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("sddmm in sparse-tir")
    # parser.add_argument(
    #     "--dataset", "-d", type=str, default="pubmed", help="dataset name"
    # )
    # args = parser.parse_args()
    # name = args.dataset
    # with open('sparsetir_res.json', 'a') as file:
    #     file.write("NEW ROUND-----------\n")


    filename = 'Mem_Baselines_fp16.json'
    filename = 'Baselines_res_extraDatasets_fp16.csv'
    filename = 'Baselines_res_ReRUN_fp16.csv'
    with open(filename, 'a') as file:
        file.write("NEW ROUND-----------\n")

    names = ['logsparse', 'strided'] + ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer', 'out.web-NotreDame'] + ['pruned_bert', 'pruned_bert_unstructured']
    names = ['logsparse', 'strided', 'out.web-NotreDame']
    names = ['pruned_bert_unstructured', 'pruned_bert']
    feat_sizes = [32, 64, 128, 256, 512][::-1]

    op_type = 'sddmm'
    for name in names:
        tot_num = 1
        if name in ['pruned_bert', 'pruned_bert_unstructured']:
            tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)
        for data_i in range(tot_num):
            for feat_size in feat_sizes:
                if (name in ['pruned_bert', 'pruned_bert_unstructured']) and (feat_size != 512):
                    continue
                # g = get_graph(op_type, data_i, feat_size, name, m=4096)
                op = get_op(op_type, data_i, feat_size, name, m=4096)
                print(f"{[[name, data_i], feat_size]}")
                try:
                    bench_sddmm(op, feat_size, data_i=data_i, name = name, filename=filename)
                except Exception as e:
                    print("OOM")
                    print(e, file=sys.stderr)                  


    # ================================================================================
    # Below is the original code, can also work (but not support graph NotraDame)

    # for name in names:
    #     if name in ['logsparse', 'strided']:
    #         g = get_fixpattern_graph(name, 4096)
    #         for feat_size in [32, 64, 128, 256, 512]:
    #             print("feat_size = ", feat_size)
    #             try:
    #                 bench_sddmm(g, feat_size, data_i=0, name = name, filename=filename)
    #             except Exception as e:
    #                 print("OOM")
    #                 print(e, file=sys.stderr)       
    #     elif name not in ['pruned_bert', 'pruned_bert_unstructured']:
    #         g = get_dataset(name)
    #         for feat_size in [32, 64, 128, 256, 512]:
    #             print("feat_size = ", feat_size)
    #             try:
    #                 bench_sddmm(g, feat_size, data_i=0, name = name, filename=filename)
    #             except Exception as e:
    #                 print("OOM")
    #                 print(e, file=sys.stderr)
    #     else:
    #         assert name in ['pruned_bert', 'pruned_bert_unstructured']
    #         get_op = None
    #         if name == 'pruned_bert':
    #             get_op = my_search_formats.test_real_op_pruned_bert
    #         else:
    #             get_op = my_search_formats.test_real_op_pruned_bert_unstructured
    #         # 
    #         _, tot_num = get_op('sddmm', float('inf'), feat_size = 32, print_infor=True)
    #         for feat_size in [512]:
    #             print("feat_size = ", feat_size, "tot_num = ", tot_num)
    #             for data_i in range(tot_num):
    #                 try:
    #                     g = get_prunedbert_graph(data_i, feat_size, get_op)
    #                     bench_sddmm(g, feat_size, data_i, name, filename=filename)
    #                 except Exception as e:
    #                     print("OOM")
    #                     print(e, file=sys.stderr)



