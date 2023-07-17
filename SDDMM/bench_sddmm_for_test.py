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

import json


dtype = "float16"
thdtyoe = th.float16
bit = 16


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


def bench_sddmm(filename, g: dgl.DGLGraph, feat_size: int, idx=None, pred_cost=None, given_params=None):
    global sddmm
    indptr, indices, _ = g.adj_tensors("csr") # g.adj_sparse("csr")
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    nnz = g.number_of_edges()

    a = th.rand(m, feat_size).to(thdtyoe)
    b = th.rand(n, feat_size).to(thdtyoe)
    c = th.zeros(nnz).to(thdtyoe)

    # dgl
    a_gpu = a.to(0)
    b_gpu = b.to(0)
    g = g.to(0)
    c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)

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
    
    # print(f"init mod preproc: {sch.mod.script()}")

    blk = sch.get_block("binary_search_block_0_0")
    (i,) = sch.get_loops(blk)
    io, ii = sch.split(i, [None, 32])
    sch.bind(ii, "threadIdx.x")
    sch.bind(io, "blockIdx.x")
    mod = lower_sparse_buffer(sch.mod)

    # print(f"\n\n\noptimized mod preproc: {mod.script()}")

    preproc = tvm.build(mod["main"], target="cuda")

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.cuda())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.cuda())
    c_nd = tvm.nd.array(c.numpy(), tvm.cuda())
    indptr_nd = tvm.nd.array(indptr.numpy(), tvm.cuda())
    indices_nd = tvm.nd.array(indices.numpy(), tvm.cuda())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.cuda())

    # print(f"\n\n\nbefore preproc")
    # print(a_nd.numpy(), b_nd.numpy(), c_nd.numpy(), indptr_nd.numpy(), indptr_nd.numpy().dtype, 
    #                     indices_nd.numpy(), indices_nd.numpy().dtype, 
    #                     mid_nd.numpy(), mid_nd.numpy().dtype)

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)

    # print(f"\n\n\npreproc success")

    # ty_candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # tx_candidates = [1, 2, 4, 8, 16, 32] # [8, 16, 32]
    # vecsize_candidates = [1, 2, 4]
    # groupsize_candidates = [1, 2, 4, 8] # [1, 2, 4]

    ty_candidates = [1, 2, 4, 8, 16, 32]
    tx_candidates = [8, 16]
    vecsize_candidates = [4]
    groupsize_candidates = [1, 2, 4, 8]

    params = None
    if feat_size == 512:
        params = [(16, 2, 4, 4), (8, 4, 4, 1)] # (tx, ty, vec_size, group_size)
    elif feat_size == 256:
        params = [(16, 16, 4, 2), (16, 32, 4, 1)]
    elif feat_size == 128:
        params = [(8, 16, 4, 2), (16, 2, 4, 8)]
    elif feat_size == 64:
        params = [(16, 4, 4, 8)]
    elif feat_size == 32:
        params = [(8, 4, 4, 8), (8, 16, 4, 4)]

    if given_params != None:
        params = [given_params]


    # ty_candidates = [4]
    # tx_candidates = [8] # [8, 16, 32]
    # vecsize_candidates = [4]
    # groupsize_candidates = [8] # [1, 2, 4]

    best = 1e9
    best_config = None
    # for ty in ty_candidates:
    #     for tx in tx_candidates:
    #         for vec_size in vecsize_candidates:
    #             for group_size in groupsize_candidates:
    for (tx, ty, vec_size, group_size) in params:
        for fake_i in range(1):
            for fake_j in range(1):
                for fake_k in range(1):
                    if tx * vec_size > feat_size:
                        continue

                    if (tx * ty < 32) or ((tx*ty)%32!=0):
                        continue

                    # schedule compute
                    sch = tir.Schedule(mod_sddmm)

                    # print(f"config: {(tx, ty, vec_size, group_size)}")

                    # print(f"\n\n\ninit mod sddmm: {sch.mod.script()}")

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


                    # print(f"\n\n\n{tx, vec_size, ty, group_size} \n\noptimized mod sddmm: {mod.script()}")
                    # print(f"\n\n\nsource code: {f.imported_modules[0].get_source()}")



                    # check result
                    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd]
                    f(*args)
                    # print(a_nd.numpy(), b_nd.numpy(), c_nd.numpy(), indptr_nd.numpy(), indptr_nd.numpy().dtype, 
                    #     indices_nd.numpy(), indices_nd.numpy().dtype, 
                    #     mid_nd.numpy(), mid_nd.numpy().dtype)
                    
                    try:
                        tvm.testing.assert_allclose(
                            c_nd.numpy(), c_golden.view(-1).cpu(), rtol=1e-2, atol=1e-2 # rtol=1e-5
                        )
                    except Exception as e:
                        print("\nERROR\n")
                        # print(e)

                    # evaluate time
                    mean_time = profile_tvm_ms(f, args)

                    print(f"config: {(tx, ty, vec_size, group_size)}, mean_time: {mean_time}          best:{best} config: {best_config}")
                    # break
                    mean_time = profile_tvm_ms(f, args)
                    print(f"config: {(tx, ty, vec_size, group_size)}, mean_time: {mean_time}          best:{best} config: {best_config}")


                    with open(filename, 'a') as f:
                        json.dump([int(idx), feat_size, (tx, ty, vec_size, group_size), mean_time, pred_cost], f)
                        f.write('\n')

                    if mean_time < best:
                        best = mean_time
                        best_config = (tx, ty, vec_size, group_size)
    print("sparse tir:\t{:.5f} ms".format(best), best_config)


