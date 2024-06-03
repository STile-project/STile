import os
os.environ['CUDA_VISIBLE_DEVICES']='3'



import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.ir import IRModule
import argparse
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
import numpy as np
import torch as th
import scipy.sparse as sp
# from utils import create_pixelfly, create_longformer
from sparsetir_artifact import profile_tvm_ms


import my_search_formats
import math
import dgl

import json


def sddmm(mb: int, nb: int, nnzb: int, block_size: int, feat_size: int, num_heads: int):
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
        I = T.dense_fixed(mb)
        J = T.sparse_variable(I, (nb, nnzb), (indptr, indices), "int32")
        J_detach = T.dense_fixed(nb)
        BI = T.dense_fixed(block_size)
        BJ = T.dense_fixed(block_size)
        H = T.dense_fixed(num_heads)
        F = T.dense_fixed(feat_size)
        A = T.match_sparse_buffer(a, (H, I, BI, F), "float16")
        B = T.match_sparse_buffer(b, (H, J_detach, BJ, F), "float16")
        C = T.match_sparse_buffer(c, (H, I, J, BI, BJ), "float16")

        with T.iter([H, I, J, BI, BJ, F], "SSSSSR", "sddmm") as [
            h,
            i,
            j,
            bi,
            bj,
            f,
        ]:
            with T.init():
                C[h, i, j, bi, bj] = T.float16(0)
            C[h, i, j, bi, bj] = C[h, i, j, bi, bj] + A[h, i, bi, f] * B[h, j, bj, f]

    return func


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = (
                    C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]
                )


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:16, 0:16],
                A_frag[0:16, 0:16],
                B_frag[0:16, 0:16],
            ]
        )
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256
                    + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256
                    + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global"
    )
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s0, s1],
    )
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    A_frag.data,
                    16,
                    16,
                    16,
                    A_frag.elem_offset // 256
                    + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="global"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vjj, vii]


@T.prim_func
def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s0, s1],
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        T.reads(B[0:16, 0:16])
        T.writes(B_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B_frag.data,
                    16,
                    16,
                    16,
                    B_frag.elem_offset // 256
                    + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "col_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="global"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="global",
        strides=[s0, s1],
    )
    with T.block("root"):
        T.reads(C_frag[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A = tir.TensorIntrin.register(
    "wmma_load_a",
    wmma_load_a_desc,
    wmma_load_a_impl,
)

WMMA_LOAD_B = tir.TensorIntrin.register(
    "wmma_load_b",
    wmma_load_b_desc,
    wmma_load_b_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)

WMMA_STORE = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)




def get_dgl_res(ori_op, A, B):
    
    np.random.seed(0)
    cuda_i = 0
    dtype_th = th.float16

    node_num = max(*(ori_op.inps[0].shape))
    indptr = ori_op.inps[0].indptr
    indptr = np.concatenate([indptr, np.full(node_num+1-len(indptr), indptr[-1]) ])
    g = dgl.graph(('csr', (indptr, ori_op.inps[0].indices, []))).to(cuda_i)
    # A = np.array(ori_op.inps[1].data)

    # A = np.random.rand(*(A.shape))

    # it is possible that node_num is smaller than A.shape[0], as we may pad zeros to A for complete blocks
    if A.shape[0] <= node_num:
        A = np.concatenate([A, np.full((node_num-A.shape[0], A.shape[1]), 0)], axis=0)
    else:
        A = A[:node_num]
    A = th.from_numpy(A).to(dtype_th).to(cuda_i)
    print(node_num, A.shape, flush=True)
    # B = np.array(ori_op.inps[2].data)

    # B = np.random.rand(*(B.shape))

    if B.shape[0] <= node_num:
        B = np.concatenate([B, np.full((node_num-B.shape[0], B.shape[1]), 0)], axis=0)
    else:
        B = B[:node_num]
    B = th.from_numpy(B).to(dtype_th).to(cuda_i)
    print(node_num, B.shape, flush=True)
    c_golden = dgl.ops.u_dot_v(g, A, B)
    return c_golden




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
    blk_index = [ blk_coo.row, blk_coo.col ]

    # tmp_csrs = [(csr[i*block_size:(i+1)*block_size, :][:, j*block_size:(j+1)*block_size], i, j) for i in range(mb) for j in range(nb)]
    # tmp_csrs = [i for i in tmp_csrs if i[0].nnz!=0]
    # remap_scr = [tmp_csrs[i][0].indices for i in range(len(tmp_csrs))]
    # 感觉可以直接利用scipy把bsr转化成csr的格式。
    # c_dgl = get_dgl_res(op)

    return mb, nb, indices, indptr, nnzb, blk_index, op #, c_dgl






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




def get_memory_usage(args_nd):
    # return how many MB the given args_nd takes
    return sum([arg.numpy().nbytes for arg in args_nd])/(1024**2)





def do_profile(name, check, filename, feat_size = 512):
    block_size = 16
    # feat_size = 512
    num_heads = 1
    # name = args.pattern

    # get_op = None
    # if name == 'pruned_bert':
    #     get_op = my_search_formats.test_real_op_pruned_bert
    # else:
    #     get_op = my_search_formats.test_real_op_pruned_bert_unstructured
    # # 
    # _, tot_num = get_op('sddmm', float('inf'), feat_size = 32, print_infor=True)

    tot_num = 1
    if 'pruned_bert' in name:
        tot_num, _ = get_pruned_bert_graphNum_and_getOpFunc(name)


    for data_i in range(tot_num):
        mb, nb, indices, indptr, nnzb, blk_index, op = get_input(name, data_i, feat_size, block_size)

        np.random.seed(0)
        data = np.random.rand(num_heads, nnzb, block_size, block_size)
        A = np.random.rand(num_heads, mb, 1, block_size, feat_size).astype("float16")
        B = np.random.rand(num_heads, 1, nb, block_size, feat_size).astype("float16")

        c_dgl = get_dgl_res(op, A.reshape((-1, feat_size)), B.reshape((-1, feat_size)))

        if check:
            C = np.matmul(A, B.transpose(0, 1, 2, 4, 3))
            # nonzero = mask.nonzero()
            # C_ground_truth = C[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]]
            C_ground_truth = C[[0 for i in range(nnzb)], blk_index[0], blk_index[1]]

        best_dur = 1e9
        best_config = None
        best_memory_usage = None
        for ty in [2, 4, 8]:
            tmp_nnzb = math.ceil(nnzb/ty)*ty

            print(f"ori nnzb: {nnzb}, padded nnzb: {tmp_nnzb}, ty: {ty}")

            tmp_indptr = indptr.copy()
            tmp_indptr[-1] = tmp_nnzb
            tmp_indices = indices.copy()
            tmp_indices = np.concatenate([tmp_indices, np.full(tmp_nnzb-nnzb, tmp_indices[-1])])

            sch = tvm.tir.Schedule(sddmm(mb, nb, tmp_nnzb, block_size, feat_size, num_heads))
            sp_iteration = sch.get_sparse_iteration("sddmm")
            h, i, j, bi, bj, f = sch.get_sp_iters(sp_iteration)
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
            mod = tvm.sparse.lower_sparse_buffer(sch.mod)
            mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
            preproc = tvm.build(mod["main"], target="cuda")

            # compute mid
            indptr_nd = tvm.nd.array(tmp_indptr.astype("int32"), tvm.cuda())
            mid_nd = tvm.nd.array(np.zeros((tmp_nnzb,), np.int32), tvm.cuda())

            preproc(indptr_nd, mid_nd)

            # schedule sddmm
            sch = tir.Schedule(mod_sddmm)
            blk = sch.get_block("sddmm0")
            h, j, bi, bj, f = sch.get_loops(blk)
            fo, fi = sch.split(f, [None, 16])
            sch.bind(h, "blockIdx.y")
            sch.reorder(j, fo, bi, fi, bj)
            assert tmp_nnzb % ty == 0
            jo, ji = sch.split(j, [None, ty])
            sch.bind(jo, "blockIdx.x")
            sch.bind(ji, "threadIdx.y")
            C_local = sch.cache_write(blk, 0, "wmma.accumulator")
            sch.reverse_compute_at(C_local, ji)
            new_blk = sch.blockize(bi)
            sch.decompose_reduction(new_blk, fo)
            A_local = sch.cache_read(blk, 1, "wmma.matrix_a")
            B_local = sch.reverse_cache_read(blk, 3, "wmma.matrix_b", [2, 1])
            sch.hide_buffer_access(blk, "read", [2, 4])
            sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a")
            sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b")
            sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
            ax0, ax1, ax2 = sch.get_loops(blk)[-3:]
            sch.reorder(ax2, ax1)
            sch.tensorize(ax0, "wmma_sync")
            sch.tensorize(sch.get_loops(sch.get_block("sddmm0_init"))[-2], "wmma_fill")
            mod = lower_sparse_buffer(sch.mod)
            f = tvm.build(mod["main"], target="cuda")
            # print(mod.script())
            # print(f.imported_modules[0].get_source())

            ctx = tvm.cuda(0)
            C_indptr = tvm.nd.array(np.copy(tmp_indptr).astype("int32"), device=ctx)
            C_indices = tvm.nd.array(np.copy(tmp_indices).astype("int32"), device=ctx)
            A_nd = tvm.nd.array(np.copy(A.reshape(-1)).astype("float16"), device=ctx)
            B_nd = tvm.nd.array(np.copy(B.reshape(-1)).astype("float16"), device=ctx)
            C_nd = tvm.nd.array(
                np.zeros((num_heads * tmp_nnzb * block_size * block_size,), dtype="float16"),
                device=ctx,
            )
            fargs = [A_nd, B_nd, C_nd, C_indptr, C_indices, mid_nd]

            
            

            memory_usage = get_memory_usage([A_nd, B_nd, C_nd, C_indices, mid_nd])
            # with open(filename, 'a') as file:
            #     json.dump( ("sparsetirBSR", (name, data_i), feat_size, ('mem', memory_usage)), file )
            #     file.write('\n')
            # break



            f(*fargs)
            if check:
                try:
                    tvm.testing.assert_allclose(
                        C_ground_truth.reshape(-1),
                        C_nd.numpy()[:nnzb*block_size*block_size],
                        rtol=1e-2,
                    )

                    # turn c_nd to csr format
                    csr_C_nd = sp.bsr_matrix((C_nd.numpy()[:nnzb*block_size*block_size].reshape((-1, block_size, block_size)),indices,indptr))
                    tmp_bsr = op.inps[0].tobsr(blocksize=(block_size, block_size))
                    csr_C_nd = csr_C_nd.multiply(tmp_bsr)
                    csr_C_nd.eliminate_zeros()
                    csr_C_nd = csr_C_nd.tocsr()
                    # print(sum(tmp_bsr.data.reshape(-1)!=0), nnzb*block_size*block_size)
                    tvm.testing.assert_allclose(
                        c_dgl.view(-1).cpu(),
                        csr_C_nd.data,
                        # C_nd.numpy()[np.nonzero(tmp_bsr.data.reshape(-1))[0]], #[:nnzb*block_size*block_size],
                        # C_ground_truth.reshape(-1),
                        rtol=1e-2, atol=1e-2
                    )

                except Exception as e:
                    print(e)
                    print(C_ground_truth.reshape(-1))
                    print(C_nd.numpy()[:nnzb*block_size*block_size])

            dur = profile_tvm_ms(f, fargs)
            print(f"{(name, data_i, feat_size, ty, dur, best_dur, best_config)}", flush=True)
            dur = profile_tvm_ms(f, fargs)
            print(f"{(name, data_i, feat_size, ty, dur, best_dur, best_config)}", flush=True)
            # best_dur = min(dur, best_dur)
            if dur < best_dur:
                best_dur = dur
                best_config = [ty]
                best_memory_usage = memory_usage
        print("avg time: {} ms".format(best_dur), best_config, flush=True)
        with open(filename, 'a') as file:
            json.dump( ("sparsetirBSR", (name, data_i), feat_size, {'latency': best_dur, 'best_config': best_config, 'mem': memory_usage}), file )
            file.write('\n')        







if __name__ == "__main__":
    # parser = argparse.ArgumentParser("SparseTIR sparse attention sddmm")
    # parser.add_argument(
    #     "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
    # )
    # parser.add_argument(
    #     "--check", "-c", action="store_true", help="Whether to check result or not."
    # )
    # args = parser.parse_args()

    # block_size = 16
    # mb = 256
    # nb = 256
    # feat_size = 64
    # num_heads = 12
    # m = mb * block_size
    # n = nb * block_size

    # if args.pattern == "pixelfly":
    #     C_block = create_pixelfly(1, mb, fmt="bsr")
    #     mask = create_pixelfly(num_heads, mb, fmt="mask")
    # else:
    #     C_block = create_longformer(1, mb, 256 // block_size, fmt="bsr")
    #     mask = create_longformer(num_heads, mb, 256 // block_size, fmt="mask")

    # indptr = C_block.indptr
    # indices = C_block.indices
    # nnzb = C_block.nnz

    # ==================================================
    filename = 'Mem_Baselines_fp16.json'
    filename = 'Baselines_res_extraDatasets_fp16.csv'
    with open(filename, 'a') as file:
        file.write("NEW ROUND-----------\n")


    names = ['pruned_bert', 'pruned_bert_unstructured']
    names = ['logsparse', 'strided', 'out.web-NotreDame']
    names = ['out.web-NotreDame']

    feat_sizes = [32, 64, 128, 256, 512]
    check = False
    for name in names:
        for feat_size in feat_sizes:
            if ('pruned_bert' in name) and (feat_size != 512):
                continue
            do_profile(name, check, filename, feat_size)

