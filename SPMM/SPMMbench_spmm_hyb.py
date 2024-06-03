# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


import dgl
import tvm
import sys
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
import tvm.sparse
from utils import get_dataset
from sparsetir_artifact import profile_tvm_ms


import json

# <jingzhi>@revision support new graph datasets
from gen_formats_v2 import test_real_op_pruned_bert, test_real_op_pruned_bert_unstructured, test_LogSparse, test_Strided, test_random_sample, load_snap_graph, test_real_op
import math


dtype = "float16"
# zerotype = T.float16(0)

@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), dtype)
    T.evaluate(0)


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    coarsening_factor: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(coarsening_factor)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), dtype)
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), dtype)
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), dtype)
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = T.float16(0) #zerotype # 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


cached_bucketing_format = None




# NOTE: should not pad here
def get_op(op_type, data_i, feat_size, name, m=4096, patch_size = 2, mask_ratio = 0.75):
    op = None
    if name == 'pruned_bert':
        op, _ = test_real_op_pruned_bert(op_type, data_i, feat_size = feat_size, print_infor=False)
    elif name == 'pruned_bert_unstructured':
        op, _ = test_real_op_pruned_bert_unstructured(op_type, data_i, feat_size = feat_size, print_infor=False)
    elif name == 'logsparse':
        op = test_LogSparse(op_type, m, m, feat_size = feat_size)
    elif name == 'strided':
        op = test_Strided(op_type, m, m, feat_size = feat_size)
    elif name == 'random':
        op = test_random_sample(op_type, m, m, feat_size = feat_size, patch_size = patch_size, mask_ratio = mask_ratio)
    elif ('.txt' in name) or ('.csv' in name) or ('.' in name):
        op = load_snap_graph(op_type, name = name, feat_size = feat_size, pad=False)
    else:
        op = test_real_op(op_type, name = name, feat_size = feat_size, pad=False)
    return op


def get_graph(op_type, data_i, feat_size, name, m=4096, patch_size = 2, mask_ratio = 0.75):
    if name in ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer']:
        g = get_dataset(name)
        return g, g.num_src_nodes()

    op = get_op(op_type, data_i, feat_size, name, m=m, patch_size = patch_size, mask_ratio = mask_ratio)
    node_num = max(*(op.inps[0].shape))
    indptr = op.inps[0].indptr
    indptr = np.concatenate([indptr, np.full(node_num+1-len(indptr), indptr[-1]) ])
    g = dgl.graph(('csr', (indptr, op.inps[0].indices, [])), idtype=th.int32)
    return g, node_num



def get_memory_usage(args_nd):
    # return how many MB the given args_nd takes
    return sum([arg.numpy().nbytes for arg in args_nd])/(1024**2)





def bench_hyb(
    name,
    # g,
    # x,
    # y_golden,
    feat_size=128,
    bucket_sizes=[],
    coarsening_factor=2,
    num_col_parts=1,
    use_implicit_unroll=False,
    # dtype="float32",
    filename='results.json'
):
    # g = get_dataset(name)
    g, _ = get_graph('spmm', 0, feat_size, name, m=4096)
    x = th.rand((g.num_src_nodes(), feat_size))
    y_golden = dgl.ops.copy_u_sum(g, x)

    num_buckets = len(bucket_sizes)
    coarsening_factor = min(coarsening_factor, feat_size // 32)
    indptr, indices, _ = g.adj_tensors("csc") # g.adj_tensors("csc") # g.adj_sparse("csc") 
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    global cached_bucketing_format
    if cached_bucketing_format is None:
        indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
        indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
        cached_bucketing_format = column_part_hyb(
            m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
        )
    row_indices, col_indices, mask = cached_bucketing_format

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            rewrites.append(
                FormatRewriteRule(
                    str(part_id) + "_" + str(bucket_id),
                    ell.specialize({nnz_cols_symbol: bucket_size}),
                    ["A"],
                    ["I", "J"],
                    ["O", "I", "J"],
                    {"I": ["O", "I"], "J": ["J"]},
                    csr2ell_index_map,
                    csr2ell_inv_index_map,
                )
            )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coarsening_factor,  # coersening_factor
    }
    for part_id in range(num_col_parts):
        for bucket_id in range(num_buckets):
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
                part_id
            ][bucket_id].shape[0]

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    sch = tvm.tir.Schedule(mod)
    for sp_iter_name in [
        "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
    ]:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
            blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
            i, j, foo, foi, fi = sch.get_loops(blk)
            sch.reorder(foo, fi, j, foi)
            if is_atomic:
                sch.annotate(blk, "atomic", True)
                write_blk = sch.cache_write(blk, 0, "local")
                sch.reverse_compute_at(write_blk, fi, True)
                # sch.unroll(sch.get_loops(write_blk)[-2])
            sch.bind(fi, "threadIdx.x")
            sch.bind(foo, "blockIdx.y")
            sch.unroll(foi)
            if use_implicit_unroll:
                sch.annotate(foi, "pragma_unroll_explicit", 0)
            sch.unroll(j)
            if use_implicit_unroll:
                sch.annotate(j, "pragma_unroll_explicit", 0)
            io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
            sch.bind(io, "blockIdx.x")
            sch.bind(ii, "threadIdx.y")
            init_blk = sch.decompose_reduction(blk, fi)
            ax0, ax1 = sch.get_loops(init_blk)[-2:]
            sch.bind(ax0, "threadIdx.x")
            sch.unroll(ax1)
            if use_implicit_unroll:
                sch.annotate(ax1, "pragma_unroll_explicit", 0)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    # print(mod.script())
    # print(f.imported_modules[0].get_source())

    # print(x.numpy().reshape(-1).astype(dtype))
    # prepare nd array
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype(dtype),
        device=tvm.cuda(0),
    )
    # print("HERE 1")
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype(dtype), device=tvm.cuda(0))
    # prepare args
    args = [b_nd, c_nd]
    # print(b_nd)
    # print(c_nd)

    for part_id in range(num_col_parts):
        for bucket_id, _ in enumerate(bucket_sizes):
            weight = tvm.nd.array(
                mask[part_id][bucket_id].numpy().reshape(-1).astype(dtype), device=tvm.cuda(0)
            )
            rows = tvm.nd.array(
                row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
            )
            cols = tvm.nd.array(
                col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
                device=tvm.cuda(0),
            )
            args += [weight, rows, cols]
            # print(weight)
            # print(rows)
            # print(cols)

    # test accuracy
    f(*args)
    # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)
    try:
        tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-2, atol=1e-2)
    except Exception as e:
        print(e)

    # evaluate time
    dur = profile_tvm_ms(f, args)
    print("tir hyb time: {:.5f} ms".format(dur))

    with open(filename, 'a') as file:
        json.dump([(name, 0), feat_size, dur, ('mem', get_memory_usage(args)), ('bucket_sizes', bucket_sizes[-1]), ('num_col_parts', num_col_parts)] , file)
        file.write('\n')


col_part_config = {
    "arxiv": 1,
    "proteins": 8,
    "pubmed": 1,
    "citeseer": 1,
    "cora": 1,
    "ppi": 16,
    "reddit": 8,
    "products": 16,
}

bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "citeseer": [1, 2, 4],
    "cora": [1, 2, 4],
    "ppi": [1, 2, 4, 8, 16, 32],
    "products": [1, 2, 4, 8, 16, 32],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    # 'out.web-NotreDame': [1, 2, 4, 8], # the formula is ceil(log_2 (nnz/n))
    # 'strided':,
    # 'logsparse':,
}




# collect the memory information of the following datasets
if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default="arxiv", help="dataset name")
    parser.add_argument("--bucketsize", "-s", type=int, default=0, help="max bucket size")
    parser.add_argument("--colnum", "-c", type=int, default=0, help="column partition number")
    # names = ['ppi', 'cora', 'citeseer', 'pubmed', 'arxiv', 'proteins', 'reddit'] # + ['pruned_bert', 'pruned_bert_unstructured']
    # names = ['out.web-NotreDame']

    filename = "Mem_SparseTIR_fp16.json"
    # with open(filename, 'a') as file:
    #     file.write(f"\n\n\n\nNew Round---------\n")

    args = parser.parse_args()
    name = args.dataset
    max_bucket_size = args.bucketsize
    bucket_sizes = None
    if max_bucket_size > 0:
        bucket_sizes = [2**i for i in range(int(math.log(max_bucket_size, 2)+1))]
        print(bucket_sizes)
    else:
        bucket_sizes = bucketing_config[name]
    num_col_parts = args.colnum
    if num_col_parts == 0:
        num_col_parts = col_part_config[name]
    print(num_col_parts)


    implicit_unroll = True
    # for name in names:
    # g = get_dataset(name)

    # print(dtype, flush=True)
    # global dtype
    # dtype = args.dtype

    for feat_size in [32, 64, 128, 256, 512]:
        print(f"name = {name}, feat_size = {feat_size}")
        # try:
        # x = th.rand((g.num_src_nodes(), feat_size))
        # y_golden = dgl.ops.copy_u_sum(g, x)
        bench_hyb(
            name,
            # g,
            # x,
            # y_golden,
            feat_size=feat_size,
            bucket_sizes=bucket_sizes, #bucketing_config[name],
            coarsening_factor=2,
            num_col_parts=num_col_parts, #col_part_config[name],
            use_implicit_unroll=implicit_unroll,
            # dtype=args.dtype,
            filename=filename,
        )
        # except Exception as e:
        #     print("OOM")
        #     print(e, file=sys.stderr)







# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
#     parser.add_argument("--dataset", "-d", type=str, default="arxiv", help="dataset name")
#     parser.add_argument("--implicit-unroll", "-i", action="store_true", help="use implicit unroll")
#     # parser.add_argument("--dtype", "-t", default="float32", help="the data type")
#     args = parser.parse_args()
#     name = args.dataset
#     g = get_dataset(name)

#     # print(dtype, flush=True)
#     # global dtype
#     # dtype = args.dtype

#     for feat_size in [32, 64, 128, 256, 512]:
#         print("feat_size = ", feat_size)
#         try:
#             x = th.rand((g.num_src_nodes(), feat_size))
#             y_golden = dgl.ops.copy_u_sum(g, x)
#             bench_hyb(
#                 g,
#                 x,
#                 y_golden,
#                 feat_size=feat_size,
#                 bucket_sizes=bucketing_config[name],
#                 coarsening_factor=2,
#                 num_col_parts=col_part_config[name],
#                 use_implicit_unroll=args.implicit_unroll,
#                 # dtype=args.dtype,
#             )
#         except Exception as e:
#             print("OOM")
#             print(e, file=sys.stderr)
