import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


# from utils import create_pixelfly, create_longformer

import dgl
import torch
import argparse
from utils import get_dataset
from sparsetir_artifact import profile_pytorch_ms


from gen_formats_v2 import test_real_op_pruned_bert, test_real_op_pruned_bert_unstructured, test_LogSparse, test_Strided, test_random_sample, load_snap_graph, test_real_op
import json
import tvm
import numpy as np


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
    g = dgl.graph(('csr', (indptr, op.inps[0].indices, [])), idtype=torch.int32)
    return g, node_num







def test_csr_spmm(name: str, feat_size: int, filename: str):
    num_heads = 12
    num_heads = 1
    # if pattern == "pixelfly":
    #     csr = create_pixelfly(1, 4096 // 16, fmt="csr", block_size=16)
    # elif pattern == "longformer":
    #     csr = create_longformer(1, 4096 // 16, 256 // 16, fmt="csr", block_size=16)
    # else:
    #     raise KeyError("Pattern {} not supported.".format(pattern))
    # g = dgl.from_scipy(csr).int()

    g, _ = get_graph('spmm', 0, feat_size, name, m=4096)

    g = g.to(0)
    w_gpu = torch.rand(num_heads, g.num_edges()).half().to(0)
    x_gpu = torch.rand(num_heads, 4096, 64).half().to(0)


    # measure memory usage
    torch.cuda.reset_peak_memory_stats()
    [dgl.ops.u_mul_e_sum(g, x_gpu[head], w_gpu[head]) for head in range(num_heads)]
    memory_usage = torch.cuda.max_memory_allocated() / (1024*1024)
    
    measure = profile_pytorch_ms(
        lambda: [
            dgl.ops.u_mul_e_sum(g, x_gpu[head], w_gpu[head])
            for head in range(num_heads)
        ]
    )

    print("cusparse csrmm time: \t{:.5f} ms".format(measure))

    with open(filename, 'a') as file:
        json.dump( ("sparsetir_csr", (name, 0), feat_size, measure, ('mem', memory_usage)), file )
        file.write('\n')

    # when w_gpu is all-ones, res == y_true.
    # res = [dgl.ops.u_mul_e_sum(g, x_gpu[head], w_gpu[head]) for head in range(num_heads)][0]
    # y_true = dgl.ops.copy_u_sum(g, x_gpu[0])
    # tvm.testing.assert_allclose(res.cpu().numpy(), y_true.cpu().numpy(), rtol=1e-4)

    return measure



filename = "Mem_SparsetirCSR_SAttention_fp16.json"
with open(filename, 'a') as file:
    file.write(f"\n\n\n\nNew Round---------\n")

names = ['strided', 'logsparse']
for name in names:
    for feat_size in [32, 64, 128, 256, 512]:
        test_csr_spmm(name, feat_size, filename)





# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("CSR spmm")
#     parser.add_argument(
#         "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
#     )
#     parser.add_argument(
#         "--check", "-c", action="store_true", help="Whether to check result or not."
#     )
#     args = parser.parse_args()
#     test_csr_spmm(args.pattern)