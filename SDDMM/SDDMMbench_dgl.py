import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

import dgl
import sys
import argparse
import torch as th
from sparsetir_artifact import profile_pytorch_ms
from utils import get_dataset

import my_search_formats
import numpy as np
import json

thdtyoe = th.float16




# use a unified method for profiling---------------------------------------------------------------------------------------------------------

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
        assert False
        op = my_search_formats.test_real_op(op_type, name = name, feat_size = feat_size)
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




def do_profile(op_type, name, data_i, feat_size, filename, m=4096, patch_size = 2, mask_ratio = 0.75):
    g, node_num = get_graph(op_type, data_i, feat_size, name, m=m, patch_size = patch_size, mask_ratio = mask_ratio)
    # prepare two input matrices
    
    print(1, ' ',  th.cuda.memory_allocated(), th.cuda.max_memory_allocated())

    a_gpu = th.rand(node_num, feat_size).to(thdtyoe).to(0)
    b_gpu = th.rand(node_num, feat_size).to(thdtyoe).to(0)
    g = g.to(0)
    
    print(2, ' ',  th.cuda.memory_allocated(), th.cuda.max_memory_allocated())


    th.cuda.reset_peak_memory_stats()
    dgl.ops.u_dot_v(g, a_gpu, b_gpu)
    memory_usage = th.cuda.max_memory_allocated() / (1024*1024)

    print(3, ' ',  th.cuda.memory_allocated(), th.cuda.max_memory_allocated())

    dur = profile_pytorch_ms(lambda: dgl.ops.u_dot_v(g, a_gpu, b_gpu))
    print( f"{[(name, data_i), feat_size]}, " , "dgl time:\t{:.5f} ms".format(dur), f" (mem, {memory_usage})")
    
    with open(filename, 'a') as file:
        json.dump( ("dgl", (name, data_i), feat_size, {'latency': dur, 'mem': memory_usage}), file )
        file.write('\n')



# use a unified method for profiling     END  ---------------------------------------------------------------------------------------------------------




def bench_sddmm_prunedbert(data_i, feat_size, get_op):
    op_type = 'sddmm'
    op, _ = get_op(op_type, data_i, feat_size = feat_size, print_infor=False)
    # 此处计算c_golden有问题，因为不一定是B比A更大，所以更为general的解决方案是直接把g pad成长宽一样的样子
    node_num = max(*(op.inps[0].shape))
    indptr = op.inps[0].indptr
    indptr = np.concatenate([indptr, np.full(node_num+1-len(indptr), indptr[-1]) ])
    g = dgl.graph(('csr', (indptr, op.inps[0].indices, []))).to(0)
    A = np.array(op.inps[1].data)
    A = np.concatenate([A, np.full((node_num-A.shape[0], A.shape[1]), 0)], axis=0)
    A = th.from_numpy(A).to(thdtyoe).to(0)
    B = np.array(op.inps[2].data)
    B = np.concatenate([B, np.full((node_num-B.shape[0], B.shape[1]), 0)], axis=0)
    B = th.from_numpy(B).to(thdtyoe).to(0)
    # c_golden = dgl.ops.u_dot_v(g, A, B)
    dur = profile_pytorch_ms(lambda: dgl.ops.u_dot_v(g, A, B))
    # print("dgl time:\t{:.5f} ms".format(dur))
    print(f"dgl time: {json.dumps((data_i, feat_size, dur))}")


def bench_sddmm(g: dgl.DGLGraph, feat_size: int):
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    a_gpu = th.rand(m, feat_size).to(thdtyoe).to(0)
    b_gpu = th.rand(n, feat_size).to(thdtyoe).to(0)
    g = g.to(0)
    dur = profile_pytorch_ms(lambda: dgl.ops.u_dot_v(g, a_gpu, b_gpu))
    print("dgl time:\t{:.5f} ms".format(dur))


if __name__ == "__main__":

    filename = 'Mem_Baselines_fp16.json'
    filename = 'Baselines_res_extraDatasets_fp16.csv'
    with open(filename, 'a') as file:
        file.write("NEW ROUND-----------\n")

    names = ['logsparse', 'strided'] + ['cora', 'ppi', 'arxiv', 'proteins', 'pubmed', 'reddit', 'citeseer', 'out.web-NotreDame'] + ['pruned_bert', 'pruned_bert_unstructured']
    names = ['logsparse', 'strided', 'out.web-NotreDame']
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
                do_profile(op_type, name, data_i, feat_size, filename, m=4096)




    # below is the original code (not support the NotreDame dataset)

    # parser = argparse.ArgumentParser("sddmm in dgl")
    # parser.add_argument(
    #     "--dataset", "-d", type=str, default="pubmed", help="dataset name"
    # )
    # args = parser.parse_args()
    # name = args.dataset
    # if name not in ['pruned_bert', 'pruned_bert_unstructured']:
    #     g = get_dataset(name)
    #     for feat_size in [32, 64, 128, 256, 512]:
    #         print("feat_size = ", feat_size)
    #         try:
    #             bench_sddmm(g, feat_size)
    #         except Exception as e:
    #             print("OOM")
    #             print(e, file=sys.stderr)
    # elif name == 'pruned_bert':
    #     # measure the latency on pruned-bert sparse matrices
    #     _, tot_num = my_search_formats.test_real_op_pruned_bert('sddmm', float('inf'), feat_size = 32, print_infor=True)
    #     for feat_size in [32, 64, 128, 256, 512]:
    #         print("feat_size = ", feat_size, "tot_num = ", tot_num)
    #         for data_i in range(tot_num):
    #             try:
    #                 bench_sddmm_prunedbert(data_i, feat_size, my_search_formats.test_real_op_pruned_bert)
    #             except Exception as e:
    #                 print("OOM")
    #                 print(e, file=sys.stderr)
    # elif name == 'pruned_bert_unstructured':
    #     # measure the latency on pruned-bert sparse matrices
    #     _, tot_num = my_search_formats.test_real_op_pruned_bert_unstructured('sddmm', float('inf'), feat_size = 32, print_infor=True)
    #     for feat_size in [32, 64, 128, 256, 512]:
    #         print("feat_size = ", feat_size, "tot_num = ", tot_num)
    #         for data_i in range(tot_num):
    #             try:
    #                 bench_sddmm_prunedbert(data_i, feat_size, my_search_formats.test_real_op_pruned_bert_unstructured)
    #             except Exception as e:
    #                 print("OOM")
    #                 print(e, file=sys.stderr)


