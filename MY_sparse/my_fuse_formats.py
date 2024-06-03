# import json
from gen_formats_v2 import *
from importlib import reload
import os
import my_cost_model
from sparsetir_artifact import profile_tvm_ms
from my_wmmas1 import *
# from my_wmmas2 import *


# <jingzhi>@response: profile memory usage
import time
from dgl import DGLError

# <jingzhi>@response: profile memory usage
# this function is from sparsetir/graphiler/utils/bench
# this function does not work because we will not release the gpu memory after we call the net.
def mem_bench(net, net_params, repeat=1000):
    try:
        # with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=0, warmup=10, active=100), record_shapes=True) as prof:
        #     for _ in range(100):
        #         net(*net_params)
        #         prof.step()
        # print(prof.key_averages())
        # warm up
        for i in range(5):
            net(*net_params)
        torch.cuda.synchronize()
        memory_offset = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        for i in range(repeat):
            net(*net_params)
        torch.cuda.synchronize()

        elapsed_time = (time.time() - start_time) / repeat * 1000
        print("elapsed time: {} ms/infer".format(elapsed_time))
        
        max_mem_consumption = (
            torch.cuda.max_memory_allocated() - memory_offset) / 1048576
        print(torch.cuda.max_memory_allocated(), memory_offset, torch.cuda.max_memory_allocated() - memory_offset)
        print("intermediate data memory usage: {} MB".format(max_mem_consumption))
        return max_mem_consumption
        
    except (RuntimeError, DGLError) as e:
        print(e)
        print("{} OOM".format(tag))
        return None
    except BaseException as e:
        print(e)
        raise




def get_C_store_atomics(formats):
    '''
    Compute the C_store atomics for the given formats.
    '''
    C_store_atomics = list()
    for fmt, tiles in formats.items():
        template_str, tile_sizes, params = fmt
        params = json.loads(params)
        if template_str == "TensorCore_template":
            # continue
            # 默认没有unroll C的存储
            # mma_n = parse_mma_shape(params["mma_shape_str"])[1]
            # C_store_atomics = C_store_atomics + [params['real_atomic']] * (tile_sizes[1][1] // mma_n)
            C_store_atomics.append(params['real_atomic'])
            continue
        if params['real_atomic']:
            if params['use_implicit_unroll']:
                C_store_atomics.append(params['real_atomic'])
            else:
                # 我们在csr和ELL的schedule中，只会对j1和ki轴进行explicit unroll，然后默认每计算完一个output point，就存储到C中一次
                C_store_atomics = C_store_atomics + [params['real_atomic']] * tile_sizes[1][1]
    return C_store_atomics



def comp_SMEM_padding_pattern(op_type, tile_sizes, dsize):
    '''
    这个函数用来计算避免shared memory的bank conflict时所需要进行的padding。
    tile_sizes: 是具体的tile sizes, 包含了每个线程的workload的信息。
    OUTPUT: (每隔多少行进行一次padding, 每次padding多少个单位, 这个SMEM的column数是多少)
    '''
    warp_size = 32
    bank_width = 32
    bank_num = 32
    if op_type == 'spmm':
        thread_j = tile_sizes[1][2] # 我们假定thread_j 一定能整除32或者是32的倍数。因为假定所有tile size都是2的指数。
        i_num_per_warp = math.ceil(warp_size/thread_j)
        if i_num_per_warp == 1:
            # 不会有bank conflict
            return (1, 0, math.prod(tile_sizes[2][1:])) # i.e., 在每一行后面padding 0个单位
        second_bank_i = dsize * tile_sizes[2][1] // bank_width # the bank id of the first element in the second row
        if second_bank_i == 0:
            # 说明两行落在同一个bank里，比如一行一个float16的情况。此时我们要在每一行尾部进行padding。
            pad_num = (bank_width - dsize * tile_sizes[2][1]) // dsize
            return (1, pad_num, math.prod(tile_sizes[2][1:]))
        else:
            # 说明相邻的两行读取同一个k的位置的元素的时候不会在同一个bank里（且是同一个以bank width 为单位的地址里）
            # 我们假定tile k一定是2的指数，因此second_bank_i一定也是2的指数
            # 我们需要找出最少多少行的同一个元素会落在同一个bank里
            i_conflict_num = int(math.ceil(bank_num/second_bank_i)) # 当second_bank_i > bank_num的时候，相邻两行在同一个bank
            if i_conflict_num >= i_num_per_warp:
                # 不会有bank conflict的问题
                return (1, 0, math.prod(tile_sizes[2][1:]))
            else:
                return (i_conflict_num, bank_width//dsize, math.prod(tile_sizes[2][1:])) # 在每i_conflict_num行后pad一个bank_width的宽度








# 尝试加快这个函数，is_atomic直接利用tile里存的信息
def get_formats_from_selected_tiles(selected_tiles, cache_set, dsize, set_atomic = None):
    '''
    selected_tiles: the selected tiles.
    '''
    # 是不是在找key上花的时间太多了

    formats = dict()
    if selected_tiles[0].op.op_type in ['spmm', 'sddmm']:
        for tile in selected_tiles:
            template_str = get_template_str(tile.op)

            # add the information used in schedule functions to params
            params = copy.deepcopy(tile.best_params)
        # if tile.op.op_type == 'spmm':
            # 
            params['real_atomic'] = tile.is_atomic_tile

            # if user specify whether to use atomicAdd ex
            if set_atomic != None:
                params['real_atomic'] = set_atomic

            # 我们不希望在schedule函数中采用sparsetir目前的atomic add的技术，因为有bug
            params['atomic'] = False

            params["idx_reordered"] = tile.op.idx_reordered[0]

            # FOR DEBUG 看是不是unroll分配了太多c local
            # params["use_implicit_unroll"] = False #True

            # 设置tile是否要用shared memory来cache inputs
            params['SMEM_in1'] = False
            if 'A' in cache_set:
                params['SMEM_in1'] = True

            # 对于TC tile，需要保证一起tune的tile是来自同一个op的，因为我们需要一个block计算多个tile
            # if template_str == "TensorCore_template":
            # 已经不需要这个了，因为我们只会有一个TC op
            # params['op_id'] = tile.op.op_id

            key = None
            if get_template_str(tile.op) == '1D_sddmm':
                key = (
                    template_str, 
                    tuple(tile.best_tile_sizes), 
                    json.dumps(params)
                    )
            else:             
                key = (
                    template_str, 
                    tuple([tuple(i) for i in tile.best_tile_sizes]), 
                    json.dumps(params)
                    )
            if key not in formats:
                formats[key] = [tile]
            else:
                formats[key].append(tile)


    # we store whether each sub-kernel needs to be set to atomic in the end
    C_store_atomics = get_C_store_atomics(formats)
    with open(f"C_store_atomics{os.environ['MyFileID']}.json", 'w') as f:
        json.dump(C_store_atomics, f)


    # store the shared memory padding infor to file
    SMEM_pad_patterns = [comp_SMEM_padding_pattern(selected_tiles[0].op.op_type, fmt[1], dsize) \
        for fmt in formats if fmt[0] == 'sparse_template_ell']
    with open(f"SMEM_pad_patterns{os.environ['MyFileID']}.json", 'w') as f:
        json.dump(SMEM_pad_patterns, f)

    print(f"the number of different formats we need {len(formats)}")
    return formats










# ==========================================================================
# Below is about scheduling the hybrid template




def schedule_csr(sch, func_str, tile_sizes, params):
    '''
    sch is returned by the fused function, and we schedule sch.
    func_str is the name of the corresponding format (e.g., "csrmm0" is the first csrmm).
    tile_sizes controls the loop tiling;
    params controls the scheduling parameters.
    '''
    # if op.op_type == 'spmm':
    # NOTE: no cache is used in this implemention

    use_implicit_unroll = params['use_implicit_unroll']

    blk0 = sch.get_block(f"{func_str}0")
    blk1 = sch.get_block(f"{func_str}1")

    # print("before schedule")
    # print(sch.mod.script())

    o, i, j = sch.get_loops(blk0)
    ki,  = sch.get_loops(blk1)[:3]
    # i, ki, j = sch.get_loops(blk1)[:3]
    # print(len(sch.get_loops(blk1)))

    i0, i1, i2 = sch.split(i, tile_sizes[0])
    j0, j1, j2 = sch.split(j, tile_sizes[1])
    # tile index k into only two levels --> there is no need to tile k because we do not consider cache read here
    # ki0, ki1 = sch.split(ki, tile_sizes[2]) 

    # ki放在最后的目的在于，schedule的时候，cache write计算出来的loop范围总是会变大，这样虽然对结果没有影响，但是可能在atomic的时候会浪费时间
    # sch.reorder(j0,j2,i_out, j1, ki)
    sch.reorder(i0, j0, i2, j2, i1, j1)
    # sch.reorder(ki)

    # sch.reorder(j0,j2, ki, j1)

    # cache write stage 【此处做了修改之前的写法是没有cache write的】--------------------------------------
    # print(" params['atomic']",  params['atomic'])
    if params['atomic']:
        sch.annotate(blk1, "atomic", True)
    # 此处修改为默认进行cache write
    write_blk = sch.cache_write(blk1, 0, "local") # sch.cache_write(blk1, 0, "local")
        # sch.reverse_compute_at(write_blk, i_out, True)
    # write_blk = sch.cache_write(blk1, 0, "local") # sch.cache_write(blk1, 0, "local")
    # # print("after cache_write")
    # # print(sch.mod.script())

    # # 一个可能的失败的原因，write_blk只有两个axis，但是j2所在的blk1有四个axis，即j0, j2, ki, j1，是不是需要我们也让write_blk有四个axis就行了？好像不是这个原因，axis的数量只和C的维度有关。待会试一试这个


    # sch.reverse_compute_at(write_blk, j1, True)
    # print("after reverse compute at")
    # print(sch.mod.script())

    # sch.bind(i0, "blockIdx.x")
    # sch.bind(j0, "blockIdx.y")

    ij0 = sch.fuse(i0, j0)
    oij0 = sch.fuse(o, ij0)
    sch.bind(oij0, "blockIdx.x")
    
    fused_ij2 = sch.fuse(i2, j2)
    # sch.bind(i2, "threadIdx.y")
    # sch.bind(j2, "threadIdx.x")
    sch.bind(fused_ij2, "threadIdx.x")



    # set unroll
    sch.unroll(j1)
    if use_implicit_unroll:
        sch.annotate(j1, "pragma_unroll_explicit", 0)

    # 此处做了修改，对i_out也unroll了
    # sch.unroll(i_out)
    # if use_implicit_unroll:
    #     sch.annotate(i_out, "pragma_unroll_explicit", 0)

    # sch.unroll(ki) # 这个地方也做了修改，原来是被注释掉了的. 还是得注释掉，因为cannot unroll non-constant loop
    if use_implicit_unroll:
        sch.annotate(ki, "pragma_unroll_explicit", 0)

    # print(sch.mod.script())
    # parallel initialization as computation
    # init_blk = sch.decompose_reduction(blk1, j2)
    init_blk = sch.decompose_reduction(blk1, ki)

    # print(sch.mod.script())


    # 似乎init_blk里面的axis的个数并不会变，这个修改完之后，就不会出现dataptr == null这个error了
    # print("#axis of init_blk: ", len(sch.get_loops(init_blk)))
    
    # print("before scheduling init_blk")
    # print(sch.mod.script())
    
    # 把j移到i的block内之后应该就不用再设置initblk的并行了---
    # ax0, ax1, ax2 = sch.get_loops(init_blk)[-3:]
    # sch.bind(ax0, "threadIdx.x")
    # sch.unroll(ax1)
    # sch.unroll(ax2)
    # # print("*"*50)
    # # print(sch.mod.script())
    # if use_implicit_unroll:
    #     sch.annotate(ax1, "pragma_unroll_explicit", 0)

    # if tile_sizes[1][1]>1 and tile_sizes[1][2]>1:
    #     print("both > 1")
    #     ax0, ax1 = sch.get_loops(init_blk)[-3:-1]
    #     sch.bind(ax0, "threadIdx.x")
    #     sch.unroll(ax1)
    #     # print("*"*50)
    #     # print(sch.mod.script())
    #     if use_implicit_unroll:
    #         sch.annotate(ax1, "pragma_unroll_explicit", 0)
    # elif tile_sizes[1][1]>1:
    #     print("j1 > 1")
    #     ax0 = sch.get_loops(init_blk)[-2]
    #     sch.bind(ax0, "threadIdx.x")
    # elif tile_sizes[1][2]>1:
    #     print("j2 > 1")
    #     ax1 = sch.get_loops(init_blk)[-2]
    #     sch.unroll(ax1)
    #     if use_implicit_unroll:
    #         sch.annotate(ax1, "pragma_unroll_explicit", 0)
    
    # print(len(sch.get_loops(init_blk)))
    # sch.bind(ax0, "threadIdx.x")
    # sch.unroll(ax1)
    # # print("*"*50)
    # # print(sch.mod.script())
    # if use_implicit_unroll:
    #     sch.annotate(ax1, "pragma_unroll_explicit", 0)
    # print("finish scheduling")
    # print(sch.mod.script())









# 将在这个函数里同时处理有无shared memory，有无cache write的情况
# 默认real atomic的时候要进行cache write，否则不cache write
# 是否使用shared memory要通过params来控制
def schedule_ell(op_type, sch, func_str, tile_sizes, params, in1_shared_str, write_blk_str):
    '''
    sch is returned by the fused function, and we schedule sch.
    func_str is the name of the corresponding format (e.g., "csrmm0" is the first csrmm).
    tile_sizes controls the loop tiling;
    params controls the scheduling parameters.
    '''
    use_implicit_unroll = params['use_implicit_unroll']

    blk0 = sch.get_block(f"{func_str}0")

    in1_shared = None
    if params["SMEM_in1"]:
        in1_shared = sch.get_block(f"{in1_shared_str}0")

    write_blk = None
    if params['real_atomic']:
        write_blk = sch.get_block(f"{write_blk_str}")


    # !!!此处修改为，我们只依靠atomic来判断，事实上，在我们fix cuda bug的时候，我们只依靠real_atomic这个参数来判断
    # 其实这个判断已经没有用处了，因为我们会在cuda代码处对何时应该做atomic进行设置
    if params['atomic']:
        sch.annotate(write_blk, "atomic", True)
    # sch.reverse_compute_at(write_blk, j1, True)

    batch, i0, j0, iblk, jblk, ki = None, None, None, None, None, None
    if op_type == 'spmm':
        i0, j0, iblk, jblk, ki = sch.get_loops(blk0)
    elif op_type == 'batched_spmm':
        batch, i0, j0, iblk, jblk, ki = sch.get_loops(blk0)

    i1, i2 = sch.split(iblk, tile_sizes[0][1:])
    j1, j2 = sch.split(jblk, tile_sizes[1][1:])

    sch.reorder(i2, j2, i1, j1)

    fused_ij2 = sch.fuse(i2, j2)

    if op_type == 'spmm':
        fused_ij0 = sch.fuse(i0, j0)
    elif op_type == 'batched_spmm':
        fused_ij0 = sch.fuse(batch, i0)
        fused_ij0 = sch.fuse(fused_ij0, j0)
    # sch.bind(fused_ij0, "blockIdx.x")

    # set unroll
    sch.unroll(j1)
    if use_implicit_unroll:
        sch.annotate(j1, "pragma_unroll_explicit", 0)

    sch.unroll(ki)
    if use_implicit_unroll:
        sch.annotate(ki, "pragma_unroll_explicit", 0)

    if params['SMEM_in1']:
        i_iter = sch.get_loops(in1_shared)[-1]

        ax0, ax1 = sch.split(i_iter, [None, tile_sizes[0][-1]*tile_sizes[1][-1]])
        sch.unroll(ax0)
        ax1, ax2 = sch.split(ax1, [None, 32])
        # ax1, ax2 = sch.split(ax1, [None, 32*4]) # 为了配合TC tile的线程数
        sch.bind(ax2, "threadIdx.x")
        sch.bind(ax1, "threadIdx.y")

    sch.bind(fused_ij0, "blockIdx.x")

    # 我们要把thread分成x和y两部分
    ax0, ax1 = sch.split(fused_ij2, [None, 32])
    # ax0, ax1 = sch.split(fused_ij2, [None, 32 * 4])
    sch.bind(ax1, "threadIdx.x")
    sch.bind(ax0, "threadIdx.y")
                    
    # it seems we do not need to explicitly parallelize the initialization blow-------
    # again, 做这个步骤会出错，暂时先不做
    # # parallel initialization as computation
    init_blk = sch.decompose_reduction(blk0, ki)






def parse_mma_shape(mma_shape_str: str):
    m_pos = 0
    n_pos = mma_shape_str.index("n")
    k_pos = mma_shape_str.index("k")
    return (
        int(mma_shape_str[m_pos + 1:n_pos]),
        int(mma_shape_str[n_pos + 1:k_pos]),
        int(mma_shape_str[k_pos + 1:]),
    )





# 下面的这个版本是加上了cache write 并且先把tensor core的计算结果存储在shared memory里，在atomic地加回global memory里的
# 我们的schedule是参考Sparsetir的，即不把feature轴也bind给threadidx.y，这样，我们也就不需要对A进行shared memory的cache read了。
# 下面的这个schedule是针对我们已经把所有cache read/write的stage都显式地定义出来的情况下这样做的。
# 这个是最终使用的版本，不允许对block_num i 的维度进一步tiling，引入更多thread
def schedule_tc(op_type, sch, feat_size, func_str, tile_sizes, params, fmt_i):
    '''
    此处的tile_sizes对应的是dbsrmm中的block size, 并不一定是wmma的单位workload, 比如此处的tile size可以是32*32: [i, k]两个维度
    一下的schedule完全参考dbsrmm里的代码。
    '''
    # if not(params['idx_reordered'][0] or params['real_atomic']):
    #     # 此处不需要cache write
    #     schedule_tc_no_cachewrite(sch, feat_size, func_str, tile_sizes, params)
    #     return
    use_C_shared = params['idx_reordered'][0] or params['real_atomic']


    mma_shape_str = params["mma_shape_str"]
    mma_m, mma_n, mma_k = parse_mma_shape(mma_shape_str)

    blk_outer = sch.get_block(f"{func_str}0")

    # (i, foo) = sch.get_loops(blk_outer)
    # ifoo = sch.fuse(i, foo)
    ifoo = None
    if op_type == 'spmm':
        ifoo = sch.fuse(*(sch.get_loops(blk_outer)))
    elif op_type == 'batched_spmm':
        (batch, i, foo) = sch.get_loops(blk_outer)
        ifoo = sch.fuse(batch, i)
        ifoo = sch.fuse(ifoo, foo)     
    sch.bind(ifoo, "blockIdx.x")


    blk_inner = sch.get_block(f"{func_str}1_update")
    blk_inner_oo = sch.get_block(f"{func_str}1_update_oo")
    ko, ii_0, ji_0, ki_0, _ = sch.get_loops(blk_inner_oo)
    ii_1, ki_1, ji_1 = sch.get_loops(blk_inner)

    # sch.unroll(ji_0)
    # sch.bind(ii_0, "threadIdx.y")

    # print(sch.mod.script())

    C_wmma = sch.get_block(f"C_shared_wmma{fmt_i}.accumulator")
    # sch.reverse_compute_at(C_wmma, io)
    # if use_C_shared:
    ax0, ax1 = sch.get_loops(C_wmma)[-2:]
    ax2, ax3 = sch.split(ax1, [None, mma_n])
    ax0, ax1 = sch.split(ax0, [None, mma_m])
    sch.reorder(ax0, ax2, ax1, ax3)
    if use_C_shared:
        sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    # sch.reverse_compute_at(C_shared, io)
    # print("AFTER C_WMMA")
    # print(sch.mod.script())


    A_wmma = sch.get_block(f"A{fmt_i}_wmma.matrix_a")
    ax0, ax1 = sch.get_loops(A_wmma)[-2:]
    ax1, ax2 = sch.split(ax1, [None, mma_k])
    sch.reorder(ax1, ax0, ax2)
    sch.unroll(ax1)
    # print("AFTER A_WMMA")
    # print(sch.mod.script())


    B_shared = sch.get_block(f"B_shared{fmt_i}")
    B_wmma = sch.get_block(f"B_shared_wmma{fmt_i}.matrix_b")
    
    init_blk = sch.get_block(f"tcspmm{fmt_i}1_init")
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    ax2, ax3 = sch.split(ax1, [None, mma_n])
    ax0, ax1 = sch.split(ax0, [None, mma_m])
    sch.reorder(ax0, ax2, ax1, ax3)
    sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    # sch.reverse_compute_at(C_shared, io)
    # print("AFTER C_WMMA")
    # print(sch.mod.script())


    # sch.hide_buffer_access(blk_inner, "read", [1, 2, 5])
    # sch.tensorize(sch.get_loops(A_wmma)[-2], "wmma_m16n16k16_load_a_shared")
    name_tail = ''
    if os.environ['MydtypeIn'] == 'float':
        name_tail = '_fp32'

    if use_C_shared:
        sch.tensorize(sch.get_loops(C_wmma)[-2], f"wmma_{mma_shape_str}_store_shared{name_tail}")
    else:
        # sch.tensorize(sch.get_loops(C_wmma)[-2], "wmma_m16n16k16_store")
        sch.tensorize(sch.get_loops(C_wmma)[-2], f"wmma_{mma_shape_str}_store{name_tail}")

    sch.tensorize(sch.get_loops(B_wmma)[-2], f"wmma_{mma_shape_str}_load_b_shared{name_tail}")

    # sch.reorder(ii, fi, ji)
    sch.tensorize(sch.get_loops(A_wmma)[-2], f"wmma_{mma_shape_str}_load_a{name_tail}")
    sch.tensorize(sch.get_loops(blk_inner)[-3], f"wmma_{mma_shape_str}_sync{name_tail}")


    ax0, ax1 = sch.get_loops(B_shared)[-2:]
    ax = sch.fuse(ax0, ax1)
    # ax0, ax1, ax2 = sch.get_loops(B_shared)[-3:]
    # ax = sch.fuse(ax0, ax1, ax2)
    warp_size = 32
    vector_length = 4
    # warp_size = 32 * 4 # 假设我们在此处设置thread总数为128，看看效果如何
    # vector_length = 2 # 4 配合thread设置成128时，把vectorize设置成2
    ax0, ax1, ax2, ax3 = sch.split(ax, [None, tile_sizes[0][1]//mma_m, warp_size, vector_length])
    sch.unroll(ax0)
    sch.bind(ax1, "threadIdx.y")
    sch.bind(ax2, "threadIdx.x")
    sch.vectorize(ax3)
    # print(sch.mod.script())
    sch.tensorize(
        sch.get_loops(init_blk)[-2], f"wmma_{mma_shape_str}_fill{name_tail}"
    )


    if use_C_shared: # params['idx_reordered'][0] or params['real_atomic']:
        # 默认C_shared只会被用在当i也被reorder了或者需要atomicAdd的情况
        C_shared = sch.get_block(f"C_shared{fmt_i}")
        ax0, ax1 = sch.get_loops(C_shared)[-2:]
        ax = sch.fuse(ax0, ax1)
        warp_size = 32
        # warp_size = 32 * 4
        vector_length = 2
        ax0, ax1, ax2, ax3 = sch.split(ax, [None, tile_sizes[0][1]//mma_m, warp_size, vector_length])
        # sch.unroll(ax0)
        sch.bind(ax1, "threadIdx.y")
        sch.bind(ax2, "threadIdx.x")
        # sch.vectorize(ax2)
        # print(sch.mod.script())



# 这个是在TC op的k轴不reorder，因此我们不会先使用shared memory来load b的情况。
def schedule_tc_knotsorted(op_type, sch, feat_size, func_str, tile_sizes, params, fmt_i):
    '''
    此处的tile_sizes对应的是dbsrmm中的block size, 并不一定是wmma的单位workload, 比如此处的tile size可以是32*32: [i, k]两个维度
    一下的schedule完全参考dbsrmm里的代码。
    '''
    # if not(params['idx_reordered'][0] or params['real_atomic']):
    #     # 此处不需要cache write
    #     schedule_tc_no_cachewrite(sch, feat_size, func_str, tile_sizes, params)
    #     return
    use_C_shared = params['idx_reordered'][0] or params['real_atomic']


    mma_shape_str = params["mma_shape_str"]
    mma_m, mma_n, mma_k = parse_mma_shape(mma_shape_str)

    blk_outer = sch.get_block(f"{func_str}0")

    # (i, foo) = sch.get_loops(blk_outer)
    # ifoo = sch.fuse(i, foo)
    ifoo = None
    if op_type == 'spmm':
        ifoo = sch.fuse(*(sch.get_loops(blk_outer)))
    elif op_type == 'batched_spmm':
        (batch, i, foo) = sch.get_loops(blk_outer)
        ifoo = sch.fuse(batch, i)
        ifoo = sch.fuse(ifoo, foo)     
    sch.bind(ifoo, "blockIdx.x")


    blk_inner = sch.get_block(f"{func_str}1_update")
    blk_inner_oo = sch.get_block(f"{func_str}1_update_oo")
    ko, ii_0, ji_0, ki_0, _ = sch.get_loops(blk_inner_oo)
    ii_1, ki_1, ji_1 = sch.get_loops(blk_inner)

    # sch.unroll(ji_0)
    # sch.bind(ii_0, "threadIdx.y")

    # print(sch.mod.script())

    C_wmma = sch.get_block(f"C_shared_wmma{fmt_i}.accumulator")
    # sch.reverse_compute_at(C_wmma, io)
    # if use_C_shared:
    ax0, ax1 = sch.get_loops(C_wmma)[-2:]
    ax2, ax3 = sch.split(ax1, [None, mma_n])
    ax0, ax1 = sch.split(ax0, [None, mma_m])
    sch.reorder(ax0, ax2, ax1, ax3)
    if use_C_shared:
        sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    # sch.reverse_compute_at(C_shared, io)
    # print("AFTER C_WMMA")
    # print(sch.mod.script())


    A_wmma = sch.get_block(f"A{fmt_i}_wmma.matrix_a")
    ax0, ax1 = sch.get_loops(A_wmma)[-2:]
    ax1, ax2 = sch.split(ax1, [None, mma_k])
    sch.reorder(ax1, ax0, ax2)
    sch.unroll(ax1)
    # print("AFTER A_WMMA")
    # print(sch.mod.script())


    # B_shared = sch.get_block(f"B_shared{fmt_i}")
    B_wmma = sch.get_block(f"B_shared_wmma{fmt_i}.matrix_b")
    
    init_blk = sch.get_block(f"tcspmm{fmt_i}1_init")
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    ax2, ax3 = sch.split(ax1, [None, mma_n])
    ax0, ax1 = sch.split(ax0, [None, mma_m])
    sch.reorder(ax0, ax2, ax1, ax3)
    sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    # sch.reverse_compute_at(C_shared, io)
    # print("AFTER C_WMMA")
    # print(sch.mod.script())


    # sch.hide_buffer_access(blk_inner, "read", [1, 2, 5])
    # sch.tensorize(sch.get_loops(A_wmma)[-2], "wmma_m16n16k16_load_a_shared")
    name_tail = ''
    if os.environ['MydtypeIn'] == 'float':
        name_tail = '_fp32'

    if use_C_shared:
        sch.tensorize(sch.get_loops(C_wmma)[-2], f"wmma_{mma_shape_str}_store_shared{name_tail}")
    else:
        # sch.tensorize(sch.get_loops(C_wmma)[-2], "wmma_m16n16k16_store")
        sch.tensorize(sch.get_loops(C_wmma)[-2], f"wmma_{mma_shape_str}_store{name_tail}")

    sch.tensorize(sch.get_loops(B_wmma)[-2], f"wmma_{mma_shape_str}_load_b{name_tail}")

    # sch.reorder(ii, fi, ji)
    sch.tensorize(sch.get_loops(A_wmma)[-2], f"wmma_{mma_shape_str}_load_a{name_tail}")
    sch.tensorize(sch.get_loops(blk_inner)[-3], f"wmma_{mma_shape_str}_sync{name_tail}")


    # ax0, ax1 = sch.get_loops(B_shared)[-2:]
    # ax = sch.fuse(ax0, ax1)
    # # ax0, ax1, ax2 = sch.get_loops(B_shared)[-3:]
    # # ax = sch.fuse(ax0, ax1, ax2)
    # warp_size = 32
    # vector_length = 4
    # # warp_size = 32 * 4 # 假设我们在此处设置thread总数为128，看看效果如何
    # # vector_length = 2 # 4 配合thread设置成128时，把vectorize设置成2
    # ax0, ax1, ax2, ax3 = sch.split(ax, [None, tile_sizes[0][1]//mma_m, warp_size, vector_length])
    # sch.unroll(ax0)
    # sch.bind(ax1, "threadIdx.y")
    # sch.bind(ax2, "threadIdx.x")
    # sch.vectorize(ax3)
    # print(sch.mod.script())
    sch.tensorize(
        sch.get_loops(init_blk)[-2], f"wmma_{mma_shape_str}_fill{name_tail}"
    )


    if use_C_shared: # params['idx_reordered'][0] or params['real_atomic']:
        # 默认C_shared只会被用在当i也被reorder了或者需要atomicAdd的情况
        C_shared = sch.get_block(f"C_shared{fmt_i}")
        ax0, ax1 = sch.get_loops(C_shared)[-2:]
        ax = sch.fuse(ax0, ax1)
        warp_size = 32
        # warp_size = 32 * 4
        vector_length = 2
        ax0, ax1, ax2, ax3 = sch.split(ax, [None, tile_sizes[0][1]//mma_m, warp_size, vector_length])
        # sch.unroll(ax0)
        sch.bind(ax1, "threadIdx.y")
        sch.bind(ax2, "threadIdx.x")
        # sch.vectorize(ax2)
        # print(sch.mod.script())











# 对于SDDMM的1D tile的schedule
def schedule_1D_SDDMM_noremap_good(op_type, sch, func_str, params):
    '''
    参考sparsetir里面的代码
    '''
    ty = params['ty']
    tx = params['tx']
    vec_size = params['vec_size']
    group_size = params['group_size']

    # schedule compute
    blk = sch.get_block(f"{func_str}0")
    j, k = sch.get_loops(blk)
    ko, kio, kii = sch.split(k, [None, tx, vec_size])
    rf_blk = sch.rfactor(kio, 1) # 这里的factor_axis = 1因为我们的C只有1维
    j = sch.get_loops(rf_blk)[0]
    joo, joi, ji = sch.split(j, [None, ty, group_size])
    sch.bind(joo, "blockIdx.x")
    sch.bind(joi, "threadIdx.y")
    sch.unroll(ji)

    # print(sch.mod.script())

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
    sch.vectorize(ax1) # 这个地方如果要remap的话，可能就vectorize不了了。
    
    # schedule rf
    sch.bind(kio, "threadIdx.x")
    sch.unroll(kii)
    sch.unroll(ko)
    # schedule write back

    # print(sch.mod.script())
    ax0, ax1 = sch.get_loops(blk)[-2:]
    sch.reorder(ax1, ax0)
    sch.bind(ax0, "threadIdx.x")
    # sch.unroll(ax2)
    sch.unroll(ax1)
    # mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    # f = tvm.build(mod["main"], target="cuda")














# 对于SDDMM的1D tile的schedule
# 这个版本尝试把thread y和thread x都fuse在一起.
# 下面这个版本可以成功编译，但是会遇到无法正确生成warp shuffle的代码的问题，所以我们目前的解决方案是使用原来的thread y 和thread x分开的代码先得到一个好的cuda，再修改它，然后再nvcc里面一键替换
# 这个版本没有rfactor，可以避免在生成初始代码的时候遇到的一些bug（比如占用shared memory的名字）
def schedule_1D_SDDMM_noremap(op_type, sch, func_str, params, blk_num):
    '''
    参考sparsetir里面的代码
    '''
    ty = params['ty']
    tx = params['tx']
    vec_size = params['vec_size']
    group_size = params['group_size']

    # schedule compute
    blk = sch.get_block(f"{func_str}0")
    j, k = sch.get_loops(blk)
    # ko, kio, kii = sch.split(k, [None, tx, vec_size])
    # rf_blk = sch.rfactor(kio, 1) # 这里的factor_axis = 1因为我们的C只有1维
    # j = sch.get_loops(rf_blk)[0]
    # joo, joi, ji = sch.split(j, [None, ty, group_size])
    joo, joi, ji = sch.split(j, [blk_num, tx*ty, None])
    sch.bind(joo, "blockIdx.x")
    sch.bind(joi, "threadIdx.x")
    # sch.unroll(ji)

    # print(sch.mod.script())

    # sch.reverse_compute_at(blk, joi, True)
    # sch.reverse_compute_at(blk, joo, True)
    # sch.set_scope(rf_blk, 0, "local")
    # read_A = sch.cache_read(blk, 0, "local")
    # read_B = sch.cache_read(blk, 2, "local")    
    write_C = sch.cache_write(blk, 0, "local")

    # print(sch.mod.script())
    
    # ko, kio, kii = sch.get_loops(rf_blk)[-3:]
    # sch.reorder(ko, ji)
    # sch.reorder(kio, ko, ji)

    # print(sch.mod.script())

    # print("BEFORE SCHEDULE READ A")

    # schedule read A
    # sch.compute_at(read_A, ji, True)

    # ax0, ax1 = sch.split(sch.get_loops(read_A)[-1], [tx, vec_size])
    # sch.bind(ax0, "threadIdx.x")
    # sch.vectorize(ax1)
    # sch.vectorize(sch.get_loops(read_A)[-1])


    # schedule read B
    # sch.compute_at(read_B, ji, True)


    # ax0, ax1 = sch.split(sch.get_loops(read_B)[-1], [tx, vec_size])
    # sch.bind(ax0, "threadIdx.x")
    # sch.vectorize(ax1)
    # sch.vectorize(sch.get_loops(read_B)[-1])


    # print(sch.mod.script())

    # 为了任何情况：是否有TC tile，是否顺序在TC tile之前，param为别的设置，下都能成功编译，我们去掉write C这个stage
    # # schedule write C
    sch.reverse_compute_at(write_C, ji, True)
    # # ax0, ax1 = sch.get_loops(write_C)[-2:]
    # # sch.vectorize(ax1) # 这个地方如果要remap的话，可能就vectorize不了了。
    # sch.bind(sch.get_loops(write_C)[-1], "threadIdx.x")

    # print("AFTER SCHEDULE WRITE C")
    # print(sch.mod.script())


    # sch.bind(sch.fuse(joi, kio), "threadIdx.x")

    # schedule rf
    # sch.bind(kio, "threadIdx.x")
    # sch.unroll(kii)
    # sch.unroll(ko)

    # schedule write back
    # print(sch.mod.script())
    # ax0, ax1 = sch.get_loops(blk)[-2:]  # vk_1, v_j
    # ax1, ax2 = sch.split(ax1, [ty, group_size])
    # sch.reorder(ax2, ax1, ax0)
    # # sch.bind(ax0, "threadIdx.x")
    # # sch.bind(ax1, "threadIdx.y")
    # sch.bind(sch.fuse(ax1, ax0), "threadIdx.x")
    # sch.unroll(ax2)

    # sch.reorder(ax1, ax0)
    # sch.bind(ax0, "threadIdx.x")
    # # sch.unroll(ax2)
    # sch.unroll(ax1)
    # mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    # f = tvm.build(mod["main"], target="cuda")

    print(sch.mod.script())








# 对于SDDMM的1D tile的schedule
# 这个版本是支持write back的时候remap的
def schedule_1D_SDDMM_remap(op_type, sch, func_str, params):
    '''
    参考sparsetir里面的代码
    '''
    ty = params['ty']
    tx = params['tx']
    vec_size = params['vec_size']
    group_size = params['group_size']

    # schedule compute
    blk = sch.get_block(f"{func_str}0")
    j, k = sch.get_loops(blk)
    ko, kio, kii = sch.split(k, [None, tx, vec_size])
    write_C = sch.reverse_cache_write(blk, 0, "local")
    rf_blk = sch.rfactor(kio, 1) # 这里的factor_axis = 1因为我们的C只有1维
    j = sch.get_loops(rf_blk)[0]
    joo, joi, ji = sch.split(j, [None, ty, group_size])
    sch.bind(joo, "blockIdx.x")
    sch.bind(joi, "threadIdx.y")
    sch.unroll(ji)

    print(sch.mod.script())

    sch.reverse_compute_at(blk, joi, True)
    sch.set_scope(rf_blk, 0, "local")
    read_A = sch.cache_read(rf_blk, 1, "local")
    read_B = sch.cache_read(rf_blk, 3, "local")    
    # write_C = sch.cache_write(blk, 0, "local")
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
    # sch.vectorize(ax1) # 这个地方如果要remap的话，可能就vectorize不了了。
    
    # schedule rf
    sch.bind(kio, "threadIdx.x")
    sch.unroll(kii)
    sch.unroll(ko)
    # schedule write back

    print(sch.mod.script())
    ax0, ax1 = sch.get_loops(blk)[-2:]
    sch.reorder(ax1, ax0)
    sch.bind(ax0, "threadIdx.x")
    # sch.unroll(ax2)
    sch.unroll(ax1)
    # mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    # f = tvm.build(mod["main"], target="cuda")




def schedule_1D_SDDMM(op_type, sch, func_str, params, only1D, blk_num):
    if os.environ['REMAP'] == 'True':
        schedule_1D_SDDMM_remap(op_type, sch, func_str, params)
    else:
        if only1D:
            schedule_1D_SDDMM_noremap_good(op_type, sch, func_str, params)
        else:
            schedule_1D_SDDMM_noremap(op_type, sch, func_str, params, blk_num)



# 对于SDDMM的TC tile的schedule
# 这个版本的schedule的一些信息：
#   1. 主要还是参考magiccube里面的写法，每个TC tile对应一个thread block，然后这个thread block里面可以有不同数量的warp，每个warp也不一定就只负责wmma限定的workload。
#   2. 会对A和B都使用shared memory，对A是因为data reuse + 可能的reorder；对B是因为j轴的sparse。
#   3. wmma里面的fragment b必须是column-major的，才能保证结果的正确性。
def schedule_TC_SDDMM(op_type, sch, func_str, tile_sizes, params):

    # print(sch.mod.script())

    # ty = params['ty']
    mma_shape_str = params["mma_shape_str"]
    mma_m, mma_n, mma_k = parse_mma_shape(mma_shape_str)
    warp_num = params['warp_num']
    vec_size = params['vec_size']

    blk = sch.get_block(f"{func_str}0")
    bnum = sch.get_loops(sch.get_block(f"{func_str[:len('tcsddmm')]}_out{func_str[len('tcsddmm'):]}0"))[0]
    m, n, k = sch.get_loops(blk)
    ko, kio, kii = sch.split(k, [None, tile_sizes[1][1]//mma_k, mma_k]) # 因为我们之前调整了SDDMM的idx 顺序，所以k轴的tile size其实应该对应tile_sizes[1]
    # 要不要默认TC tile的row number 一定和mma_m相同？暂时认为是相同的好了。因为magiccube和TC-GNN都是这么设置的。
    n0, n1, n2 = sch.split(n, [None, warp_num, mma_n])

    sch.reorder(n1, ko, kio, n0, m, n2, kii)
    sch.bind(bnum, "blockIdx.x")
    sch.bind(n1, "threadIdx.y")
    # sch.unroll(kio)

    # print(sch.mod.script())

    read_A = sch.reverse_cache_read(blk, 0, "shared")  #此处buffer id还需要确定
    read_B = sch.reverse_cache_read(blk, 2, "shared")  #此处buffer id还需要确定
    A_wmma = sch.reverse_cache_read(blk, 0, "wmma.matrix_a") #此处buffer id还需要确定
    B_wmma = sch.reverse_cache_read(blk, 2, "wmma.matrix_b") #此处buffer id还需要确定
    C_wmma = sch.cache_write(blk, 0, "wmma.accumulator") #此处buffer id还需要确定

    # print(sch.mod.script())

    # sch.compute_at(A_wmma, ko, True)
    # sch.compute_at(read_A, ko, True)
    sch.compute_at(A_wmma, n0, True)
    sch.compute_at(read_A, n0, True)
    # sch.compute_at(A_wmma, kio, True)
    # sch.compute_at(read_A, ko, True)
    sch.compute_at(B_wmma, n0, True)
    sch.compute_at(read_B, n0, True)

    # print(sch.mod.script())

    # schedule read A
    ax = sch.fuse(*(sch.get_loops(read_A)[-2:]))
    ax0, ax1, ax2, ax3 = sch.split(ax, [None, warp_num, 32, vec_size])
    sch.bind(ax1, "threadIdx.y")
    sch.bind(ax2, "threadIdx.x")
    sch.vectorize(ax3)
    # schedule A wmma
    # ax0, ax1 = sch.get_loops(A_wmma)[-2:]
    # ax1, ax2 = sch.split(ax1, [None, mma_k])
    # sch.reorder(ax1, ax0, ax2)

    # schedule read B
    # print(sch.mod.script())
    ax = sch.fuse(*(sch.get_loops(read_B)[-2:]))
    ax0, ax1, ax2 = sch.split(ax, [None, 32, vec_size])
    sch.bind(ax1, "threadIdx.x")
    sch.vectorize(ax2)

    # schedule C wmma
    # print(sch.mod.script())
    ax0, ax1 = sch.get_loops(C_wmma)[-2:]
    ax1, ax2, ax3 = sch.split(ax1, [None, warp_num, mma_n])   
    sch.reorder(ax2, ax1, ax0, ax3)
    sch.bind(ax2, "threadIdx.y")

    # init zeros
    init_blk = sch.decompose_reduction(blk, ko)
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    ax2, ax3 = sch.split(ax1, [None, mma_n])
    ax0, ax1 = sch.split(ax0, [None, mma_m])
    sch.reorder(ax0, ax2, ax1, ax3)
    sch.unroll(ax2)
    # tensorize
    # print(sch.mod.script())
    sch.hide_buffer_access(blk, "read", [2, 4])   # 此处的buffer id还需要确定
    sch.tensorize(sch.get_loops(A_wmma)[-2], "wmma_{}_load_a_shared".format(mma_shape_str))

    if os.environ['REMAP'] == 'True':
        sch.tensorize(sch.get_loops(C_wmma)[-2], "wmma_{}_store_shared".format(mma_shape_str))
    else:
        sch.tensorize(sch.get_loops(C_wmma)[-2], "wmma_{}_store".format(mma_shape_str))

    sch.tensorize(sch.get_loops(B_wmma)[-2], "wmma_{}_load_b_shared_colmajor".format(mma_shape_str))
    sch.tensorize(sch.get_loops(blk)[-3], "wmma_{}_sync_ijk".format(mma_shape_str))
    sch.tensorize(
        sch.get_loops(init_blk)[-2], "wmma_{}_fill".format(mma_shape_str) ) # block name还需要再确定


    if os.environ['REMAP'] != 'True':
        return

    # schedule store block
    blk = sch.get_block(f"{func_str[:len('tcsddmm')]}_store{func_str[len('tcsddmm'):]}0")
    # sch.compute_at(blk, bnum, True)
    # print(sch.mod.script())
    ax = sch.get_loops(blk)[-1]
    ax0, ax1, ax2 = sch.split(ax, [None, warp_num, 32])
    sch.bind(ax1, "threadIdx.y")
    sch.bind(ax2, "threadIdx.x")
    # print(sch.mod.script())


# =============================================
# 在schedule之前，还需要设置必须的input parameter，然后得到初始的sch变量





def gen_definitions_csr(fmt_i, dtype, zerotype):
    parameters_csr = f'''
    a{fmt_i}: T.handle,
    m{fmt_i}: T.int32,
    n{fmt_i}: T.int32,
    nnz{fmt_i}: T.int32,

    indptr_k{fmt_i}: T.handle,
    indices_k{fmt_i}: T.handle,
    indices_reordered_i{fmt_i}: T.handle,
    '''


    idx_definitions_csr = f'''    
    I{fmt_i} = T.dense_fixed(m{fmt_i})
    Krow{fmt_i} = T.dense_fixed(m{fmt_i}+1)
    Knnz{fmt_i} = T.dense_fixed(nnz{fmt_i})
    '''

    buffer_definitions_csr = f'''
    A{fmt_i} = T.match_sparse_buffer(a{fmt_i}, [Knnz{fmt_i}], {dtype})
    K{fmt_i}_indptr = T.match_sparse_buffer(indptr_k{fmt_i}, [Krow{fmt_i}], dtype="int32")
    K{fmt_i}_indices = T.match_sparse_buffer(indices_k{fmt_i}, [Knnz{fmt_i}], dtype="int32")
    I{fmt_i}_indices = T.match_sparse_buffer(indices_reordered_i{fmt_i}, [I{fmt_i}], dtype="int32")
    '''

    comp_statements_csr = f'''
    for o, i, j in T.grid(1, m{fmt_i}, n{fmt_i}):
        with T.block("csrmm{fmt_i}0"):
            vo, vi, vj = T.axis.remap("SSS", [o, i, j])
            # T.reads(K0_indptr[vo, vi : vi + 2], I0_indices[vo, vi], A0[vo, vi, 0 : 128], B[0 : 128, vj], K0_indices[vo, vi, 0 : 128])
            # T.writes(C[I0_indices[vo, vi], vj])
            T.block_attr({{"sparse":True}})
            for ki in T.serial(K{fmt_i}_indptr[vi + 1] - K{fmt_i}_indptr[vi]):
                with T.block("csrmm{fmt_i}1"):
                    vki = T.axis.reduce(k_tot, ki)
                    # T.reads(I0_indices[vo, vi], A0[vo, vi, vki], B[K0_indices[vo, vi, vki], vj], K0_indices[vo, vi, vki])
                    # T.writes(C[I0_indices[vo, vi], vj])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        C[I{fmt_i}_indices[vi], vj] = {zerotype}
                    C[I{fmt_i}_indices[vi], vj] = C[I{fmt_i}_indices[vi], vj] + A{fmt_i}[ K{fmt_i}_indptr[vi]+vki ] * B[K{fmt_i}_indices[K{fmt_i}_indptr[vi]+vki], vj]


    '''

    return parameters_csr, idx_definitions_csr, buffer_definitions_csr, comp_statements_csr









def gen_definitions_ell(fmt_i, dtype, zerotype, params, tile_sizes, dsize):

    parameters_ell = f'''
    a{fmt_i}: T.handle,
    m_num{fmt_i}: T.int32,
    n_num{fmt_i}: T.int32,
    nnz_cols{fmt_i}: T.int32,
    m_blk{fmt_i}: T.int32,
    n_blk{fmt_i}: T.int32,

    indices_k{fmt_i}: T.handle,
    indices_reordered_i{fmt_i}: T.handle,
    '''


    idx_definitions_ell = f'''
    I{fmt_i} = T.dense_fixed(m_num{fmt_i}*m_blk{fmt_i})
    # I_out{fmt_i} = T.sparse_fixed(I{fmt_i}, (m_num{fmt_i}*m_blk{fmt_i}, 1), indices_reordered_i{fmt_i}, idtype="int32", sorted=False)
    # J{fmt_i} = T.dense_fixed(n_num{fmt_i}*n_blk{fmt_i})
    K{fmt_i} = T.dense_fixed(nnz_cols{fmt_i})
    IN{fmt_i} = T.dense_fixed(m_num{fmt_i})
    # JN{fmt_i} = T.dense_fixed(n_num{fmt_i})
    # I_blk{fmt_i} = T.dense_fixed(m_blk{fmt_i})
    # J_blk{fmt_i} = T.dense_fixed(n_blk{fmt_i})
    A_iter{fmt_i} = T.dense_fixed(m_blk{fmt_i}*nnz_cols{fmt_i})
    '''


    buffer_definitions_ell = f'''
    A{fmt_i} = T.match_sparse_buffer(a{fmt_i}, (IN{fmt_i}, A_iter{fmt_i}), {dtype})
    K_out{fmt_i} = T.match_sparse_buffer(indices_k{fmt_i}, [I{fmt_i}, K{fmt_i}], "int32")
    I_out{fmt_i} = T.match_sparse_buffer(indices_reordered_i{fmt_i}, [I{fmt_i}], "int32")
    '''
    if params['real_atomic']:
        buffer_definitions_ell = buffer_definitions_ell + f'''
    C_local{fmt_i} = T.alloc_buffer([1], dtype={dtype}, scope="local")
    '''


    cache_reads_ell = ''

    # ====================================================================
    # about padding SMEM for in1
    pad_pattern = comp_SMEM_padding_pattern('spmm', tile_sizes, dsize)
    # v_iter = 'voffset = T.axis.spatial(1, 0)'
    # if pad_pattern[1] != 0:
    #     v_iter = f'voffset = T.axis.spatial(m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, i{fmt_i}//nnz_cols{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
    #     v_iter = f'voffset = T.axis.spatial(m_blk{fmt_i}*{pad_pattern[1]}, i{fmt_i}//nnz_cols{fmt_i}*{pad_pattern[1]} )'
    # v_iter = f'voffset = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}+m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, i{fmt_i} + i{fmt_i}//nnz_cols{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
    v_iter = ''

    v_iter_read = 'voffset = T.axis.spatial(1, 0)'
    if pad_pattern[1] != 0:
        v_iter_read = f'voffset = T.axis.spatial(m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, iblk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
        # v_iter_read = f'voffset = T.axis.spatial(m_blk{fmt_i}*{pad_pattern[1]}, iblk{fmt_i}*{pad_pattern[1]} )'
    # v_iter_read = f'voffset = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}+ m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, iblk{fmt_i}*nnz_cols{fmt_i}+ki{fmt_i}+iblk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
    # v_iter_read = ''

    A_read_var = f"A{fmt_i}[vinum, viblk * nnz_cols{fmt_i} + vki]"
    if params['SMEM_in1']:
        A_read_var = f"A_shared[viblk * nnz_cols{fmt_i} + vki{' + voffset' if v_iter_read!='' else '' }]"
        # A_read_var = f"A_shared[voffset]"
    C_write_var = f"C[I_out{fmt_i}[vi], vj]"
    if params['real_atomic']:
        C_write_var = f"C_local{fmt_i}[0]"
    # ============

    comp_statements_ell = f'''
    for inum{fmt_i}, jnum{fmt_i} in T.grid(m_num{fmt_i}, n_num{fmt_i}):'''

    if params['SMEM_in1']:
        comp_statements_ell = comp_statements_ell + f'''
        for i{fmt_i} in T.serial(m_blk{fmt_i}*nnz_cols{fmt_i}):
            with T.block("ellmm_shared{fmt_i}0"):
                vinum, vjnum, vi = T.axis.remap("SSS", [inum{fmt_i}, jnum{fmt_i}, i{fmt_i}])
                {v_iter}
                # T.reads(A{fmt_i}[vinum, vi])
                # T.writes(A_shared[vi])
                T.block_attr({{"sparse":True}})
                A_shared[vi{' + voffset' if v_iter!='' else ''}] = A{fmt_i}[vinum, vi]
                # A_shared[voffset] = A{fmt_i}[vinum, vi]'''

    # if params['SMEM_in1']:
    #     comp_statements_ell = comp_statements_ell + f'''
    #     for i_grp{fmt_i} in T.serial({math.prod(tile_sizes[0][1:])//pad_pattern[0]}):
    #         for i_grp_i{fmt_i} in T.serial({pad_pattern[0]}):
    #             for ki{fmt_i} in T.serial(nnz_cols{fmt_i}):
    #                 with T.block("ellmm_shared{fmt_i}0"):
    #                     vinum, vjnum = T.axis.remap("SS", [inum{fmt_i}, jnum{fmt_i}])
    #                     vi = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}, i_grp{fmt_i}*{pad_pattern[0]}*nnz_cols{fmt_i}+i_grp_i{fmt_i}*nnz_cols{fmt_i}+ki{fmt_i})
    #                     voffset = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}+{math.prod(tile_sizes[0][1:])//pad_pattern[0]*pad_pattern[1]}, i_grp{fmt_i}*{pad_pattern[0]}*nnz_cols{fmt_i}+i_grp_i{fmt_i}*nnz_cols{fmt_i}+ki{fmt_i}+i_grp{fmt_i}*{pad_pattern[1]})
    #                     # {v_iter}
    #                     # T.reads(A{fmt_i}[vinum, vi])
    #                     # T.writes(A_shared[vi])
    #                     T.block_attr({{"sparse":True}})
    #                     # A_shared[vi{' + voffset' if v_iter!='' else ''}] = A{fmt_i}[vinum, vi]
    #                     A_shared[voffset] = A{fmt_i}[vinum, vi]'''

    comp_statements_ell = comp_statements_ell + f'''
        for iblk{fmt_i}, jblk{fmt_i} in T.grid(m_blk{fmt_i}, n_blk{fmt_i}):
            for ki{fmt_i} in T.serial(nnz_cols{fmt_i}):
                with T.block("ellmm{fmt_i}0"):
                    {f'vinum = T.axis.spatial(m_num{fmt_i}, inum{fmt_i})' if not params['SMEM_in1'] else ''}
                    vi = T.axis.spatial(m_num{fmt_i}*m_blk{fmt_i}, inum{fmt_i} * m_blk{fmt_i} + iblk{fmt_i})
                    viblk = T.axis.spatial(m_blk{fmt_i}, iblk{fmt_i})
                    {v_iter_read}
                    vki = T.axis.reduce(nnz_cols{fmt_i}, ki{fmt_i})
                    vj = T.axis.spatial(n_num{fmt_i}*n_blk{fmt_i}, jnum{fmt_i} * n_blk{fmt_i} + jblk{fmt_i})
                    vi_out = T.axis.spatial(1, 0)
                    # T.reads(I_out0_indices[vi, vi_out], A_shared[vi % 672 * 16 + vki], B[K_out0[vi, vki], vj], K_out0[vi, vki])
                    # T.writes(C[I_out0_indices[vi, vi_out], vj])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        {C_write_var} = {zerotype}
                    {C_write_var} = {C_write_var} + {A_read_var} * B[K_out{fmt_i}[vi, vki], vj]'''

    if params['real_atomic']:
        comp_statements_ell = comp_statements_ell + f'''
            for ax0, ax1 in T.grid(1, 1):
                with T.block("C_local{fmt_i}"):
                    v0 = T.axis.spatial(m_tot, I_out{fmt_i}[inum{fmt_i} * m_blk{fmt_i} + iblk{fmt_i}] + ax0)
                    v1 = T.axis.spatial(n_tot, jnum{fmt_i} * n_blk{fmt_i} + jblk{fmt_i} + ax1)
                    # T.reads(C_local{fmt_i}[v0, v1])
                    # T.writes(C[v0, v1])
                    T.block_attr({{"sparse":True}})
                    # T.evaluate(T.atomic_add(C.data, v0*n_tot+v1, C_local{fmt_i}[v0, v1]))
                    C[v0, v1] = C_local{fmt_i}[0]
        '''

    return parameters_ell, idx_definitions_ell, buffer_definitions_ell, comp_statements_ell, cache_reads_ell




def gen_definitions_ell_batchspmm(fmt_i, dtype, zerotype, params, tile_sizes, dsize, batch_num):

    parameters_ell = f'''
    a{fmt_i}: T.handle,
    m_num{fmt_i}: T.int32,
    n_num{fmt_i}: T.int32,
    nnz_cols{fmt_i}: T.int32,
    m_blk{fmt_i}: T.int32,
    n_blk{fmt_i}: T.int32,

    indices_k{fmt_i}: T.handle,
    indices_reordered_i{fmt_i}: T.handle,
    '''


    idx_definitions_ell = f'''
    I{fmt_i} = T.dense_fixed(m_num{fmt_i}*m_blk{fmt_i})
    # I_out{fmt_i} = T.sparse_fixed(I{fmt_i}, (m_num{fmt_i}*m_blk{fmt_i}, 1), indices_reordered_i{fmt_i}, idtype="int32", sorted=False)
    # J{fmt_i} = T.dense_fixed(n_num{fmt_i}*n_blk{fmt_i})
    K{fmt_i} = T.dense_fixed(nnz_cols{fmt_i})
    IN{fmt_i} = T.dense_fixed(m_num{fmt_i})
    # JN{fmt_i} = T.dense_fixed(n_num{fmt_i})
    # I_blk{fmt_i} = T.dense_fixed(m_blk{fmt_i})
    # J_blk{fmt_i} = T.dense_fixed(n_blk{fmt_i})
    A_iter{fmt_i} = T.dense_fixed(m_blk{fmt_i}*nnz_cols{fmt_i})
    BATCH{fmt_i} = T.dense_fixed({batch_num})
    '''


    buffer_definitions_ell = f'''
    A{fmt_i} = T.match_sparse_buffer(a{fmt_i}, (BATCH{fmt_i}, IN{fmt_i}, A_iter{fmt_i}), {dtype})
    K_out{fmt_i} = T.match_sparse_buffer(indices_k{fmt_i}, [I{fmt_i}, K{fmt_i}], "int32")
    I_out{fmt_i} = T.match_sparse_buffer(indices_reordered_i{fmt_i}, [I{fmt_i}], "int32")
    '''
    if params['real_atomic']:
        buffer_definitions_ell = buffer_definitions_ell + f'''
    C_local{fmt_i} = T.alloc_buffer([1], dtype={dtype}, scope="local")
    '''


    cache_reads_ell = ''

    # ====================================================================
    # about padding SMEM for in1
    pad_pattern = comp_SMEM_padding_pattern('spmm', tile_sizes, dsize)
    # v_iter = 'voffset = T.axis.spatial(1, 0)'
    # if pad_pattern[1] != 0:
    #     v_iter = f'voffset = T.axis.spatial(m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, i{fmt_i}//nnz_cols{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
    #     v_iter = f'voffset = T.axis.spatial(m_blk{fmt_i}*{pad_pattern[1]}, i{fmt_i}//nnz_cols{fmt_i}*{pad_pattern[1]} )'
    # v_iter = f'voffset = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}+m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, i{fmt_i} + i{fmt_i}//nnz_cols{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
    v_iter = ''

    v_iter_read = 'voffset = T.axis.spatial(1, 0)'
    if pad_pattern[1] != 0:
        v_iter_read = f'voffset = T.axis.spatial(m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, iblk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
        # v_iter_read = f'voffset = T.axis.spatial(m_blk{fmt_i}*{pad_pattern[1]}, iblk{fmt_i}*{pad_pattern[1]} )'
    # v_iter_read = f'voffset = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}+ m_blk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]}, iblk{fmt_i}*nnz_cols{fmt_i}+ki{fmt_i}+iblk{fmt_i}//{pad_pattern[0]}*{pad_pattern[1]} )'
    # v_iter_read = ''

    A_read_var = f"A{fmt_i}[vbatch, vinum, viblk * nnz_cols{fmt_i} + vki]"
    if params['SMEM_in1']:
        A_read_var = f"A_shared[viblk * nnz_cols{fmt_i} + vki{' + voffset' if v_iter_read!='' else '' }]"
        # A_read_var = f"A_shared[voffset]"
    C_write_var = f"C[vbatch, I_out{fmt_i}[vi], vj]"
    if params['real_atomic']:
        C_write_var = f"C_local{fmt_i}[0]"
    # ============

    comp_statements_ell = f'''
    for batch{fmt_i}, inum{fmt_i}, jnum{fmt_i} in T.grid({batch_num}, m_num{fmt_i}, n_num{fmt_i}):'''

    if params['SMEM_in1']:
        comp_statements_ell = comp_statements_ell + f'''
        for i{fmt_i} in T.serial(m_blk{fmt_i}*nnz_cols{fmt_i}):
            with T.block("ellmm_shared{fmt_i}0"):
                vbatch, vinum, vjnum, vi = T.axis.remap("SSSS", [batch{fmt_i}, inum{fmt_i}, jnum{fmt_i}, i{fmt_i}])
                {v_iter}
                # T.reads(A{fmt_i}[vinum, vi])
                # T.writes(A_shared[vi])
                T.block_attr({{"sparse":True}})
                A_shared[vi{' + voffset' if v_iter!='' else ''}] = A{fmt_i}[vbatch, vinum, vi]
                # A_shared[voffset] = A{fmt_i}[vinum, vi]'''

    # if params['SMEM_in1']:
    #     comp_statements_ell = comp_statements_ell + f'''
    #     for i_grp{fmt_i} in T.serial({math.prod(tile_sizes[0][1:])//pad_pattern[0]}):
    #         for i_grp_i{fmt_i} in T.serial({pad_pattern[0]}):
    #             for ki{fmt_i} in T.serial(nnz_cols{fmt_i}):
    #                 with T.block("ellmm_shared{fmt_i}0"):
    #                     vinum, vjnum = T.axis.remap("SS", [inum{fmt_i}, jnum{fmt_i}])
    #                     vi = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}, i_grp{fmt_i}*{pad_pattern[0]}*nnz_cols{fmt_i}+i_grp_i{fmt_i}*nnz_cols{fmt_i}+ki{fmt_i})
    #                     voffset = T.axis.spatial(m_blk{fmt_i}*nnz_cols{fmt_i}+{math.prod(tile_sizes[0][1:])//pad_pattern[0]*pad_pattern[1]}, i_grp{fmt_i}*{pad_pattern[0]}*nnz_cols{fmt_i}+i_grp_i{fmt_i}*nnz_cols{fmt_i}+ki{fmt_i}+i_grp{fmt_i}*{pad_pattern[1]})
    #                     # {v_iter}
    #                     # T.reads(A{fmt_i}[vinum, vi])
    #                     # T.writes(A_shared[vi])
    #                     T.block_attr({{"sparse":True}})
    #                     # A_shared[vi{' + voffset' if v_iter!='' else ''}] = A{fmt_i}[vinum, vi]
    #                     A_shared[voffset] = A{fmt_i}[vinum, vi]'''

    comp_statements_ell = comp_statements_ell + f'''
        for iblk{fmt_i}, jblk{fmt_i} in T.grid(m_blk{fmt_i}, n_blk{fmt_i}):
            for ki{fmt_i} in T.serial(nnz_cols{fmt_i}):
                with T.block("ellmm{fmt_i}0"):
                    {f'vinum = T.axis.spatial(m_num{fmt_i}, inum{fmt_i})' if not params['SMEM_in1'] else ''}
                    vbatch = T.axis.spatial({batch_num}, batch{fmt_i})
                    vi = T.axis.spatial(m_num{fmt_i}*m_blk{fmt_i}, inum{fmt_i} * m_blk{fmt_i} + iblk{fmt_i})
                    viblk = T.axis.spatial(m_blk{fmt_i}, iblk{fmt_i})
                    {v_iter_read}
                    vki = T.axis.reduce(nnz_cols{fmt_i}, ki{fmt_i})
                    vj = T.axis.spatial(n_num{fmt_i}*n_blk{fmt_i}, jnum{fmt_i} * n_blk{fmt_i} + jblk{fmt_i})
                    vi_out = T.axis.spatial(1, 0)
                    # T.reads(I_out0_indices[vi, vi_out], A_shared[vi % 672 * 16 + vki], B[K_out0[vi, vki], vj], K_out0[vi, vki])
                    # T.writes(C[I_out0_indices[vi, vi_out], vj])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        {C_write_var} = {zerotype}
                    {C_write_var} = {C_write_var} + {A_read_var} * B[vbatch, K_out{fmt_i}[vi, vki], vj]'''

    if params['real_atomic']:
        comp_statements_ell = comp_statements_ell + f'''
            for ax0, ax1 in T.grid(1, 1):
                with T.block("C_local{fmt_i}"):
                    v2 = T.axis.spatial({batch_num}, batch{fmt_i})
                    v0 = T.axis.spatial(m_tot, I_out{fmt_i}[inum{fmt_i} * m_blk{fmt_i} + iblk{fmt_i}] + ax0)
                    v1 = T.axis.spatial(n_tot, jnum{fmt_i} * n_blk{fmt_i} + jblk{fmt_i} + ax1)
                    # T.reads(C_local{fmt_i}[v0, v1])
                    # T.writes(C[v0, v1])
                    T.block_attr({{"sparse":True}})
                    # T.evaluate(T.atomic_add(C.data, v0*n_tot+v1, C_local{fmt_i}[v0, v1]))
                    C[v2, v0, v1] = C_local{fmt_i}[0]
        '''

    return parameters_ell, idx_definitions_ell, buffer_definitions_ell, comp_statements_ell, cache_reads_ell












# 暂时完全和TC-SPMM一样，只支持reorder i和k。每次计算一个TC tile的工作量。
# 支持cache write
# 为了能够成功编译，此处我们手写出所有应该有的优化阶段。
def gen_definitions_tc(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params):

    use_C_shared = params['idx_reordered'][0] or params['real_atomic']

    parameters_tc = f'''
    a{fmt_i}: T.handle,
    mb{fmt_i}: T.int32,
    nb{fmt_i}: T.int32,
    nnzb{fmt_i}: T.int32,
    fb{fmt_i}: T.int32,
    funit{fmt_i}: T.int32,
    tile_size{fmt_i}: T.int32,
    group_size{fmt_i}: T.int32,

    iblk_indices{fmt_i}: T.handle,
    indptr{fmt_i}: T.handle,
    indices{fmt_i}: T.handle,
    '''

    idx_definitions_tc = f'''
    IO{fmt_i} = T.dense_fixed(mb{fmt_i})
    # KO{fmt_i} = T.dense_variable(IO{fmt_i}, (nb{fmt_i}, nnzb{fmt_i}), indptr{fmt_i}, "int32")
    II{fmt_i} = T.dense_fixed(tile_size{fmt_i})
    # KI{fmt_i} = T.sparse_fixed(KO{fmt_i}, (nb{fmt_i} * group_size{fmt_i}, group_size{fmt_i}), indices{fmt_i}, "int32", sorted=False)
    KI_dense{fmt_i} = T.dense_fixed(group_size{fmt_i})
    N{fmt_i} = T.dense_fixed(nnzb{fmt_i})
    IOaddOne{fmt_i} = T.dense_fixed(mb{fmt_i} + 1)
    # N_row{fmt_i} = T.dense_fixed(mb{fmt_i} * tile_size{fmt_i})

    # J{fmt_i} = T.dense_fixed(feat_size{fmt_i})

    # JO{fmt_i} = T.dense_fixed(fb{fmt_i})
    # JI{fmt_i} = T.dense_fixed(funit{fmt_i})
    '''

    buffer_definitions_tc = f'''
    A{fmt_i} = T.match_sparse_buffer(a{fmt_i}, [N{fmt_i}, II{fmt_i}, KI_dense{fmt_i}], {dtype})
    KO_indptr{fmt_i} = T.match_sparse_buffer(indptr{fmt_i}, [IOaddOne{fmt_i}], dtype="int32")
    KI_indices{fmt_i} = T.match_sparse_buffer(indices{fmt_i}, [N{fmt_i}, KI_dense{fmt_i}], dtype="int32")
    # I_indices{fmt_i} = T.match_sparse_buffer(iblk_indices{fmt_i}, [N{fmt_i}], dtype="int32")
    I_indices{fmt_i} = T.match_sparse_buffer(iblk_indices{fmt_i}, [IO{fmt_i}{f', II{fmt_i}' if use_C_shared else ''}], dtype="int32")

    # T.assume_buffer_domain(KO_indptr{fmt_i}, [0, nnzb{fmt_i}])
    '''


    # TODO: 不确定这个地方的C的layout改写成 两个维度行不行？试试
    comp_statements_tc0 = f'''
    for io, jo in T.grid(mb{fmt_i}, fb{fmt_i}):
        with T.block("tcspmm{fmt_i}0"):
            vio, vjo = T.axis.remap("SS", [io, jo])
            T.block_attr({{"sparse":True}})
            for ko, ii, ki, ji in T.grid(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], tile_size{fmt_i}, group_size{fmt_i}, funit{fmt_i}):
                with T.block("tcspmm{fmt_i}1"):
                    vko = T.axis.reduce(nnzb{fmt_i}, ko)
                    vk = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + vko)
                    vii = T.axis.spatial(tile_size{fmt_i}, ii)
                    vki = T.axis.reduce(group_size{fmt_i}, ki)
                    vji = T.axis.spatial(funit{fmt_i}, ji)
                    # vko, vii, vki, vji = T.axis.remap("RSRS", [ko, ii, ki, ji])
                    # vii, vki, vji = T.axis.remap("SRS", [ii, ki, ji])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = {zerotype}
                    # C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]
                    C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[vk, vii, vki] * B[KI_indices{fmt_i}[vk, vki], vjo*funit{fmt_i}+vji]

    '''
    # C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] = {zerotype}
    #                 C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]


    comp_statements_tc = f'''
    for io, jo in T.grid(mb{fmt_i}, fb{fmt_i}):
        with T.block("tcspmm{fmt_i}0"):
            vio, vjo = T.axis.remap("SS", [io, jo])
            T.block_attr({{"sparse":True}})
            {f'C_shared{fmt_i} = T.alloc_buffer([tile_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="shared")' if use_C_shared else ''}
            C_shared_wmma_accumulator{fmt_i} = T.alloc_buffer([tile_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="wmma.accumulator")
            A{fmt_i}_wmma_matrix_a = T.alloc_buffer([nnzb{fmt_i}, tile_size{fmt_i}, group_size{fmt_i}], dtype={dtype}, scope="wmma.matrix_a")
            B_shared{fmt_i} = T.alloc_buffer([group_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="shared")
            B_shared_wmma_matrix_b{fmt_i} = T.alloc_buffer([group_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="wmma.matrix_b")
            for ii, ji in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("tcspmm{fmt_i}1_init"):
                    vii, vji = T.axis.remap("SS", [ii, ji])
                    T.block_attr({{"sparse":True}})
                    C_shared_wmma_accumulator{fmt_i}[vii, vji] = {zerotype}
            for ko in T.serial(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio]):
                for ii_0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
                    # for ax_00 in T.unroll(1):
                    #     with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                    #         v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                    #         for ax0, ax1 in T.grid({mma_m}, group_size{fmt_i}):
                    #             with T.block("A{fmt_i}_wmma.matrix_a"):
                    #                 v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ax0)
                    #                 v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                    #                 T.block_attr({{"sparse":True}})
                    #                 A{fmt_i}_wmma_matrix_a[v0, v1, v2] = A{fmt_i}[v0, v1, v2]
                    for ax1_0 in T.unroll(group_size{fmt_i} // {mma_k}):
                        with T.block("A{fmt_i}_wmma.matrix_a_ooo"):
                            v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                            v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                            v2 = T.axis.spatial(group_size{fmt_i}, ax1_0 * {mma_k})
                            for ax1_00 in T.unroll(1):
                                with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                                    for ax0, ax1_1 in T.grid({mma_m}, {mma_k}):
                                        with T.block("A{fmt_i}_wmma.matrix_a"):
                                            v3, v4 = T.axis.remap("SS", [ax0, ax1_1])
                                            # v1 = T.axis.spatial({mma_m}, ii_0 * {mma_m} + ax0)
                                            # v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                                            T.block_attr({{"sparse":True}})
                                            A{fmt_i}_wmma_matrix_a[v0, v1+v3, v2+v4] = A{fmt_i}[v0, v1+v3, v2+v4]
                    for ji_0 in T.serial(funit{fmt_i} // {mma_n}):
                        for ax0, ax1 in T.grid(group_size{fmt_i}, {mma_n}):
                            with T.block("B_shared{fmt_i}"):
                                v0, v1 = T.axis.remap("SS", [ko, ax0])
                                v2 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ax1)
                                T.block_attr({{"sparse":True}})
                                B_shared{fmt_i}[v1, v2] = B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + v0, v1], vjo * funit{fmt_i} + v2]
                        for ki_0 in T.serial(group_size{fmt_i} // {mma_k}):
                            for ax0, ax1 in T.grid({mma_k}, {mma_n}):
                                with T.block("B_shared_wmma{fmt_i}.matrix_b"):
                                    # v0 = T.axis.spatial(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                                    v1 = T.axis.spatial(group_size{fmt_i}, ki_0 * {mma_k} + ax0)
                                    v2 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ax1)
                                    T.block_attr({{"sparse":True}})
                                    B_shared_wmma_matrix_b{fmt_i}[v1, v2] = B_shared{fmt_i}[v1, v2]
                            for ax_00 in T.unroll(1):
                                with T.block("tcspmm{fmt_i}1_update_oo"):
                                    v1 = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                                    v2 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                                    v3 = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k})
                                    v4 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n})
                                    for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
                                        with T.block("tcspmm{fmt_i}1_update"):
                                            # vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                                            # vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
                                            # vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
                                            # vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
                                            vii = T.axis.spatial({mma_m}, ii_1)
                                            vki = T.axis.reduce({mma_k}, ki_1)
                                            vji = T.axis.spatial({mma_n}, ji_1)
                                            T.block_attr({{"sparse":True}})
                                            C_shared_wmma_accumulator{fmt_i}[v2+vii, v4+vji] = C_shared_wmma_accumulator{fmt_i}[v2+vii, v4+vji] + A{fmt_i}_wmma_matrix_a[v1, v2+vii, v3+vki] * B_shared_wmma_matrix_b{fmt_i}[v3+vki, v4+vji]
                            # 
                            # for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
                            #     with T.block("tcspmm{fmt_i}1_update"):
                            #         vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                            #         vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
                            #         vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
                            #         vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
                            #         T.block_attr({{"sparse":True}})
                            #         C_shared_wmma_accumulator{fmt_i}[vii, vji] = C_shared_wmma_accumulator{fmt_i}[vii, vji] + A{fmt_i}_wmma_matrix_a[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B_shared_wmma_matrix_b{fmt_i}[vki, vji]
            # for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
            #     with T.block("C_shared_wmma{fmt_i}.accumulator"):
            #         v0, v1 = T.axis.remap("SS", [ax0, ax1])
            #         T.block_attr({{"sparse":True}})
            #         C_shared{fmt_i}[v0, v1] = C_shared_wmma_accumulator{fmt_i}[v0, v1]
            # for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
            #     with T.block("C_shared{fmt_i}"):
            #         v0, v1 = T.axis.remap("SS", [ax0, ax1])
            #         T.block_attr({{"sparse":True}})
            #         C[I_indices{fmt_i}[vio, v0], vjo * funit{fmt_i} + v1] = C_shared{fmt_i}[v0, v1]
    '''
    
    if use_C_shared:
        comp_statements_tc = comp_statements_tc + f'''
            for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("C_shared_wmma{fmt_i}.accumulator"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.block_attr({{"sparse":True}})
                    C_shared{fmt_i}[v0, v1] = C_shared_wmma_accumulator{fmt_i}[v0, v1]
            for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("C_shared{fmt_i}"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.block_attr({{"sparse":True}})
                    C[I_indices{fmt_i}[vio, v0], vjo * funit{fmt_i} + v1] = C_shared{fmt_i}[v0, v1]
    '''
    else:
        comp_statements_tc = comp_statements_tc + f'''
            # for ax0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
            #     for ax1 in T.serial(funit{fmt_i} // {mma_n}):
            #         with T.block("C_shared_wmma{fmt_i}.accumulator_ooo"):
            #             v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i} + ax0 * {mma_m})
            #             v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ax1 * {mma_n})
            #             for ax000 in T.unroll(1):
            #                 with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
            #                     for ax2, ax3 in T.grid({mma_m}, {mma_n}):
            #                         with T.block("C_shared_wmma{fmt_i}.accumulator"):
            #                             vi, vj = T.axis.remap("SS", [ax2, ax3])
            #                             T.block_attr({{"sparse":True}})
            #                             C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
            for ax000 in T.unroll(1):
                with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
                    v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
                    v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i})                  
                    for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                        with T.block("C_shared_wmma{fmt_i}.accumulator"):
                            vi, vj = T.axis.remap("SS", [ax0, ax1])
                            T.block_attr({{"sparse":True}})
                            C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
    '''

    #     comp_statements_tc = f'''
    # for io, jo in T.grid(mb{fmt_i}, fb{fmt_i}):
    #     with T.block("tcspmm{fmt_i}0"):
    #         vio, vjo = T.axis.remap("SS", [io, jo])
    #         T.block_attr({{"sparse":True}})
    #         {f'C_shared{fmt_i} = T.alloc_buffer([m_tot, n_tot], dtype={dtype}, scope="shared")' if use_C_shared else ''}
    #         C_shared_wmma_accumulator{fmt_i} = T.alloc_buffer([m_tot, n_tot], dtype={dtype}, scope="wmma.accumulator")
    #         A{fmt_i}_wmma_matrix_a = T.alloc_buffer([nnzb{fmt_i}, tile_size{fmt_i}, group_size{fmt_i}], dtype={dtype}, scope="wmma.matrix_a")
    #         B_shared{fmt_i} = T.alloc_buffer([k_tot, n_tot], dtype={dtype}, scope="shared")
    #         B_shared_wmma_matrix_b{fmt_i} = T.alloc_buffer([k_tot, n_tot], dtype={dtype}, scope="wmma.matrix_b")
    #         for ax in T.unroll(1):
    #             with T.block("tcspmm{fmt_i}1_init_oo"):
    #                 v0 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
    #                 v1 = T.axis.spatial(n_tot, vjo * funit{fmt_i})
    #                 for ii, ji in T.grid(tile_size{fmt_i}, funit{fmt_i}):
    #                     with T.block("tcspmm{fmt_i}1_init"):
    #                         vii, vji = T.axis.remap("SS", [ii, ji])
    #                         # v2 = T.axis.spatial(m_tot, v0 + ii)
    #                         # v3 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ji)
    #                         T.block_attr({{"sparse":True}})
    #                         C_shared_wmma_accumulator{fmt_i}[v0 + vii, v1 + vji] = {zerotype}
    #         for ko in T.serial(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio]):
    #             for ii_0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
    #                 # for ax_00 in T.unroll(1):
    #                 #     with T.block("A{fmt_i}_wmma.matrix_a_oo"):
    #                 #         v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
    #                 #         for ax0, ax1 in T.grid({mma_m}, group_size{fmt_i}):
    #                 #             with T.block("A{fmt_i}_wmma.matrix_a"):
    #                 #                 v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ax0)
    #                 #                 v2 = T.axis.spatial(group_size{fmt_i}, ax1)
    #                 #                 T.block_attr({{"sparse":True}})
    #                 #                 A{fmt_i}_wmma_matrix_a[v0, v1, v2] = A{fmt_i}[v0, v1, v2]
    #                 for ax1_0 in T.unroll(group_size{fmt_i} // {mma_k}):
    #                     with T.block("A{fmt_i}_wmma.matrix_a_ooo"):
    #                         v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
    #                         v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
    #                         v2 = T.axis.spatial(group_size{fmt_i}, ax1_0 * {mma_k})
    #                         for ax1_00 in T.unroll(1):
    #                             with T.block("A{fmt_i}_wmma.matrix_a_oo"):
    #                                 for ax0, ax1_1 in T.grid({mma_m}, {mma_k}):
    #                                     with T.block("A{fmt_i}_wmma.matrix_a"):
    #                                         v3, v4 = T.axis.remap("SS", [ax0, ax1_1])
    #                                         # v1 = T.axis.spatial({mma_m}, ii_0 * {mma_m} + ax0)
    #                                         # v2 = T.axis.spatial(group_size{fmt_i}, ax1)
    #                                         T.block_attr({{"sparse":True}})
    #                                         A{fmt_i}_wmma_matrix_a[v0, v1+v3, v2+v4] = A{fmt_i}[v0, v1+v3, v2+v4]
    #                 for ji_0 in T.serial(funit{fmt_i} // {mma_n}):
    #                     for ax0, ax1 in T.grid(group_size{fmt_i}, {mma_n}):
    #                         with T.block("B_shared{fmt_i}"):
    #                             # v0, v1 = T.axis.remap("SS", [ko, ax0])
    #                             # 
    #                             v0 = T.axis.spatial(k_tot, KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + ko, ax0])
    #                             v1 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ji_0 * {mma_n} + ax1)
    #                             T.block_attr({{"sparse":True}})
    #                             B_shared{fmt_i}[v0, v1] = B[v0, v1]
    #                     for ki_0 in T.serial(group_size{fmt_i} // {mma_k}):
    #                         for ax_00 in T.unroll(1):
    #                             with T.block("B_shared_wmma{fmt_i}.matrix_b_oo"):
    #                                 v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
    #                                 for ax0, ax1 in T.grid({mma_k}, {mma_n}):
    #                                     with T.block("B_shared_wmma{fmt_i}.matrix_b"):
    #                                         v1 = T.axis.spatial(k_tot, KI_indices{fmt_i}[v0, ki_0 * {mma_k} + ax0])
    #                                         v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ji_0 * {mma_n} + ax1)
    #                                         T.block_attr({{"sparse":True}})
    #                                         B_shared_wmma_matrix_b{fmt_i}[v1, v2] = B_shared{fmt_i}[v1, v2]
    #                         for ax_00 in T.unroll(1):
    #                             with T.block("tcspmm{fmt_i}1_update_oo"):
    #                                 v1 = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
    #                                 v2 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
    #                                 v3 = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k})
    #                                 v4 = T.axis.spatial(funit{fmt_i}, vjo * funit{fmt_i} + ji_0 * {mma_n})
    #                                 v5 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
    #                                 for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
    #                                     with T.block("tcspmm{fmt_i}1_update"):
    #                                         # vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
    #                                         # vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
    #                                         # vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
    #                                         # vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
    #                                         vii = T.axis.spatial({mma_m}, ii_1)
    #                                         vki = T.axis.reduce({mma_k}, ki_1)
    #                                         vji = T.axis.spatial({mma_n}, ji_1)
    #                                         T.block_attr({{"sparse":True}})
    #                                         C_shared_wmma_accumulator{fmt_i}[v2+v5+vii, v4+vji] = C_shared_wmma_accumulator{fmt_i}[v2+v5+vii, v4+vji] + A{fmt_i}_wmma_matrix_a[v1, v5+vii, v3+vki] * B_shared_wmma_matrix_b{fmt_i}[KI_indices{fmt_i}[v1, v3+vki], v4+vji]
    #                         # 
    #         # for ax000 in T.unroll(1):
    #         #     with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
    #         #         v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
    #         #         v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i})                  
    #         #         for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
    #         #             with T.block("C_shared_wmma{fmt_i}.accumulator"):
    #         #                 vi, vj = T.axis.remap("SS", [ax0, ax1])
    #         #                 T.block_attr({{"sparse":True}})
    #         #                 C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]

    #         for ax0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
    #             for ax1 in T.unroll(funit{fmt_i} // {mma_n}):
    #                 with T.block("C_shared_wmma{fmt_i}.accumulator_ooo"):
    #                     v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i} + ax0 * {mma_m})
    #                     v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ax1 * {mma_n})
    #                     for ax000 in T.unroll(1):
    #                         with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
    #                             for ax2, ax3 in T.grid({mma_m}, {mma_n}):
    #                                 with T.block("C_shared_wmma{fmt_i}.accumulator"):
    #                                     vi, vj = T.axis.remap("SS", [ax2, ax3])
    #                                     # vi = T.axis.spatial(m_tot, v1 + ax2)
    #                                     # vj = T.axis.spatial(n_tot, v2 + ax3)
    #                                     T.block_attr({{"sparse":True}})
    #                                     C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[v1 + vi, v2 + vj]
    # '''


        # 如果在i轴没有reorder或者为使用atomicAdd，那区别就是C的index表达式不同
    #     comp_statements_tc = comp_statements_tc + f'''
    #         for ax0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
    #             for ax1 in T.serial(funit{fmt_i} // {mma_n}):
    #                 with T.block("C_shared_wmma{fmt_i}.accumulator_ooo"):
    #                     v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i} + ax0 * {mma_m})
    #                     v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ax1 * {mma_n})
    #                     for ax000 in T.unroll(1):
    #                         with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
    #                             for ax2, ax3 in T.grid({mma_m}, {mma_n}):
    #                                 with T.block("C_shared_wmma{fmt_i}.accumulator"):
    #                                     vi, vj = T.axis.remap("SS", [ax2, ax3])
    #                                     T.block_attr({{"sparse":True}})
    #                                     C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
    #         # for ax000 in T.unroll(1):
    #         #     with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
    #         #         v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
    #         #         v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i})                  
    #         #         for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
    #         #             with T.block("C_shared_wmma{fmt_i}.accumulator"):
    #         #                 vi, vj = T.axis.remap("SS", [ax0, ax1])
    #         #                 T.block_attr({{"sparse":True}})
    #         #                 C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
    # '''


    #     comp_statements_tc = f'''
    # for io, jo in T.grid(mb{fmt_i}, fb{fmt_i}):
    #     with T.block("tcspmm{fmt_i}0"):
    #         vio, vjo = T.axis.remap("SS", [io, jo])
    #         T.block_attr({{"sparse":True}})
    #         for ko, ii, ki, ji in T.grid(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], tile_size{fmt_i}, group_size{fmt_i}, funit{fmt_i}):
    #             with T.block("tcspmm{fmt_i}1"):
    #                 vko, vii, vki, vji = T.axis.remap("RSRS", [ko, ii, ki, ji])
    #                 T.block_attr({{"sparse":True}})
    #                 with T.init():
    #                     C[I_indices{fmt_i}[vio]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = {zerotype}
    #                 C[I_indices{fmt_i}[vio]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]

    # '''

    return parameters_tc, idx_definitions_tc, buffer_definitions_tc, comp_statements_tc






def gen_definitions_tc_batchspmm(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params, batch_num):

    use_C_shared = params['idx_reordered'][0] or params['real_atomic']

    parameters_tc = f'''
    a{fmt_i}: T.handle,
    mb{fmt_i}: T.int32,
    nb{fmt_i}: T.int32,
    nnzb{fmt_i}: T.int32,
    fb{fmt_i}: T.int32,
    funit{fmt_i}: T.int32,
    tile_size{fmt_i}: T.int32,
    group_size{fmt_i}: T.int32,

    iblk_indices{fmt_i}: T.handle,
    indptr{fmt_i}: T.handle,
    indices{fmt_i}: T.handle,
    '''

    idx_definitions_tc = f'''
    IO{fmt_i} = T.dense_fixed(mb{fmt_i})
    # KO{fmt_i} = T.dense_variable(IO{fmt_i}, (nb{fmt_i}, nnzb{fmt_i}), indptr{fmt_i}, "int32")
    II{fmt_i} = T.dense_fixed(tile_size{fmt_i})
    # KI{fmt_i} = T.sparse_fixed(KO{fmt_i}, (nb{fmt_i} * group_size{fmt_i}, group_size{fmt_i}), indices{fmt_i}, "int32", sorted=False)
    KI_dense{fmt_i} = T.dense_fixed(group_size{fmt_i})
    N{fmt_i} = T.dense_fixed(nnzb{fmt_i})
    IOaddOne{fmt_i} = T.dense_fixed(mb{fmt_i} + 1)
    # N_row{fmt_i} = T.dense_fixed(mb{fmt_i} * tile_size{fmt_i})

    # J{fmt_i} = T.dense_fixed(feat_size{fmt_i})

    # JO{fmt_i} = T.dense_fixed(fb{fmt_i})
    # JI{fmt_i} = T.dense_fixed(funit{fmt_i})

    BATCH{fmt_i} = T.dense_fixed({batch_num})
    '''

    buffer_definitions_tc = f'''
    A{fmt_i} = T.match_sparse_buffer(a{fmt_i}, [BATCH{fmt_i}, N{fmt_i}, II{fmt_i}, KI_dense{fmt_i}], {dtype})
    KO_indptr{fmt_i} = T.match_sparse_buffer(indptr{fmt_i}, [IOaddOne{fmt_i}], dtype="int32")
    KI_indices{fmt_i} = T.match_sparse_buffer(indices{fmt_i}, [N{fmt_i}, KI_dense{fmt_i}], dtype="int32")
    # I_indices{fmt_i} = T.match_sparse_buffer(iblk_indices{fmt_i}, [N{fmt_i}], dtype="int32")
    I_indices{fmt_i} = T.match_sparse_buffer(iblk_indices{fmt_i}, [IO{fmt_i}{f', II{fmt_i}' if use_C_shared else ''}], dtype="int32")

    # T.assume_buffer_domain(KO_indptr{fmt_i}, [0, nnzb{fmt_i}])
    '''


    # TODO: 不确定这个地方的C的layout改写成 两个维度行不行？试试
    comp_statements_tc0 = f'''
    for batch, io, jo in T.grid({batch_num}, mb{fmt_i}, fb{fmt_i}):
        with T.block("tcspmm{fmt_i}0"):
            vio, vjo = T.axis.remap("SS", [io, jo])
            T.block_attr({{"sparse":True}})
            for ko, ii, ki, ji in T.grid(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], tile_size{fmt_i}, group_size{fmt_i}, funit{fmt_i}):
                with T.block("tcspmm{fmt_i}1"):
                    vko = T.axis.reduce(nnzb{fmt_i}, ko)
                    vk = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + vko)
                    vii = T.axis.spatial(tile_size{fmt_i}, ii)
                    vki = T.axis.reduce(group_size{fmt_i}, ki)
                    vji = T.axis.spatial(funit{fmt_i}, ji)
                    # vko, vii, vki, vji = T.axis.remap("RSRS", [ko, ii, ki, ji])
                    # vii, vki, vji = T.axis.remap("SRS", [ii, ki, ji])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = {zerotype}
                    # C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]
                    C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[vk, vii, vki] * B[KI_indices{fmt_i}[vk, vki], vjo*funit{fmt_i}+vji]

    '''
    # C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] = {zerotype}
    #                 C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]


    comp_statements_tc = f'''
    for batch, io, jo in T.grid({batch_num}, mb{fmt_i}, fb{fmt_i}):
        with T.block("tcspmm{fmt_i}0"):
            vbatch, vio, vjo = T.axis.remap("SSS", [batch, io, jo])
            T.block_attr({{"sparse":True}})
            {f'C_shared{fmt_i} = T.alloc_buffer([tile_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="shared")' if use_C_shared else ''}
            C_shared_wmma_accumulator{fmt_i} = T.alloc_buffer([tile_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="wmma.accumulator")
            A{fmt_i}_wmma_matrix_a = T.alloc_buffer([{batch_num}, nnzb{fmt_i}, tile_size{fmt_i}, group_size{fmt_i}], dtype={dtype}, scope="wmma.matrix_a")
            B_shared{fmt_i} = T.alloc_buffer([group_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="shared")
            B_shared_wmma_matrix_b{fmt_i} = T.alloc_buffer([group_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="wmma.matrix_b")
            for ii, ji in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("tcspmm{fmt_i}1_init"):
                    vii, vji = T.axis.remap("SS", [ii, ji])
                    T.block_attr({{"sparse":True}})
                    C_shared_wmma_accumulator{fmt_i}[vii, vji] = {zerotype}
            for ko in T.serial(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio]):
                for ii_0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
                    # for ax_00 in T.unroll(1):
                    #     with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                    #         v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                    #         for ax0, ax1 in T.grid({mma_m}, group_size{fmt_i}):
                    #             with T.block("A{fmt_i}_wmma.matrix_a"):
                    #                 v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ax0)
                    #                 v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                    #                 T.block_attr({{"sparse":True}})
                    #                 A{fmt_i}_wmma_matrix_a[v0, v1, v2] = A{fmt_i}[v0, v1, v2]
                    for ax1_0 in T.unroll(group_size{fmt_i} // {mma_k}):
                        with T.block("A{fmt_i}_wmma.matrix_a_ooo"):
                            v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                            v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                            v2 = T.axis.spatial(group_size{fmt_i}, ax1_0 * {mma_k})
                            for ax1_00 in T.unroll(1):
                                with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                                    for ax0, ax1_1 in T.grid({mma_m}, {mma_k}):
                                        with T.block("A{fmt_i}_wmma.matrix_a"):
                                            v3, v4 = T.axis.remap("SS", [ax0, ax1_1])
                                            # v1 = T.axis.spatial({mma_m}, ii_0 * {mma_m} + ax0)
                                            # v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                                            T.block_attr({{"sparse":True}})
                                            A{fmt_i}_wmma_matrix_a[vbatch, v0, v1+v3, v2+v4] = A{fmt_i}[vbatch, v0, v1+v3, v2+v4]
                    for ji_0 in T.serial(funit{fmt_i} // {mma_n}):
                        for ax0, ax1 in T.grid(group_size{fmt_i}, {mma_n}):
                            with T.block("B_shared{fmt_i}"):
                                v0, v1 = T.axis.remap("SS", [ko, ax0])
                                v2 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ax1)
                                T.block_attr({{"sparse":True}})
                                B_shared{fmt_i}[v1, v2] = B[vbatch, KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + v0, v1], vjo * funit{fmt_i} + v2]
                        for ki_0 in T.serial(group_size{fmt_i} // {mma_k}):
                            for ax0, ax1 in T.grid({mma_k}, {mma_n}):
                                with T.block("B_shared_wmma{fmt_i}.matrix_b"):
                                    # v0 = T.axis.spatial(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                                    v1 = T.axis.spatial(group_size{fmt_i}, ki_0 * {mma_k} + ax0)
                                    v2 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ax1)
                                    T.block_attr({{"sparse":True}})
                                    B_shared_wmma_matrix_b{fmt_i}[v1, v2] = B_shared{fmt_i}[v1, v2]
                            for ax_00 in T.unroll(1):
                                with T.block("tcspmm{fmt_i}1_update_oo"):
                                    v1 = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                                    v2 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                                    v3 = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k})
                                    v4 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n})
                                    for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
                                        with T.block("tcspmm{fmt_i}1_update"):
                                            # vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                                            # vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
                                            # vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
                                            # vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
                                            vii = T.axis.spatial({mma_m}, ii_1)
                                            vki = T.axis.reduce({mma_k}, ki_1)
                                            vji = T.axis.spatial({mma_n}, ji_1)
                                            T.block_attr({{"sparse":True}})
                                            C_shared_wmma_accumulator{fmt_i}[v2+vii, v4+vji] = C_shared_wmma_accumulator{fmt_i}[v2+vii, v4+vji] + A{fmt_i}_wmma_matrix_a[vbatch, v1, v2+vii, v3+vki] * B_shared_wmma_matrix_b{fmt_i}[v3+vki, v4+vji]
                            # 
                            # for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
                            #     with T.block("tcspmm{fmt_i}1_update"):
                            #         vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                            #         vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
                            #         vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
                            #         vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
                            #         T.block_attr({{"sparse":True}})
                            #         C_shared_wmma_accumulator{fmt_i}[vii, vji] = C_shared_wmma_accumulator{fmt_i}[vii, vji] + A{fmt_i}_wmma_matrix_a[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B_shared_wmma_matrix_b{fmt_i}[vki, vji]
            # for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
            #     with T.block("C_shared_wmma{fmt_i}.accumulator"):
            #         v0, v1 = T.axis.remap("SS", [ax0, ax1])
            #         T.block_attr({{"sparse":True}})
            #         C_shared{fmt_i}[v0, v1] = C_shared_wmma_accumulator{fmt_i}[v0, v1]
            # for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
            #     with T.block("C_shared{fmt_i}"):
            #         v0, v1 = T.axis.remap("SS", [ax0, ax1])
            #         T.block_attr({{"sparse":True}})
            #         C[I_indices{fmt_i}[vio, v0], vjo * funit{fmt_i} + v1] = C_shared{fmt_i}[v0, v1]
    '''
    
    if use_C_shared:
        comp_statements_tc = comp_statements_tc + f'''
            for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("C_shared_wmma{fmt_i}.accumulator"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.block_attr({{"sparse":True}})
                    C_shared{fmt_i}[v0, v1] = C_shared_wmma_accumulator{fmt_i}[v0, v1]
            for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("C_shared{fmt_i}"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.block_attr({{"sparse":True}})
                    C[vbatch, I_indices{fmt_i}[vio, v0], vjo * funit{fmt_i} + v1] = C_shared{fmt_i}[v0, v1]
    '''
    else:
        comp_statements_tc = comp_statements_tc + f'''
            # for ax0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
            #     for ax1 in T.serial(funit{fmt_i} // {mma_n}):
            #         with T.block("C_shared_wmma{fmt_i}.accumulator_ooo"):
            #             v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i} + ax0 * {mma_m})
            #             v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ax1 * {mma_n})
            #             for ax000 in T.unroll(1):
            #                 with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
            #                     for ax2, ax3 in T.grid({mma_m}, {mma_n}):
            #                         with T.block("C_shared_wmma{fmt_i}.accumulator"):
            #                             vi, vj = T.axis.remap("SS", [ax2, ax3])
            #                             T.block_attr({{"sparse":True}})
            #                             C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
            for ax000 in T.unroll(1):
                with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
                    v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
                    v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i})                  
                    for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                        with T.block("C_shared_wmma{fmt_i}.accumulator"):
                            vi, vj = T.axis.remap("SS", [ax0, ax1])
                            T.block_attr({{"sparse":True}})
                            C[vbatch, v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
    '''

    return parameters_tc, idx_definitions_tc, buffer_definitions_tc, comp_statements_tc





# 这个版本的定义用于 k轴没有被sorted的情况，也就不需要b shared了
# load B 无法tensorize
def gen_definitions_tc_batchspmm_knotsorted(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params, batch_num):

    use_C_shared = params['idx_reordered'][0] or params['real_atomic']

    parameters_tc = f'''
    a{fmt_i}: T.handle,
    mb{fmt_i}: T.int32,
    nb{fmt_i}: T.int32,
    nnzb{fmt_i}: T.int32,
    fb{fmt_i}: T.int32,
    funit{fmt_i}: T.int32,
    tile_size{fmt_i}: T.int32,
    group_size{fmt_i}: T.int32,

    iblk_indices{fmt_i}: T.handle,
    indptr{fmt_i}: T.handle,
    indices{fmt_i}: T.handle,
    '''

    idx_definitions_tc = f'''
    IO{fmt_i} = T.dense_fixed(mb{fmt_i})
    # KO{fmt_i} = T.dense_variable(IO{fmt_i}, (nb{fmt_i}, nnzb{fmt_i}), indptr{fmt_i}, "int32")
    II{fmt_i} = T.dense_fixed(tile_size{fmt_i})
    # KI{fmt_i} = T.sparse_fixed(KO{fmt_i}, (nb{fmt_i} * group_size{fmt_i}, group_size{fmt_i}), indices{fmt_i}, "int32", sorted=False)
    KI_dense{fmt_i} = T.dense_fixed(group_size{fmt_i})
    N{fmt_i} = T.dense_fixed(nnzb{fmt_i})
    IOaddOne{fmt_i} = T.dense_fixed(mb{fmt_i} + 1)
    # N_row{fmt_i} = T.dense_fixed(mb{fmt_i} * tile_size{fmt_i})

    # J{fmt_i} = T.dense_fixed(feat_size{fmt_i})

    # JO{fmt_i} = T.dense_fixed(fb{fmt_i})
    # JI{fmt_i} = T.dense_fixed(funit{fmt_i})

    BATCH{fmt_i} = T.dense_fixed({batch_num})
    '''

    buffer_definitions_tc = f'''
    A{fmt_i} = T.match_sparse_buffer(a{fmt_i}, [BATCH{fmt_i}, N{fmt_i}, II{fmt_i}, KI_dense{fmt_i}], {dtype})
    KO_indptr{fmt_i} = T.match_sparse_buffer(indptr{fmt_i}, [IOaddOne{fmt_i}], dtype="int32")
    KI_indices{fmt_i} = T.match_sparse_buffer(indices{fmt_i}, [N{fmt_i}], dtype="int32")
    # I_indices{fmt_i} = T.match_sparse_buffer(iblk_indices{fmt_i}, [N{fmt_i}], dtype="int32")
    I_indices{fmt_i} = T.match_sparse_buffer(iblk_indices{fmt_i}, [IO{fmt_i}{f', II{fmt_i}' if use_C_shared else ''}], dtype="int32")

    # T.assume_buffer_domain(KO_indptr{fmt_i}, [0, nnzb{fmt_i}])
    '''


    # TODO: 不确定这个地方的C的layout改写成 两个维度行不行？试试
    comp_statements_tc0 = f'''
    for batch, io, jo in T.grid({batch_num}, mb{fmt_i}, fb{fmt_i}):
        with T.block("tcspmm{fmt_i}0"):
            vio, vjo = T.axis.remap("SS", [io, jo])
            T.block_attr({{"sparse":True}})
            for ko, ii, ki, ji in T.grid(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], tile_size{fmt_i}, group_size{fmt_i}, funit{fmt_i}):
                with T.block("tcspmm{fmt_i}1"):
                    vko = T.axis.reduce(nnzb{fmt_i}, ko)
                    vk = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + vko)
                    vii = T.axis.spatial(tile_size{fmt_i}, ii)
                    vki = T.axis.reduce(group_size{fmt_i}, ki)
                    vji = T.axis.spatial(funit{fmt_i}, ji)
                    # vko, vii, vki, vji = T.axis.remap("RSRS", [ko, ii, ki, ji])
                    # vii, vki, vji = T.axis.remap("SRS", [ii, ki, ji])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = {zerotype}
                    # C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]
                    C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, 1]*tile_size{fmt_i}+vii, vjo*funit{fmt_i}+vji] + A{fmt_i}[vk, vii, vki] * B[KI_indices{fmt_i}[vk, vki], vjo*funit{fmt_i}+vji]

    '''
    # C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] = {zerotype}
    #                 C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] = C[I_indices{fmt_i}[vio, vii], vjo*funit{fmt_i}+vji] + A{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B[KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + vko, vki], vjo*funit{fmt_i}+vji]


    comp_statements_tc = f'''
    for batch, io, jo in T.grid({batch_num}, mb{fmt_i}, fb{fmt_i}):
        with T.block("tcspmm{fmt_i}0"):
            vbatch, vio, vjo = T.axis.remap("SSS", [batch, io, jo])
            T.block_attr({{"sparse":True}})
            {f'C_shared{fmt_i} = T.alloc_buffer([tile_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="shared")' if use_C_shared else ''}
            C_shared_wmma_accumulator{fmt_i} = T.alloc_buffer([tile_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="wmma.accumulator")
            A{fmt_i}_wmma_matrix_a = T.alloc_buffer([{batch_num}, nnzb{fmt_i}, tile_size{fmt_i}, group_size{fmt_i}], dtype={dtype}, scope="wmma.matrix_a")
            # B_shared{fmt_i} = T.alloc_buffer([group_size{fmt_i}, funit{fmt_i}], dtype={dtype}, scope="shared")
            B{fmt_i}_wmma_matrix_b{fmt_i} = T.alloc_buffer([{batch_num}, {mma_k}, {mma_n}], dtype={dtype}, scope="wmma.matrix_b")
            for ii, ji in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("tcspmm{fmt_i}1_init"):
                    vii, vji = T.axis.remap("SS", [ii, ji])
                    T.block_attr({{"sparse":True}})
                    C_shared_wmma_accumulator{fmt_i}[vii, vji] = {zerotype}
            for ko in T.serial(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio]):
                for ii_0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
                    # for ax_00 in T.unroll(1):
                    #     with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                    #         v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                    #         for ax0, ax1 in T.grid({mma_m}, group_size{fmt_i}):
                    #             with T.block("A{fmt_i}_wmma.matrix_a"):
                    #                 v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ax0)
                    #                 v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                    #                 T.block_attr({{"sparse":True}})
                    #                 A{fmt_i}_wmma_matrix_a[v0, v1, v2] = A{fmt_i}[v0, v1, v2]
                    

                    # for ax1_0 in T.unroll(group_size{fmt_i} // {mma_k}):
                    #     with T.block("A{fmt_i}_wmma.matrix_a_ooo"):
                    #         v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                    #         v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                    #         v2 = T.axis.spatial(group_size{fmt_i}, ax1_0 * {mma_k})
                    #         for ax1_00 in T.unroll(1):
                    #             with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                    #                 for ax0, ax1_1 in T.grid({mma_m}, {mma_k}):
                    #                     with T.block("A{fmt_i}_wmma.matrix_a"):
                    #                         v3, v4 = T.axis.remap("SS", [ax0, ax1_1])
                    #                         # v1 = T.axis.spatial({mma_m}, ii_0 * {mma_m} + ax0)
                    #                         # v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                    #                         T.block_attr({{"sparse":True}})
                    #                         A{fmt_i}_wmma_matrix_a[vbatch, v0, v1+v3, v2+v4] = A{fmt_i}[vbatch, v0, v1+v3, v2+v4]
                    for ki_0 in T.serial(group_size{fmt_i} // {mma_k}):

                        # for ax0, ax1 in T.grid(group_size{fmt_i}, {mma_n}):
                        #     with T.block("B_shared{fmt_i}"):
                        #         v0, v1 = T.axis.remap("SS", [ko, ax0])
                        #         v2 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ax1)
                        #         T.block_attr({{"sparse":True}})
                        #         B_shared{fmt_i}[v1, v2] = B[vbatch, KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + v0]*group_size{fmt_i} + v1, vjo * funit{fmt_i} + v2]
                        for ji_0 in T.unroll(funit{fmt_i} // {mma_n}):

                            for ax1_0000 in T.unroll(1):
                                with T.block("A{fmt_i}_wmma.matrix_a_ooo"):
                                    v0 = T.axis.spatial(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                                    v1 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                                    v2 = T.axis.spatial(group_size{fmt_i}, ki_0 * {mma_k})
                                    for ax1_00 in T.unroll(1):
                                        with T.block("A{fmt_i}_wmma.matrix_a_oo"):
                                            for ax0, ax1_1 in T.grid({mma_m}, {mma_k}):
                                                with T.block("A{fmt_i}_wmma.matrix_a"):
                                                    v3, v4 = T.axis.remap("SS", [ax0, ax1_1])
                                                    # v1 = T.axis.spatial({mma_m}, ii_0 * {mma_m} + ax0)
                                                    # v2 = T.axis.spatial(group_size{fmt_i}, ax1)
                                                    T.block_attr({{"sparse":True}})
                                                    A{fmt_i}_wmma_matrix_a[vbatch, v0, v1+v3, v2+v4] = A{fmt_i}[vbatch, v0, v1+v3, v2+v4]


                            for ax1_000 in T.unroll(1):
                                v00 = T.axis.spatial({mma_k}, KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + ko])
                                with T.block("B_shared_wmma{fmt_i}.matrix_b_ooo"):
                                    v0 = T.axis.spatial({mma_k}, v00*group_size{fmt_i} + ki_0*{mma_k})
                                    v2 = T.axis.spatial({mma_n}, vjo * funit{fmt_i} + ji_0 * {mma_n})
                                    for ax1_00 in T.unroll(1):
                                        with T.block("B_shared_wmma{fmt_i}.matrix_b_oo"):
                                            for ax0, ax1 in T.grid({mma_k}, {mma_n}):
                                                with T.block("B_shared_wmma{fmt_i}.matrix_b"):
                                                    v1, v3 = T.axis.remap("SS", [ax0, ax1])
                                                    T.block_attr({{"sparse":True}})
                                                    B{fmt_i}_wmma_matrix_b{fmt_i}[vbatch,   v1, v3] = B[vbatch, v0 + v1, v2 + v3]
                            for ax_00 in T.unroll(1):
                                with T.block("tcspmm{fmt_i}1_update_oo"):
                                    v1 = T.axis.reduce(nnzb{fmt_i}, KO_indptr{fmt_i}[vio] + ko)
                                    v2 = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m})
                                    v3 = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k})
                                    v4 = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n})

                                    v00 = T.axis.reduce(k_tot, KI_indices{fmt_i}[KO_indptr{fmt_i}[vio] + ko]*group_size{fmt_i})
                                    v01 = T.axis.spatial(n_tot, vjo * funit{fmt_i})
                                    
                                    for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
                                        with T.block("tcspmm{fmt_i}1_update"):
                                            # vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                                            # vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
                                            # vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
                                            # vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
                                            vii = T.axis.spatial({mma_m}, ii_1)
                                            vki = T.axis.reduce({mma_k}, ki_1)
                                            vji = T.axis.spatial({mma_n}, ji_1)
                                            T.block_attr({{"sparse":True}})
                                            C_shared_wmma_accumulator{fmt_i}[v2+vii, v4+vji] = C_shared_wmma_accumulator{fmt_i}[v2+vii, v4+vji] + A{fmt_i}_wmma_matrix_a[vbatch, v1, v2+vii, v3+vki] * B{fmt_i}_wmma_matrix_b{fmt_i}[vbatch, vki, vji]
                            # 
                            # for ii_1, ki_1, ji_1 in T.grid({mma_m}, {mma_k}, {mma_n}):
                            #     with T.block("tcspmm{fmt_i}1_update"):
                            #         vko = T.axis.reduce(KO_indptr{fmt_i}[vio + 1] - KO_indptr{fmt_i}[vio], ko)
                            #         vii = T.axis.spatial(tile_size{fmt_i}, ii_0 * {mma_m} + ii_1)
                            #         vki = T.axis.reduce(group_size{fmt_i}, ki_0 * {mma_k} + ki_1)
                            #         vji = T.axis.spatial(funit{fmt_i}, ji_0 * {mma_n} + ji_1)
                            #         T.block_attr({{"sparse":True}})
                            #         C_shared_wmma_accumulator{fmt_i}[vii, vji] = C_shared_wmma_accumulator{fmt_i}[vii, vji] + A{fmt_i}_wmma_matrix_a[KO_indptr{fmt_i}[vio] + vko, vii, vki] * B_shared_wmma_matrix_b{fmt_i}[vki, vji]
            # for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
            #     with T.block("C_shared_wmma{fmt_i}.accumulator"):
            #         v0, v1 = T.axis.remap("SS", [ax0, ax1])
            #         T.block_attr({{"sparse":True}})
            #         C_shared{fmt_i}[v0, v1] = C_shared_wmma_accumulator{fmt_i}[v0, v1]
            # for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
            #     with T.block("C_shared{fmt_i}"):
            #         v0, v1 = T.axis.remap("SS", [ax0, ax1])
            #         T.block_attr({{"sparse":True}})
            #         C[I_indices{fmt_i}[vio, v0], vjo * funit{fmt_i} + v1] = C_shared{fmt_i}[v0, v1]
    '''
    
    if use_C_shared:
        comp_statements_tc = comp_statements_tc + f'''
            for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("C_shared_wmma{fmt_i}.accumulator"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.block_attr({{"sparse":True}})
                    C_shared{fmt_i}[v0, v1] = C_shared_wmma_accumulator{fmt_i}[v0, v1]
            for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                with T.block("C_shared{fmt_i}"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.block_attr({{"sparse":True}})
                    C[vbatch, I_indices{fmt_i}[vio, v0], vjo * funit{fmt_i} + v1] = C_shared{fmt_i}[v0, v1]
    '''
    else:
        comp_statements_tc = comp_statements_tc + f'''
            # for ax0 in T.thread_binding(tile_size{fmt_i}//{mma_m}, thread="threadIdx.y"):
            #     for ax1 in T.serial(funit{fmt_i} // {mma_n}):
            #         with T.block("C_shared_wmma{fmt_i}.accumulator_ooo"):
            #             v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i} + ax0 * {mma_m})
            #             v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i} + ax1 * {mma_n})
            #             for ax000 in T.unroll(1):
            #                 with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
            #                     for ax2, ax3 in T.grid({mma_m}, {mma_n}):
            #                         with T.block("C_shared_wmma{fmt_i}.accumulator"):
            #                             vi, vj = T.axis.remap("SS", [ax2, ax3])
            #                             T.block_attr({{"sparse":True}})
            #                             C[v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
            for ax000 in T.unroll(1):
                with T.block("C_shared_wmma{fmt_i}.accumulator_oo"):
                    v1 = T.axis.spatial(m_tot, I_indices{fmt_i}[vio] * tile_size{fmt_i})
                    v2 = T.axis.spatial(n_tot, vjo * funit{fmt_i})                  
                    for ax0, ax1 in T.grid(tile_size{fmt_i}, funit{fmt_i}):
                        with T.block("C_shared_wmma{fmt_i}.accumulator"):
                            vi, vj = T.axis.remap("SS", [ax0, ax1])
                            T.block_attr({{"sparse":True}})
                            C[vbatch, v1 + vi, v2 + vj] = C_shared_wmma_accumulator{fmt_i}[vi, vj]
    '''

    return parameters_tc, idx_definitions_tc, buffer_definitions_tc, comp_statements_tc








# SDDMM 的 1D tile
def gen_definitions_1D_SDDMM(fmt_i, dtype, zerotype):

    parameters_tc = f'''
    nnz{fmt_i}: T.int32,
    mid_idx{fmt_i}: T.handle,
    j_idx{fmt_i}: T.handle,
    remap{fmt_i}: T.handle,
    '''

    idx_definitions_tc = f'''
    J{fmt_i} = T.dense_fixed(nnz{fmt_i})
    '''

    buffer_definitions_tc = f'''
    mid{fmt_i} = T.match_sparse_buffer(mid_idx{fmt_i}, [J{fmt_i}], dtype="int32")
    J_indices{fmt_i} = T.match_sparse_buffer(j_idx{fmt_i}, [J{fmt_i}], dtype="int32")
    REMAP{fmt_i} = T.match_sparse_buffer(remap{fmt_i}, [J{fmt_i}], dtype="int32")

    '''

    # 默认会对最终C的结果重映射，因为默认最后找到的是hybrid模式，一部分nnz会被TC tile 覆盖。
    comp_statements_tc = f'''
    for j, k in T.grid(nnz{fmt_i}, k_tot):
        with T.block("sddmm{fmt_i}0"):
            vj, vk = T.axis.remap("SR", [j, k])
            T.block_attr({{"sparse":True}})
            # with T.init():
            #     C[ vj ] = {zerotype}
            # C[ vj ] = C[ vj ] + A_1D[ mid{fmt_i}[vj], vk ] * B_1D[ J_indices{fmt_i}[vj], vk ]
            with T.init():
                C[ REMAP{fmt_i}[vj] ] = {zerotype}
            C[ REMAP{fmt_i}[vj] ] = C[ REMAP{fmt_i}[vj] ] + A_1D[ mid{fmt_i}[vj], vk ] * B_1D[ J_indices{fmt_i}[vj], vk ]
    '''

    if os.environ['REMAP'] == 'False':
        parameters_tc = parameters_tc + f'''
    c{fmt_i}: T.handle,
    '''
        buffer_definitions_tc = buffer_definitions_tc + f'''
    C{fmt_i} = T.match_sparse_buffer(c{fmt_i}, [J{fmt_i}], dtype={dtype})
    '''
        comp_statements_tc = f'''
    for j, k in T.grid(nnz{fmt_i}, k_tot):
        with T.block("sddmm{fmt_i}0"):
            vj, vk = T.axis.remap("SR", [j, k])
            T.block_attr({{"sparse":True}})
            # with T.init():
            #     C[ vj ] = {zerotype}
            # C[ vj ] = C[ vj ] + A_1D[ mid{fmt_i}[vj], vk ] * B_1D[ J_indices{fmt_i}[vj], vk ]
            with T.init():
                C{fmt_i}[ vj ] = {zerotype}
            C{fmt_i}[ vj ] = C{fmt_i}[ vj ] + A_1D[ mid{fmt_i}[vj], vk ] * B_1D[ J_indices{fmt_i}[vj], vk ]
    '''        

    return parameters_tc, idx_definitions_tc, buffer_definitions_tc, comp_statements_tc






def gen_definitions_tc_SDDMM(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params):

    parameters_tc = f'''
    nnzb{fmt_i}: T.int32,
    mb{fmt_i}: T.int32,
    nb{fmt_i}: T.int32,
    nnz{fmt_i}: T.int32,
    i_indices{fmt_i}: T.handle,
    j_indices{fmt_i}: T.handle,
    indptr{fmt_i}: T.handle,
    indicesDense{fmt_i}: T.handle,
    indices{fmt_i}: T.handle,
    '''

    idx_definitions_tc = f'''
    BNUM{fmt_i} = T.dense_fixed(nnzb{fmt_i})
    BNUMaddone{fmt_i} = T.dense_fixed(nnzb{fmt_i}+1)
    MB{fmt_i} = T.dense_fixed(mb{fmt_i})
    NB{fmt_i} = T.dense_fixed(nb{fmt_i})
    NNZ{fmt_i} = T.dense_fixed(nnz{fmt_i})
    '''

    buffer_definitions_tc = f'''
    I_indices{fmt_i} = T.match_sparse_buffer(i_indices{fmt_i}, [BNUM{fmt_i}, MB{fmt_i}], dtype="int32")
    J_indices{fmt_i} = T.match_sparse_buffer(j_indices{fmt_i}, [BNUM{fmt_i}, NB{fmt_i}], dtype="int32")
    Indptr{fmt_i} = T.match_sparse_buffer(indptr{fmt_i}, [BNUMaddone{fmt_i}], dtype="int32")
    IndicesDense{fmt_i} = T.match_sparse_buffer(indicesDense{fmt_i}, [NNZ{fmt_i}], dtype="int32")
    Indices{fmt_i} = T.match_sparse_buffer(indices{fmt_i}, [NNZ{fmt_i}], dtype="int32")
    '''


    # 涉及到的buffer
    # I_indices{fmt_i}  J_indices{fmt_i}  Indptr{fmt_i} IndicesDense{fmt_i}  Indices{fmt_i}
    # A, B, C, C_shared


    # 每个TC block对应一个thread block，或者我们可以把若干个TC block合并成一个thread block，具体得看之后的benchmark的结果
    comp_statements_tc = f'''
    for bnum in T.serial(nnzb{fmt_i}):
        with T.block("tcsddmm_out{fmt_i}0"):
            vbnum = T.axis.remap("S", [bnum])
            C_shared{fmt_i} = T.alloc_buffer([mb{fmt_i}, nb{fmt_i}], dtype={dtype}, scope="shared")
            for m, n, k in T.grid(mb{fmt_i}, nb{fmt_i}, k_tot):
                with T.block("tcsddmm{fmt_i}0"):
                    vm, vn, vk = T.axis.remap("SSR", [m, n, k])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        C_shared{fmt_i}[vm, vn] = {zerotype}
                    C_shared{fmt_i}[vm, vn] = C_shared{fmt_i}[vm, vn] + A[ I_indices{fmt_i}[vbnum, vm], vk ] * B[ J_indices{fmt_i}[vbnum, vn], vk ]
        # 
    # for bnum in T.serial(nnzb{fmt_i}):
            for i in T.serial( Indptr{fmt_i}[vbnum+1] - Indptr{fmt_i}[vbnum] ):
                with T.block("tcsddmm_store{fmt_i}0"):
                    vi = T.axis.remap("S", [i])
                    vm = T.axis.spatial(mb{fmt_i}, IndicesDense{fmt_i}[ Indptr{fmt_i}[vbnum]+i ] // nb{fmt_i})
                    vn = T.axis.spatial(nb{fmt_i}, IndicesDense{fmt_i}[ Indptr{fmt_i}[vbnum]+i ] % nb{fmt_i})
                    T.block_attr({{"sparse":True}})
                    C[ Indices{fmt_i}[ Indptr{fmt_i}[vbnum]+vi ] ] = C_shared{fmt_i}[vm, vn ]

    # for bnum in T.serial(nnzb{fmt_i}):
    #     for i in T.serial( Indptr{fmt_i}[bnum+1] - Indptr{fmt_i}[bnum] ):
    #         with T.block("tcsddmm_store{fmt_i}0"):
    #             vbnum, vi = T.axis.remap("SS", [bnum, i])
    #             vm = T.axis.spatial(mb{fmt_i}, IndicesDense{fmt_i}[ Indptr{fmt_i}[vbnum]+i ] // nb{fmt_i})
    #             vn = T.axis.spatial(nb{fmt_i}, IndicesDense{fmt_i}[ Indptr{fmt_i}[vbnum]+i ] % nb{fmt_i})
    #             T.block_attr({{"sparse":True}})
    #             C[ Indices{fmt_i}[ Indptr{fmt_i}[vbnum]+vi ] ] = C_shared{fmt_i}[vm, vn ]
    '''


    if os.environ['REMAP'] == 'False':
        parameters_tc = parameters_tc + f'''
    c{fmt_i}: T.handle,
    '''
        buffer_definitions_tc = buffer_definitions_tc + f'''
    C{fmt_i} = T.match_sparse_buffer(c{fmt_i}, [BNUM{fmt_i}, MB{fmt_i}, NB{fmt_i}], dtype={dtype})
    '''
        comp_statements_tc = f'''
    for bnum in T.serial(nnzb{fmt_i}):
        with T.block("tcsddmm_out{fmt_i}0"):
            vbnum = T.axis.remap("S", [bnum])
            # C_shared{fmt_i} = T.alloc_buffer([mb{fmt_i}, nb{fmt_i}], dtype={dtype}, scope="shared")
            for m, n, k in T.grid(mb{fmt_i}, nb{fmt_i}, k_tot):
                with T.block("tcsddmm{fmt_i}0"):
                    vm, vn, vk = T.axis.remap("SSR", [m, n, k])
                    T.block_attr({{"sparse":True}})
                    with T.init():
                        C{fmt_i}[vbnum, vm, vn] = {zerotype}
                    C{fmt_i}[vbnum, vm, vn] = C{fmt_i}[vbnum, vm, vn] + A[ I_indices{fmt_i}[vbnum, vm], vk ] * B[ J_indices{fmt_i}[vbnum, vn], vk ]
    '''

    return parameters_tc, idx_definitions_tc, buffer_definitions_tc, comp_statements_tc









# 加了两个input： a_ell, ell_nnz_tot
def gen_fused_definitions(parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads, dtype, cache_set):
    template = f'''@T.prim_func
def my_fusedFormats(
    b: T.handle,
    c: T.handle,
    m_tot: T.int32,
    n_tot: T.int32,
    k_tot: T.int32,
    max_shared: T.int32,
    {parameters}
) -> None:
    T.func_attr({{"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2, "horizontal_fuse": True}})

    # IO = T.dense_fixed(mb)
    # II = T.dense_fixed(tile_size)
    # K = T.dense_fixed(nb * group_size)
    # J = T.dense_fixed(feat_size)
    I_tot = T.dense_fixed(m_tot)
    J_tot = T.dense_fixed(n_tot)
    K_tot = T.dense_fixed(k_tot)

    {idx_definitions}


    # A_ell = T.match_buffer(a_ell, [ell_nnz_tot], {dtype})

    B = T.match_sparse_buffer(b, [K_tot, J_tot], {dtype})
    C = T.match_sparse_buffer(c, [I_tot, J_tot], {dtype})

    {f'A_shared = T.alloc_buffer([max_shared], dtype={dtype}, scope="shared")' if 'A' in cache_set else ''}


    {buffer_definitions}

    {cache_reads}

    {comp_statements}

    '''
    return template




def gen_fused_definitions_batchspmm(parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads, dtype, cache_set, batch_num):
    template = f'''@T.prim_func
def my_fusedFormats(
    b: T.handle,
    c: T.handle,
    m_tot: T.int32,
    n_tot: T.int32,
    k_tot: T.int32,
    max_shared: T.int32,
    {parameters}
) -> None:
    T.func_attr({{"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2, "horizontal_fuse": True}})

    # IO = T.dense_fixed(mb)
    # II = T.dense_fixed(tile_size)
    # K = T.dense_fixed(nb * group_size)
    # J = T.dense_fixed(feat_size)
    I_tot = T.dense_fixed(m_tot)
    J_tot = T.dense_fixed(n_tot)
    K_tot = T.dense_fixed(k_tot)

    BATCH_tot = T.dense_fixed({batch_num})

    {idx_definitions}


    # A_ell = T.match_buffer(a_ell, [ell_nnz_tot], {dtype})

    B = T.match_sparse_buffer(b, [BATCH_tot, K_tot, J_tot], {dtype})
    C = T.match_sparse_buffer(c, [BATCH_tot, I_tot, J_tot], {dtype})

    {f'A_shared = T.alloc_buffer([max_shared], dtype={dtype}, scope="shared")' if 'A' in cache_set else ''}


    {buffer_definitions}

    {cache_reads}

    {comp_statements}

    '''
    return template






# 加了两个input： a_ell, ell_nnz_tot
def gen_fused_definitions_sddmm(parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads, dtype, cache_set):
    template = f'''@T.prim_func
def my_fusedFormats(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    a_1D: T.handle,
    b_1D: T.handle,
    nnz_tot: T.int32,
    m_tot: T.int32,
    n_tot: T.int32,
    k_tot: T.int32,
    {parameters}
) -> None:
    T.func_attr({{"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2, "horizontal_fuse": True}})

    I_tot = T.dense_fixed(m_tot)
    J_tot = T.dense_fixed(n_tot)
    K_tot = T.dense_fixed(k_tot)
    NNZ_tot = T.dense_fixed(nnz_tot)

    {idx_definitions}


    A = T.match_sparse_buffer(a, [I_tot, K_tot], {dtype})
    B = T.match_sparse_buffer(b, [J_tot, K_tot], {dtype})
    C = T.match_sparse_buffer(c, [NNZ_tot], {dtype})

    A_1D = T.match_sparse_buffer(a_1D, [I_tot, K_tot], {dtype})
    B_1D = T.match_sparse_buffer(b_1D, [J_tot, K_tot], {dtype})

    # {f'A_shared = T.alloc_buffer([max_shared], dtype={dtype}, scope="shared")' if 'A' in cache_set else ''}


    {buffer_definitions}

    {cache_reads}

    {comp_statements}

    '''
    return template






def gen_op_definition_BSPMM(formats, dtype, zerotype, cache_set, dsize, op_type, batch_num=1):
    '''
        formats: all the selected formats. Each format is a tuple of (template_str, tile.best_tile_sizes, tile.best_params)
        OUTPUT:
            the string of the fused op definition.
    '''
    parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads = '', '', '', '', ''
    for i, fmt in enumerate(formats):
        param_def, idx_def, buffer_def, comp_def, cache_r = None, None, None, None, ''
        if fmt[0] == "sparse_template":
            fmt_i = f'{i}'
            param_def, idx_def, buffer_def, comp_def = gen_definitions_csr(fmt_i, dtype, zerotype)
        elif fmt[0] == "sparse_template_ell":
            fmt_i = f'{i}'
            params = json.loads(fmt[-1])
            tile_sizes = fmt[1]
            if op_type == 'spmm':
                param_def, idx_def, buffer_def, comp_def, cache_r = gen_definitions_ell(fmt_i, dtype, zerotype, params, tile_sizes, dsize)
            elif op_type == 'batched_spmm':
                param_def, idx_def, buffer_def, comp_def, cache_r = gen_definitions_ell_batchspmm(fmt_i, dtype, zerotype, params, tile_sizes, dsize, batch_num)
        elif fmt[0] == "TensorCore_template":
            fmt_i = f'{i}'
            mma_m, mma_n, mma_k = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])
            params = json.loads(fmt[-1])
            if op_type == 'spmm':
                param_def, idx_def, buffer_def, comp_def = gen_definitions_tc(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params)
            elif op_type == 'batched_spmm':
                if formats[fmt][0].op.TC_k_notsorted:
                    param_def, idx_def, buffer_def, comp_def = gen_definitions_tc_batchspmm_knotsorted(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params, batch_num)
                else:
                    param_def, idx_def, buffer_def, comp_def = gen_definitions_tc_batchspmm(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params, batch_num)

        parameters = parameters + param_def
        idx_definitions = idx_definitions + idx_def
        buffer_definitions = buffer_definitions + buffer_def
        comp_statements = comp_statements + comp_def
        cache_reads = cache_reads + cache_r

    if op_type == 'spmm':
        return gen_fused_definitions(parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads, dtype, cache_set)
    elif op_type == 'batched_spmm':
        return gen_fused_definitions_batchspmm(parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads, dtype, cache_set, batch_num)





def gen_op_definition_BSDDMM(formats, dtype, zerotype, cache_set, dsize, op_type, batch_num=1, fmt_i_start=0):
    parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads = '', '', '', '', ''
    for i, fmt in enumerate(formats):
        param_def, idx_def, buffer_def, comp_def, cache_r = None, None, None, None, ''
        if fmt[0] == "1D_sddmm":
            fmt_i = f'{i+fmt_i_start}'
            params = json.loads(fmt[-1])
            if op_type == 'sddmm':
                param_def, idx_def, buffer_def, comp_def = gen_definitions_1D_SDDMM(fmt_i, dtype, zerotype)
        elif fmt[0] == "TC_sddmm":
            fmt_i = f'{i+fmt_i_start}'
            mma_m, mma_n, mma_k = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])
            params = json.loads(fmt[-1])
            if op_type == 'sddmm':
                param_def, idx_def, buffer_def, comp_def = gen_definitions_tc_SDDMM(fmt_i, dtype, zerotype, mma_m, mma_n, mma_k, params)

        parameters = parameters + param_def
        idx_definitions = idx_definitions + idx_def
        buffer_definitions = buffer_definitions + buffer_def
        comp_statements = comp_statements + comp_def
        cache_reads = cache_reads + cache_r

    if op_type == 'sddmm':
        return gen_fused_definitions_sddmm(parameters, idx_definitions, buffer_definitions, comp_statements, cache_reads, dtype, cache_set)




def gen_op_definition(formats, dtype, zerotype, cache_set, dsize, op_type, batch_num=1, fmt_i_start=0):
    '''
        formats: all the selected formats. Each format is a tuple of (template_str, tile.best_tile_sizes, tile.best_params)
        OUTPUT:
            the string of the fused op definition.
    '''
    if op_type in ['spmm', 'batched_spmm']:
        return gen_op_definition_BSPMM(formats, dtype, zerotype, cache_set, dsize, op_type, batch_num=batch_num)
    elif op_type in ['sddmm']:
        return gen_op_definition_BSDDMM(formats, dtype, zerotype, cache_set, dsize, op_type, batch_num=batch_num, fmt_i_start=fmt_i_start)





def get_padded_j_TC(ori_j_len, mma_n):
    if (ori_j_len < mma_n):
        ori_j_len = mma_n
    elif (ori_j_len > mma_n):
        ori_j_len = math.ceil(ori_j_len / (2*mma_n)) * 2*mma_n    
    return ori_j_len







def set_params_BSPMM(func, formats, dsize):
    '''
        Prepare the parameters for the fused kernel.
        INPUT:
            func: the fused kernel definition.
            formats: all the selected formats. Each format key is a tuple of (template_str, tile.best_tile_sizes, tile.best_params)
        OUTPUT:
            sch: the initial sch.
            (max_i, max_j): the final output shape of the original op after optimization.
    '''
    params_start = 6 #5
    params_dict = dict()

    max_i, max_j = -1, -1

    a_ell_offset = 1
    TC_out_SMEM = None

    for i, fmt in enumerate(formats):
        tiles = formats[fmt]
        if fmt[0] == "sparse_template":
            n = tiles[0].op.idx_lens[1]
            m = math.prod(fmt[1][0][1:]) * len(tiles)
            # nnz = sum([tile.nnz_when_selected // len(tile.position_space_when_selected['j']) for tile in tiles])
            nnz = sum([tile.nnz_when_selected // tile.j_num for tile in tiles])

            _, M, N, NNZ = func.params[params_start:params_start+4]
            params_dict[M] = int(m)
            params_dict[N] = int(n)
            params_dict[NNZ] = int(nnz)

            params_start = params_start + 7

            # print(type(m), type(n), type(nnz))
            
        
        elif fmt[0] == "sparse_template_ell":
            m = math.prod(fmt[1][0][1:]) * len(tiles)
            n = tiles[0].op.idx_lens[1]
            nnz_cols = math.prod(fmt[1][2][1:])
            m_blk = math.prod(fmt[1][0][1:])
            n_blk = math.prod(fmt[1][1][1:])
            m_num = len(tiles)
            n_num = math.ceil(n/n_blk)

            _, M_NUM, N_NUM, NNZ_COLS, M_BLK, N_BLK = func.params[params_start:params_start+6]
            params_dict[M_NUM] = int(m_num)
            params_dict[N_NUM] = int(n_num)
            params_dict[NNZ_COLS] = int(nnz_cols)
            params_dict[M_BLK] = int(m_blk)
            params_dict[N_BLK] = int(n_blk)

            SMEM_pad_pattern = comp_SMEM_padding_pattern('spmm', fmt[1], dsize)
            # SMEM_pad_num = (m_blk-1)//SMEM_pad_pattern[0]*SMEM_pad_pattern[1]
            SMEM_pad_num = m_blk*SMEM_pad_pattern[1]

            a_ell_offset = max(a_ell_offset, m_blk * nnz_cols + SMEM_pad_num)

            params_start = params_start + 8

        elif fmt[0] == "TensorCore_template":
            # tile_size, group_size = fmt[1][0][1], fmt[1][2][1]
            # mb = math.ceil(tiles[0].op.idx_lens[0] / tile_size)
            # nb = math.ceil(tiles[0].op.idx_lens[2] / group_size)
            # nnzb = len(tiles)
            # feat_size = tiles[0].op.idx_lens[1]
            # mma_n = fmt[1][1][1]
            # feat_size = get_padded_j_TC(feat_size, mma_n)
            # # if (feat_size < mma_n):
            # #     feat_size = mma_n
            # # elif (feat_size > mma_n):
            # #     feat_size = math.ceil(feat_size / (2*mma_n)) * 2*mma_n


            # _, MB, NB, NNZB, F, T, G = func.params[params_start:params_start+7]
            # params_dict[MB] = mb
            # params_dict[NB] = nb
            # params_dict[NNZB] = nnzb
            # params_dict[F] = feat_size
            # params_dict[T] = tile_size
            # params_dict[G] = group_size

            # params_start = params_start + 9

            # max_i = max( max_i, mb * tile_size )
            # max_j = max( max_j, feat_size )


            # =========================================
            tile_size, group_size = fmt[1][0][1], fmt[1][2][1]

            # mb = math.ceil(tiles[0].op.idx_lens[0] / tile_size)
            # 此处考虑类似dbsrmm里的mb，即把空行压缩掉了
            mb = len(set([t.tile_pos[0] for t in tiles]))

            nb = math.ceil(tiles[0].op.idx_lens[2] / group_size)
            nnzb = len(tiles)
            feat_size = tiles[0].op.idx_lens[1]
            # mma_n = fmt[1][1][1]
            mma_n = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])[1]
            feat_size = get_padded_j_TC(feat_size, mma_n)
            
            # funit是被tile_sizes决定的，此处的写法有问题
            # funit = min(2, feat_size // mma_n) * mma_n
            funit = fmt[1][1][1]
            fb = feat_size//funit

            # FOR DEBUG----------
            # funit = mma_n
            # fb = feat_size//funit
            # -------------------


            _, MB, NB, NNZB, FB, FUNIT, T, G = func.params[params_start:params_start+8]
            print(mb, nb, nnzb, fb, funit, tile_size, group_size)
            params_dict[MB] = int(mb)
            params_dict[NB] = int(nb)
            params_dict[NNZB] = int(nnzb)
            params_dict[FB] = int(fb)
            params_dict[FUNIT] = int(funit)
            params_dict[T] = int(tile_size)
            params_dict[G] = int(group_size)

            params_start = params_start + 11 # 10 如果我们使用了cachewrite+ shared memory

            # 这个地方写错了，应该考虑的是是否可能有额外的padding
            # max_i = max( max_i, mb * tile_size )
            max_i = max( max_i, math.ceil(tiles[0].op.idx_lens[0]/tile_size)*tile_size )

            max_j = max( max_j, feat_size )

            a_ell_offset = max(a_ell_offset, group_size * mma_n)

            params = json.loads(fmt[-1])
            if params['idx_reordered'][0] or params['real_atomic']:
                # 默认当i没有被reorder的时候并且不atomicAdd，不使用C_shared
                if TC_out_SMEM == None:
                    TC_out_SMEM = tile_size * funit
                else:
                    TC_out_SMEM = max(TC_out_SMEM, tile_size * funit)

            # TC_out_SMEM = tile_size * funit


    # set m_tot, n_tot, k_tot
    max_i = max( max_i, tiles[0].op.idx_lens[0]+1 ) # 此处的这个调整是因为我们有时候会修改和whole_row TC tile交叉的ELL tile的output范围
    max_j = max( max_j, tiles[0].op.idx_lens[1] )

    M_TOT, N_TOT, K_TOT, MAX_SHARED = func.params[2:6] # [2:5]
    params_dict[M_TOT] = int(max_i)
    params_dict[N_TOT] = int(max_j)
    params_dict[K_TOT] = int(tiles[0].op.idx_lens[2])
    params_dict[MAX_SHARED] = int(a_ell_offset)

    print(max_i, max_j, tiles[0].op.idx_lens[2], a_ell_offset)


    # 需要把a_ell_offset也存到json文件里面使得我们在fix cuda bug的时候可以处理
    with open(f"A_ell_max_shared{os.environ['MyFileID']}.json", 'w') as f:
        json.dump((a_ell_offset, TC_out_SMEM), f)


    # print(type(max_i), type(max_j), type(tiles[0].op.idx_lens[2]))

    mod = tvm.IRModule.from_expr(
            func.specialize(params_dict)
        )
    
    # print(mod.script())


    # 因为对于ELL的function定义中我们已经写成了lower完sparse iter之后的形式，再运行这一函数会造成segment fault，所以此处需要修改
    # mod = lower_sparse_iter(mod)
    
    # print(mod.script())

    sch = tir.Schedule(mod)
    
    # print(sch.mod.script())
    return sch, (max_i, max_j)






def set_params_BSDDMM(func, formats, dsize):
    params_start = 7+2 #5
    params_dict = dict()

    A_SMEM = 0
    B_SMEM = 0
    C_SMEM = 0
    for i, fmt in enumerate(formats):
        tiles = formats[fmt]
        
        if fmt[0] == "1D_sddmm":
            nnz = sum([t.nnz_when_selected//t.j_num for t in tiles])
            NNZ = func.params[params_start]

            max_bucket_size = json.loads(fmt[2])['max_bucket_size']
            nnz = len(tiles) * max_bucket_size

            # 因为我们可能会允许在parameter tuning的时候改变max_bucket_size，所以之前计算nnz的时候有问题
            nnz = sum([t.nnz for t in tiles])
            real_max_bucket_size = json.loads(fmt[2])['ty']*json.loads(fmt[2])['group_size']
            nnz = math.ceil(nnz/real_max_bucket_size)*real_max_bucket_size

            # params_dict[NNZ] = math.ceil(int(nnz)/max_bucket_size) * max_bucket_size # 我们会默认把1D tile pad完整
            params_dict[NNZ] = nnz
            # params_dict[NNZ] = math.ceil(int(nnz)/fmt[1][0]) * fmt[1][0] # 我们会默认把1D tile pad完整

            if os.environ['REMAP'] == 'True':
                params_start = params_start + 4
            else:
                params_start = params_start + 5

        elif fmt[0] == "TC_sddmm":
            mb, nb = fmt[1][0][1], fmt[1][2][1] # nb = fmt[1][2][1]因为我们调整了sddmm op中的idx 顺序： i, k, j  为了利用spmm里面已有的代码
            nnzb = len(tiles)
            # nnz = sum([t.nnz_when_selected//t.j_num for t in tiles])
            # 与SPMM不同，此处我们直接计算原始nnz的结果，因为最后在计算的时候我们也不会重新调整tile之间重复覆盖的nnz
            nnz = sum([t.nnz for t in tiles])

            NNZB, MB, NB, NNZ = func.params[params_start:params_start+4]
            print(mb, nb, nnzb, nnz)
            params_dict[MB] = int(mb)
            params_dict[NB] = int(nb)
            params_dict[NNZB] = int(nnzb)
            params_dict[NNZ] = int(nnz)

            mma_m, mma_n, mma_k = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])
            warp_num = json.loads(fmt[2])["warp_num"]
            A_SMEM = max(A_SMEM, mb*fmt[1][1][1])
            B_SMEM = max(B_SMEM, mma_k*warp_num*mma_n)
            # C_SMEM = max(C_SMEM, mb*nb)

            if os.environ['REMAP'] == 'False':
                params_start = params_start + 10
                C_SMEM = max(C_SMEM, 0)
            else:
                params_start = params_start + 9 # 10 如果我们使用了cachewrite+ shared memory
                C_SMEM = max(C_SMEM, mb*nb)

            # params = json.loads(fmt[-1])
            # if params['idx_reordered'][0] or params['real_atomic']:
            #     # 默认当i没有被reorder的时候并且不atomicAdd，不使用C_shared
            #     if TC_out_SMEM == None:
            #         TC_out_SMEM = tile_size * funit
            #     else:
            #         TC_out_SMEM = max(TC_out_SMEM, tile_size * funit)



    # set m_tot, n_tot, k_tot
    NNZ_TOT, M_TOT, N_TOT, K_TOT = func.params[5:9] # [3:7] # [2:5]
    params_dict[M_TOT] = int(tiles[0].op.idx_lens[0])
    params_dict[N_TOT] = int(tiles[0].op.idx_lens[2])
    params_dict[K_TOT] = int(tiles[0].op.idx_lens[1])
    # params_dict[NNZ_TOT] = int(tiles[0].op.inps[0].nnz+1) # 我觉得这个地方可能不太合理，因为可能tiles对应的op的稀疏矩阵已经更新过了
    params_dict[NNZ_TOT] = int(sum([ t.nnz_when_selected//t.j_num for tiles in formats.values() for t in tiles ])) # + 1

    os.environ['A_SMEM'] = f"{A_SMEM}"
    os.environ['B_SMEM'] = f"{B_SMEM}"
    os.environ['C_SMEM'] = f"{C_SMEM}"

    # if A_SMEM > 0:
    #     os.environ['A_SMEM'] = f"{A_SMEM}"
    # if B_SMEM > 0:
    #     os.environ['B_SMEM'] = f"{B_SMEM}"
    # if C_SMEM > 0:
    #     os.environ['C_SMEM'] = f"{C_SMEM}"

    # print(max_i, max_j, tiles[0].op.idx_lens[2], a_ell_offset)


    # 需要把a_ell_offset也存到json文件里面使得我们在fix cuda bug的时候可以处理
    # with open(f"A_ell_max_shared{os.environ['MyFileID']}.json", 'w') as f:
    #     json.dump((a_ell_offset, TC_out_SMEM), f)


    # print(type(max_i), type(max_j), type(tiles[0].op.idx_lens[2]))

    mod = tvm.IRModule.from_expr(
            func.specialize(params_dict)
        )
    
    # print(mod.script())


    # 因为对于ELL的function定义中我们已经写成了lower完sparse iter之后的形式，再运行这一函数会造成segment fault，所以此处需要修改
    # mod = lower_sparse_iter(mod)
    
    # print(mod.script())

    sch = tir.Schedule(mod)
    
    # print(sch.mod.script())
    return sch, None









def set_params(func, formats, dsize, op_type):
    '''
        Prepare the parameters for the fused kernel.
        INPUT:
            func: the fused kernel definition.
            formats: all the selected formats. Each format key is a tuple of (template_str, tile.best_tile_sizes, tile.best_params)
        OUTPUT:
            sch: the initial sch.
            (max_i, max_j): the final output shape of the original op after optimization.
    '''
    if op_type in ['spmm', 'batched_spmm']:
        return set_params_BSPMM(func, formats, dsize)
    elif op_type in ['sddmm']:
        return set_params_BSDDMM(func, formats, dsize)








def set_params_measure_one_tile(func, formats, dsize, max_i, max_j, max_k):
    '''
        Prepare the parameters for the fused kernel.
        INPUT:
            func: the fused kernel definition.
            formats: all the selected formats. Each format key is a tuple of (template_str, tile.best_tile_sizes, tile.best_params)
        OUTPUT:
            sch: the initial sch.
            (max_i, max_j): the final output shape of the original op after optimization.
    '''
    params_start = 6 #5
    params_dict = dict()

    # max_i, max_j = -1, -1

    a_ell_offset = 1
    TC_out_SMEM = None

    for i, fmt in enumerate(formats):
        tiles = formats[fmt]
        repeat_row = 108*100 // (tiles[0].op.idx_lens[1] // math.prod(tiles[0].tile_sizes[1][1:])) # 108
        if fmt[0] == "sparse_template":
            n = tiles[0].op.idx_lens[1]
            m = math.prod(fmt[1][0][1:]) * len(tiles) * repeat_row
            # nnz = sum([tile.nnz_when_selected // len(tile.position_space_when_selected['j']) for tile in tiles])
            nnz = sum([tile.nnz_when_selected // tile.j_num for tile in tiles]) * repeat_row

            _, M, N, NNZ = func.params[params_start:params_start+4]
            params_dict[M] = int(m)
            params_dict[N] = int(n)
            params_dict[NNZ] = int(nnz)

            params_start = params_start + 7

            # print(type(m), type(n), type(nnz))
            
        
        elif fmt[0] == "sparse_template_ell":
            m = math.prod(fmt[1][0][1:]) * len(tiles) * repeat_row
            n = tiles[0].op.idx_lens[1]
            nnz_cols = math.prod(fmt[1][2][1:])
            m_blk = math.prod(fmt[1][0][1:])
            n_blk = math.prod(fmt[1][1][1:])
            m_num = len(tiles) * repeat_row
            n_num = math.ceil(n/n_blk)

            _, M_NUM, N_NUM, NNZ_COLS, M_BLK, N_BLK = func.params[params_start:params_start+6]
            params_dict[M_NUM] = int(m_num)
            params_dict[N_NUM] = int(n_num)
            params_dict[NNZ_COLS] = int(nnz_cols)
            params_dict[M_BLK] = int(m_blk)
            params_dict[N_BLK] = int(n_blk)

            SMEM_pad_pattern = comp_SMEM_padding_pattern('spmm', fmt[1], dsize)
            # SMEM_pad_num = (m_blk-1)//SMEM_pad_pattern[0]*SMEM_pad_pattern[1]
            SMEM_pad_num = m_blk*SMEM_pad_pattern[1]

            a_ell_offset = max(a_ell_offset, m_blk * nnz_cols + SMEM_pad_num)

            params_start = params_start + 8

        elif fmt[0] == "TensorCore_template":
            # tile_size, group_size = fmt[1][0][1], fmt[1][2][1]
            # mb = math.ceil(tiles[0].op.idx_lens[0] / tile_size)
            # nb = math.ceil(tiles[0].op.idx_lens[2] / group_size)
            # nnzb = len(tiles)
            # feat_size = tiles[0].op.idx_lens[1]
            # mma_n = fmt[1][1][1]
            # feat_size = get_padded_j_TC(feat_size, mma_n)
            # # if (feat_size < mma_n):
            # #     feat_size = mma_n
            # # elif (feat_size > mma_n):
            # #     feat_size = math.ceil(feat_size / (2*mma_n)) * 2*mma_n


            # _, MB, NB, NNZB, F, T, G = func.params[params_start:params_start+7]
            # params_dict[MB] = mb
            # params_dict[NB] = nb
            # params_dict[NNZB] = nnzb
            # params_dict[F] = feat_size
            # params_dict[T] = tile_size
            # params_dict[G] = group_size

            # params_start = params_start + 9

            # max_i = max( max_i, mb * tile_size )
            # max_j = max( max_j, feat_size )


            # =========================================
            tile_size, group_size = fmt[1][0][1], fmt[1][2][1]

            # mb = math.ceil(tiles[0].op.idx_lens[0] / tile_size)
            # 此处考虑类似dbsrmm里的mb，即把空行压缩掉了
            mb = len(set([t.tile_pos[0] for t in tiles])) * repeat_row

            nb = math.ceil(tiles[0].op.idx_lens[2] / group_size)
            nnzb = len(tiles) * repeat_row
            feat_size = tiles[0].op.idx_lens[1]
            # mma_n = fmt[1][1][1]
            mma_n = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])[1]
            feat_size = get_padded_j_TC(feat_size, mma_n)
            
            # funit是被tile_sizes决定的，此处的写法有问题
            # funit = min(2, feat_size // mma_n) * mma_n
            funit = fmt[1][1][1]
            fb = feat_size//funit

            # FOR DEBUG----------
            # funit = mma_n
            # fb = feat_size//funit
            # -------------------


            _, MB, NB, NNZB, FB, FUNIT, T, G = func.params[params_start:params_start+8]
            print(mb, nb, nnzb, fb, funit, tile_size, group_size)
            params_dict[MB] = int(mb)
            params_dict[NB] = int(nb)
            params_dict[NNZB] = int(nnzb)
            params_dict[FB] = int(fb)
            params_dict[FUNIT] = int(funit)
            params_dict[T] = int(tile_size)
            params_dict[G] = int(group_size)

            params_start = params_start + 11 # 10 如果我们使用了cachewrite+ shared memory

            # max_i = max( max_i, mb * tile_size )
            # max_j = max( max_j, feat_size )

            a_ell_offset = max(a_ell_offset, group_size * mma_n)

            params = json.loads(fmt[-1])
            if params['idx_reordered'][0] or params['real_atomic']:
                # 默认当i没有被reorder的时候并且不atomicAdd，不使用C_shared
                if TC_out_SMEM == None:
                    TC_out_SMEM = tile_size * funit
                else:
                    TC_out_SMEM = max(TC_out_SMEM, tile_size * funit)

            # TC_out_SMEM = tile_size * funit


    # set m_tot, n_tot, k_tot
    # max_i = max( max_i, tiles[0].op.idx_lens[0] )
    # max_j = max( max_j, tiles[0].op.idx_lens[1] )

    M_TOT, N_TOT, K_TOT, MAX_SHARED = func.params[2:6] # [2:5]
    params_dict[M_TOT] = int(max_i)
    params_dict[N_TOT] = int(max_j)
    # params_dict[K_TOT] = int(tiles[0].op.idx_lens[2])
    params_dict[K_TOT] = int(max_k)
    params_dict[MAX_SHARED] = int(a_ell_offset)

    print(max_i, max_j, tiles[0].op.idx_lens[2], a_ell_offset)


    # 需要把a_ell_offset也存到json文件里面使得我们在fix cuda bug的时候可以处理
    with open(f"A_ell_max_shared{os.environ['MyFileID']}.json", 'w') as f:
        json.dump((a_ell_offset, TC_out_SMEM), f)


    # print(type(max_i), type(max_j), type(tiles[0].op.idx_lens[2]))

    mod = tvm.IRModule.from_expr(
            func.specialize(params_dict)
        )
    
    # print(mod.script())


    # 因为对于ELL的function定义中我们已经写成了lower完sparse iter之后的形式，再运行这一函数会造成segment fault，所以此处需要修改
    # mod = lower_sparse_iter(mod)
    
    # print(mod.script())

    sch = tir.Schedule(mod)
    
    # print(sch.mod.script())
    return sch, (max_i, max_j)








def schedule_fused_kernel_and_build_BSPMM(op_type, sch, formats, target, save_to_file=False):
    '''
    Input:
        sch: the initial sch with a part of the parameters set.
        formats: the selected formats.
    Output:
        f: the built function.
    '''
    for i, fmt in enumerate(formats):
        tile_sizes, params = fmt[1], json.loads(fmt[2])
        if fmt[0] == "sparse_template":
            func_str = f'csrmm{i}'
            schedule_csr(sch, func_str, tile_sizes, params)
        elif fmt[0] == "sparse_template_ell":
            func_str = f'ellmm{i}'
            in1_shared_str = f'ellmm_shared{i}'
            write_blk_str = f'C_local{i}'
            schedule_ell(op_type, sch, func_str, tile_sizes, params, in1_shared_str, write_blk_str)
        elif fmt[0] == "TensorCore_template":
            func_str = f'tcspmm{i}'

            feat_size = formats[fmt][0].op.idx_lens[1]
            # mma_n = fmt[1][1][1]
            mma_n = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])[1]
            feat_size = get_padded_j_TC(feat_size, mma_n)
            
            if formats[fmt][0].op.TC_k_notsorted:
                schedule_tc_knotsorted(op_type, sch, feat_size, func_str, tile_sizes, params, i)
            else:
                schedule_tc(op_type, sch, feat_size, func_str, tile_sizes, params, i)

    print(sch.mod.script())
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)

    # RemoveUnusedArgs这个函数在改写了ELL的函数定义之后会报错，很奇怪，暂时先注释掉
    # mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    # print(sch.mod.script())

    if save_to_file:
        # 如果save to file，就不对mod进行build
        with open("mod_before_build.py", 'w') as f:
            f.write("import tvm\n")
            f.write("from tvm.script import tir as T\n")
            f.write(mod.script())
        return

    # print(tvm.lower(mod))
    with open("mod_after_lower2.py", 'w') as f:
        f.write(str(tvm.lower(mod)))

    f = tvm.build(mod, target=target)
    return f




def schedule_fused_kernel_and_build_BSDDMM(op_type, sch, formats, target, save_to_file=False):
    '''
    Input:
        sch: the initial sch with a part of the parameters set.
        formats: the selected formats.
    Output:
        f: the built function.
    '''
    for i, fmt in enumerate(formats):
        tile_sizes, params = fmt[1], json.loads(fmt[2])
        if fmt[0] == "1D_sddmm":
            func_str = f'sddmm{i}'
            
            only1D = True # 记录是否仅有一种1D tile （有TC tile 或者有别的1D tile类型【当然这种情况现在不可能】，就都为FALSE）
            if len(formats) > 1:
                # 说明不只有一种1D tile
                only1D = False
            
            blk_num = math.ceil(sum([t.nnz for t in formats[fmt]]) / (params['ty']*params['group_size']))
            # schedule_1D_SDDMM(op_type, sch, func_str, params, only1D, len(formats[fmt])*params['max_bucket_size']//(params['ty']*params['group_size']))
            schedule_1D_SDDMM(op_type, sch, func_str, params, only1D, blk_num)
        elif fmt[0] == "TC_sddmm":
            func_str = f'tcsddmm{i}'
            schedule_TC_SDDMM(op_type, sch, func_str, tile_sizes, params)

    print(sch.mod.script())
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)

    # RemoveUnusedArgs这个函数在改写了ELL的函数定义之后会报错，很奇怪，暂时先注释掉
    # mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    # print(sch.mod.script())

    if save_to_file:
        # 如果save to file，就不对mod进行build
        with open("mod_before_build.py", 'w') as f:
            f.write("import tvm\n")
            f.write("from tvm.script import tir as T\n")
            f.write(mod.script())
        return

    # print(tvm.lower(mod))
    with open("mod_after_lower2.py", 'w') as f:
        f.write(str(tvm.lower(mod)))

    f = tvm.build(mod, target=target)
    return f





def schedule_fused_kernel_and_build(op_type, sch, formats, target, save_to_file=False):
    '''
    Input:
        sch: the initial sch with a part of the parameters set.
        formats: the selected formats.
    Output:
        f: the built function.
    '''
    if op_type in ['spmm', 'batched_spmm']:
        return schedule_fused_kernel_and_build_BSPMM(op_type, sch, formats, target, save_to_file=save_to_file)
    elif op_type in ['sddmm']:
        return schedule_fused_kernel_and_build_BSDDMM(op_type, sch, formats, target, save_to_file=save_to_file)










# =============================================
# 还需要准备input数据，以此测量真实的fused kernel的运行时间。



def prepare_inputs_BSPMM(ori_op, op_type, batch_num, formats, out_shape, dtype, dev):
    '''
    Input:
        formats: the selected formats.
        out_shape: the output shape.
        dtype: the data type for the input values. E.g., the two matrices for spmm.
    Output:
        args_nd: the list of inputs for the scheduled fused kernel.  
        # (max_i, max_j): the output shape.
    '''
    def to_tvm_array(v, dtype, dev):
        return tvm.nd.array(np.array(v).astype(dtype), device=dev)

    args = list()
    # max_i, max_j = -1, -1

    a_ells = list()

    for fmt in formats:
        tiles = formats[fmt]
        if fmt[0] == "sparse_template":
            a = np.concatenate([tile.position_space_when_selected.data for tile in tiles])
            indptr_k = np.empty(sum(tile.position_space_when_selected.shape[0] for tile in tiles) + 1, 
                dtype=tiles[0].position_space_when_selected.indptr.dtype)

            indices_k = np.concatenate(
                [tile.op.idx_values_list[0][2][tile.position_space_when_selected.indices] for tile in tiles]
                )

            # TODO: 此处似乎并没有考虑tile在i轴的range可能超出原input范围的情况，感觉还是得padding，比如有2个tile都需要pad i这种情况。
            indices_reordered_i = np.concatenate(
                [tile.op.idx_values_list[0][0][tile.tile_i_rng[0]:tile.tile_i_rng[1]+1] for tile in tiles]
                )

            # set indices_k
            last_indptr = 0
            sum_dim = 0
            for tile in tiles:
                b = tile.position_space_when_selected
                idxs = slice(sum_dim, sum_dim + b.shape[0])
                indptr_k[idxs] = b.indptr[:-1]
                indptr_k[idxs] += last_indptr
                sum_dim += b.shape[0]
                last_indptr += b.indptr[-1]
            indptr_k[-1] = last_indptr

            args = args + [to_tvm_array(a, dtype, dev) ]+ [to_tvm_array(arg, "int32", dev) for arg in [indptr_k, indices_k, indices_reordered_i]] # [a, indptr_k, indices_k, indices_reordered_i]

            # print(len(a))
            # print("ELL: ", a.shape, indptr_k.shape, indices_k.shape, indices_reordered_i.shape)

        elif fmt[0] == "sparse_template_ell":
            best_tile_sizes = fmt[1]
            ilen_blk, klen_blk = math.prod(best_tile_sizes[0][1:]), math.prod(best_tile_sizes[2][1:])
            # 先初始化
            a = np.zeros(( ilen_blk * len(tiles), klen_blk))
            indices_k = np.concatenate([
                np.full((ilen_blk, klen_blk), tile.op.idx_values_list[0][2][0]) 
                for tile in tiles], axis=0)

            # 希望在一开始准备op搜索tile的时候，就pad op到一个regular的shape，从而避免这一步np.full的默认值起作用。 
            # 因为对于atomic和nonatomic的情况，这个默认值应该不一样。
            indices_reordered_i = np.concatenate([
                np.full(
                    ilen_blk, 
                    tile.i_vals[-1] if tile.tile_i_rng == None else tile.op.idx_values_list[0][0][tile.op.hyb_new_rows[0][-1]],
                    # tile.i_vals[-1],
                    # tile.op.idx_values_list[0][0][tile.op.hyb_new_rows[0][-1]]
                    ) 
                for tile in tiles])

            # set valid values
            for count, tile in enumerate(tiles):
                a_ptr = count * ilen_blk
                covered_csr = tile.position_space_when_selected
                nnzs = covered_csr.getnnz(axis=1)
                idx_values = tile.op.idx_values_list[0]
                hyb_new_rows = tile.op.hyb_new_rows[0]
                for i in range(covered_csr.shape[0]):
                    idxs = slice(0, nnzs[i])
                    a[a_ptr+i][idxs] = covered_csr[i].data
                    indices_k[a_ptr+i][idxs] = idx_values[2][covered_csr[i].indices]
                idxs = slice(a_ptr, a_ptr + covered_csr.shape[0])
                if tile.tile_i_rng == None:
                    indices_reordered_i[idxs] = tile.i_vals
                else:
                    indices_reordered_i[idxs] = idx_values[0][ hyb_new_rows[tile.tile_i_rng[0]: tile.tile_i_rng[1]+1] ]
                # indices_reordered_i[idxs] = tile.i_vals
                # indices_reordered_i[idxs] = idx_values[0][ hyb_new_rows[tile.tile_i_rng[0]: tile.tile_i_rng[1]+1] ]

            # a = a.flatten()
            a = np.reshape(a, ( len(tiles), ilen_blk * klen_blk) )
            # indices_k = indices_k.flatten()

            # print(indices_k.tolist(), indices_reordered_i, a)
            if op_type == 'batched_spmm':
                a = np.asarray([a for i in range(batch_num)])

            args = args + [to_tvm_array(a, dtype, dev)] + [to_tvm_array(arg, "int32", dev) for arg in [indices_k, indices_reordered_i]] #  [a, indices_k, indices_reordered_i]
            # print("ELL: ", a.shape, indices_k.shape, indices_reordered_i.shape)

            # a_ells.append(a)

        elif fmt[0] == "TensorCore_template":
            # 首先要对tiles排序，因为计算的时候需要获取每个block的坐标，要和indptr一致
            tiles = sorted(tiles, key=lambda tile: tile.tile_pos)

            best_tile_sizes = fmt[1]
            ilen_blk, klen_blk = math.prod(best_tile_sizes[0][1:]), math.prod(best_tile_sizes[2][1:])
            # mma_n = best_tile_sizes[1][1]
            mma_n = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])[1]

            # we do not want to modify tile.position_space_when_selected directly
            covered_csrs = [ tile.position_space_when_selected.copy() for tile in tiles ]
            for csr_matrix in covered_csrs:
                csr_matrix.resize((ilen_blk, klen_blk))
            a = np.concatenate(
                [ csr_matrix.toarray(order='C') for csr_matrix in covered_csrs],
                axis = None)
            a = np.reshape(a, (-1, ilen_blk, klen_blk))

            # i_blk_num = math.ceil(ori_op.idx_lens[0] / ilen_blk)
            # poi_is, counts = np.unique([ tile.tile_pos[0] for tile in tiles] + list(range(i_blk_num)), return_counts=True)
            # indptr = np.concatenate( ([0], np.cumsum(counts-1)) )
            # 我们考虑类似dbsrmm中那样，把空行压缩了
            poi_is, counts = np.unique([ tile.tile_pos[0] for tile in tiles], return_counts=True)
            indptr = np.concatenate( ([0], np.cumsum(counts)) )

            indices = np.concatenate([
                np.full(
                    klen_blk, 
                    # tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.tile_k_rng[1]] ]
                    tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.k_vals[-1]] ]
                    )
                for tile in tiles])
            
            indices = np.reshape(indices, (-1, klen_blk))

            for count, tile in enumerate(tiles):
                # print(len(indices[count][:tile.position_space_when_selected.shape[1] ]), 
                #     len(tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.k_vals] ]), 
                #     len(indices[count]), 
                #     tile.position_space_when_selected.shape[1],
                #     tile.get_key())

                indices[count][:tile.position_space_when_selected.shape[1] ] = \
                    tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.k_vals] ]
                    # tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.tile_k_rng[0]:tile.tile_k_rng[1]+1] ]


            # print([len(arg) for arg in [a, indptr, indices]], indptr)
            # indices = indices.flatten()

            # 暂时不考虑i的任意reorder
            # i_indices = np.array([t.tile_pos[0] for t in tiles])
            # 考虑了i的任意reorder之后：
            params = json.loads(fmt[-1])
            if params['idx_reordered'][0] or params['real_atomic']:
                i_indices = [
                    np.full(
                        ilen_blk, 
                        tiles[tile_i].op.idx_values_list[0][0][ tiles[tile_i].tile_i_rng[1] ]
                        )
                    for tile_i in indptr[:-1] ]

                for count, tile_i in enumerate(indptr[:-1]):
                    tile = tiles[tile_i]
                    i_indices[count][:tile.position_space_when_selected.shape[0]] = \
                        tile.op.idx_values_list[0][0][ tile.tile_i_rng[0]:tile.tile_i_rng[1]+1 ]

                # i_indices = np.concatenate(i_indices)
                i_indices = np.array(i_indices)
            
            else:
                # 没有reorder i且不使用atomicAdd, i_indices存的其实就是tile 的 position_i
                i_indices = poi_is

            if op_type == 'batched_spmm':
                a = np.asarray([a for i in range(batch_num)])

            if tiles[0].op.TC_k_notsorted:
                indices = np.asarray([ tile.tile_pos[2] for tile in tiles])

            args = args + [to_tvm_array(a, dtype, dev)] + [to_tvm_array(arg, "int32", dev) for arg in [i_indices, indptr, indices]] #  [a, indptr, indices]

            # print(args[-1].shape)


    # 准备公用的参数
    max_i, max_j = out_shape
    b = None
    if max_j - ori_op.idx_lens[1] > 0:
        b = np.concatenate(( ori_op.inps[1].data, np.zeros((ori_op.idx_lens[2], max_j - ori_op.idx_lens[1])) ), 
            axis=1)#.flatten() #assume in2 is of shape (idx_lens[2]，max_j)
    else:
        b = ori_op.inps[1].data#.flatten()

    print("b.shape", b.shape)

    c = np.zeros( (max_i, max_j) )

    # a_ell = np.concatenate(a_ells)

    if op_type == 'batched_spmm':
        print(b, b.shape, batch_num, list(range(batch_num)))
        b = np.asarray([b for i in range(batch_num)])
        c = np.zeros( (batch_num, max_i, max_j) )

    args = [to_tvm_array(arg, dtype, dev) for arg in [b, c]] + args

    # 将参数转化为tvm array
    # args_nd = [tvm.nd.array(np.array(arg).astype(dtype), device=dev) for arg in args]

    # print("FINISH PREPARE INs")
    # print(args[0].shape)

    return args, list()






def prepare_inputs_BSDDMM(ori_op, op_type, batch_num, formats, out_shape, dtype, dev):
    '''
    Input:
        formats: the selected formats.
        out_shape: the output shape.
        dtype: the data type for the input values. E.g., the two matrices for spmm.
    Output:
        args_nd: the list of inputs for the scheduled fused kernel.  
        # (max_i, max_j): the output shape.
    '''
    def to_tvm_array(v, dtype, dev):
        return tvm.nd.array(np.array(v).astype(dtype), device=dev)

    args = list()
    # max_i, max_j = -1, -1

    c_indices = [list(), list()]
    params_start = 5 # 在每种tile的input parameter之前还有5个公共的input

    for fmt in formats:
        tiles = formats[fmt]
        if fmt[0] == "1D_sddmm":

            # 这些input应该可以直接从tile的信息中获取
            mid_idx = None # the original i index of an non-zero
            j_idx = None # the original j index of an non-zero
            remap = None # the index of an non-zero in the final output

            tile_sizes = fmt[1]
            mid_idx = np.concatenate([t.op.ori_row_ids_1d[ t.tile_pos[0]*tile_sizes[0] : (t.tile_pos[0]+1)*tile_sizes[0] ] for t in tiles])
            j_idx = np.concatenate([t.op.ori_col_ids_1d[ t.tile_pos[0]*tile_sizes[0] : (t.tile_pos[0]+1)*tile_sizes[0] ] for t in tiles])
            remap = np.concatenate([t.op.nnz_id_matrix[ t.tile_pos[0]*tile_sizes[0] : (t.tile_pos[0]+1)*tile_sizes[0] ] for t in tiles])

            max_bucket_size = json.loads(fmt[2])['max_bucket_size']
            to_pad = math.ceil(len(mid_idx)/max_bucket_size)*max_bucket_size - len(mid_idx)
            # to_pad的计算之前有问题，因为我们可能会改变max_bucket_size的大小
            real_max_bucket_size = json.loads(fmt[2])['ty']*json.loads(fmt[2])['group_size']
            assert sum([t.nnz for t in tiles]) == len(mid_idx)
            to_pad = math.ceil(len(mid_idx)/real_max_bucket_size)*real_max_bucket_size - len(mid_idx)

            print(f"Padding infor: {to_pad, len(mid_idx), real_max_bucket_size}")

            mid_idx = np.concatenate([mid_idx, np.full(to_pad, mid_idx[-1])])
            j_idx = np.concatenate([j_idx, np.full(to_pad, j_idx[-1])])
            remap = np.concatenate([remap, np.full(to_pad, remap[-1])])

            print(f"len(mid_idx) : {len(mid_idx)}")

            args = args + [to_tvm_array(arg, "int32", dev) for arg in [mid_idx, j_idx, remap]] #  [a, indices_k, indices_reordered_i]
            # print("ELL: ", a.shape, indices_k.shape, indices_reordered_i.shape)

            if os.environ['REMAP']=='False':
                c = np.zeros(len(remap))
                args = args + [to_tvm_array(c, dtype, dev)]
                c_indices[1].append(len(args)-1+params_start) # +3是因为在args头部还有三个parameter未加

            # a_ells.append(a)

        elif fmt[0] == "TC_sddmm":
            i_indices = None # the original i of a row of a TC tile: [nnzb, block_i]
            j_indices = None # the original j of a column of a TC tile: [nnzb, block_j]
            indptr = None # stores the starting nnz and ending nnz of a TC tile: [nnzb+1]
            indicesDense = None # the index of an non-zero in a flattened TC block: [NNZ]
            indices = None # the index of an non-zero in the final output: [NNZ]

            tile_sizes = fmt[1]
            i_indices = np.asarray([t.op.idx_values_list[0][0][ t.tile_i_rng[0] : t.tile_i_rng[1]+1 ] for t in tiles])
            # j_indices = np.asarray([t.op.idx_values_list[0][2][t.op.k_vals[t.tile_pos[0]]][ t.tile_k_rng[0] : t.tile_k_rng[1]+1 ] for t in tiles])
            
            # 因为indicesDense和indices都没有把重复覆盖的nnz除去，所以此处应该使用的是原始的nnz而不是nnz_when_selected
            # <jingzhi>@revision: do not use t.nnz, but use cscs.nnz below
            # indptr = np.cumsum([0]+[t.nnz for t in tiles]) # the original code adopted which can work
            # indptr = np.cumsum([0]+[t.nnz_when_selected // t.j_num for t in tiles]) # not used

            # coos = [t.position_space_when_selected.tocoo(copy=False) for t in tiles]
            # indicesDense = np.concatenate([np.sort(coo.row*tile_sizes[2][1] + coo.col, axis=None) for coo in coos])

            # 为了加快一点速度，此处换一个写法：
            max_pos_i = max([t.tile_pos[0] for t in tiles])
            assert len(set([t.op.op_id for t in tiles])) == 1
            sub_op = tiles[0].op
            j_lists = [sub_op.idx_values_list[0][2][sub_op.k_vals[i]] for i in range(max_pos_i+1)]
            j_indices = np.asarray([ j_lists[t.tile_pos[0]][ t.tile_k_rng[0] : t.tile_k_rng[1]+1 ] for t in tiles])

            # nnz_id_matrix_list = [sub_op.nnz_id_matrix[ i*tile_sizes[0][1]:(i+1)*tile_sizes[0][1], : ][ :, sub_op.k_vals[i] ] for i in range(max_pos_i+1)]
            # csrs = [nnz_id_matrix_list[t.tile_pos[0]][ :, t.tile_k_rng[0]:t.tile_k_rng[1]+1  ] for t in tiles]
            # indicesDense = np.concatenate([ np.repeat(np.arange(csr.shape[0]), csr.getnnz(axis=1))*tile_sizes[2][1] + csr.indices for csr in csrs])
            # indices = np.concatenate([csr.data-1 for csr in csrs])
            nnz_id_matrix_list = [(sub_op.nnz_id_matrix[ i*tile_sizes[0][1]:(i+1)*tile_sizes[0][1], : ][ :, sub_op.k_vals[i] ]).tocsc() for i in range(max_pos_i+1)]
            cscs = [nnz_id_matrix_list[t.tile_pos[0]][ :, t.tile_k_rng[0]:t.tile_k_rng[1]+1  ] for t in tiles]
            indicesDense = np.concatenate([ np.repeat(np.arange(csc.shape[1]), csc.getnnz(axis=0)) + csc.indices*tile_sizes[2][1] for csc in cscs])
            indices = np.concatenate([csc.data-1 for csc in cscs])
            
            # <jingzhi>@revision: gen indptr based on cscs
            indptr = np.cumsum([0]+[csc.nnz for csc in cscs])

            # 
            # csrs = [t.op.nnz_id_matrix[ t.tile_i_rng[0]:t.tile_i_rng[1]+1, : ][ :, t.op.k_vals[t.tile_pos[0]][ t.tile_k_rng[0]:t.tile_k_rng[1]+1 ]  ].tocsr() for t in tiles]
            # for csr in csrs:
            #     csr.has_sorted_indices = False
            #     csr.sort_indices()
            # indices = np.concatenate([csr.data-1 for csr in csrs])

            print(f"len(indicesDense) : {len(indicesDense)}")


            args = args + [to_tvm_array(arg, "int32", dev) for arg in [i_indices, j_indices, indptr, indicesDense, indices]] #  [a, indptr, indices]

            if os.environ['REMAP']=='False':
                c = np.zeros((len(tiles), tile_sizes[0][1], tile_sizes[2][1]))
                print(c.shape)
                args = args + [to_tvm_array(c, dtype, dev)]
                c_indices[0].append(len(args)-1+params_start) # +3是因为在args头部还有三个parameter未加

            # print(args[-1].shape)


    # 准备公用的参数
    a = ori_op.inps[1].data
    b = ori_op.inps[2].data#.flatten()
    # c = np.zeros( ori_op.inps[0].nnz )
    # 为了让我们只测一部分selected tiles的时候也能成功运行，所以要和set parameter的时候一致
    c = np.zeros( int(sum([ t.nnz_when_selected//t.j_num for tiles in formats.values() for t in tiles ])) )

    args = [to_tvm_array(arg, dtype, dev) for arg in [a, b, c]] + args

    args = args[:3]+args[:2]+args[3:]

    # 将参数转化为tvm array
    # args_nd = [tvm.nd.array(np.array(arg).astype(dtype), device=dev) for arg in args]

    # print("FINISH PREPARE INs")
    # print(args[0].shape)

    return args, c_indices






def prepare_inputs(ori_op, op_type, batch_num, formats, out_shape, dtype, dev):
    '''
    Input:
        formats: the selected formats.
        out_shape: the output shape.
        dtype: the data type for the input values. E.g., the two matrices for spmm.
    Output:
        args_nd: the list of inputs for the scheduled fused kernel.  
        # (max_i, max_j): the output shape.
    '''
    if op_type in ['spmm', 'batched_spmm']:
        return prepare_inputs_BSPMM(ori_op, op_type, batch_num, formats, out_shape, dtype, dev)
    elif op_type in ['sddmm']:
        return prepare_inputs_BSDDMM(ori_op, op_type, batch_num, formats, out_shape, dtype, dev)





def prepare_inputs_based_on_gened_inputs(gened_inputs, tiles_1d, op_type, dtype, dev):
    def to_tvm_array(v, dtype, dev):
        return tvm.nd.array(np.array(v).astype(dtype), device=dev)

    assert op_type == 'sddmm'
    args_nd, c_indices = gened_inputs
    if len(c_indices[1]) == 0:
        # no 1d tiles selected
        return
    c = args_nd[c_indices[1][0]]
    nnz = sum([t.nnz for t in tiles_1d])
    real_max_bucket_size = tiles_1d[0].params['ty']*tiles_1d[0].params['group_size']
    nnz_padded = math.ceil(nnz/real_max_bucket_size)*real_max_bucket_size
    if nnz_padded < len(c.numpy()):
        for i in range(c_indices[1][0]-3, c_indices[1][0]):
            args_nd[i] = to_tvm_array(args_nd[i].numpy()[:nnz_padded], "int32", dev)
        args_nd[c_indices[1][0]] = to_tvm_array(args_nd[c_indices[1][0]].numpy()[:nnz_padded], dtype, dev)
    elif nnz_padded > len(c.numpy()):
        to_pad = nnz_padded - len(c.numpy())
        for i in range(c_indices[1][0]-3, c_indices[1][0]):
            args = args_nd[i].numpy()
            args = np.concatenate([args, np.full(to_pad, args[-1])])
            args_nd[i] = to_tvm_array(args, "int32", dev)
        args = args_nd[c_indices[1][0]].numpy()
        args = np.concatenate([args, np.full(to_pad, 0)])
        args_nd[c_indices[1][0]] = to_tvm_array(args, dtype, dev)


def prepare_inputs_BSDDMM_one_tile(ori_op, op_type, batch_num, formats, out_shape, dtype, dev, row_window_width = 1):
    '''
    Input:
        formats: the selected formats.
        out_shape: the output shape.
        dtype: the data type for the input values. E.g., the two matrices for spmm.
        row_window_width: the number of TC blocks per row window. For general benchmark, row_window_width=1.
    Output:
        args_nd: the list of inputs for the scheduled fused kernel.  
        # (max_i, max_j): the output shape.
    NOTE: this method is only used for benchmarking TC tiles, so
        the tiles in formats are all the same, i.e., the same as the tile to be measured.
        we can directly use the data stored in ori_op.
    '''
    def to_tvm_array(v, dtype, dev):
        return tvm.nd.array(np.array(v).astype(dtype), device=dev)

    args = list()
    # max_i, max_j = -1, -1

    c_indices = [list(), list()]
    params_start = 5 # 在每种tile的input parameter之前还有5个公共的input

    for fmt in formats:
        tiles = formats[fmt]
        if fmt[0] == "1D_sddmm":
            assert False, "Not support 1D_sddmm!"
        elif fmt[0] == "TC_sddmm":
            i_indices = None # the original i of a row of a TC tile: [nnzb, block_i]
            j_indices = None # the original j of a column of a TC tile: [nnzb, block_j]
            indptr = None # stores the starting nnz and ending nnz of a TC tile: [nnzb+1]
            indicesDense = None # the index of an non-zero in a flattened TC block: [NNZ]
            indices = None # the index of an non-zero in the final output: [NNZ]

            # 我们现在的策略是fake op的sparse tensor的布局按照行来，有row_window_num行，每一行有row_window_width个block
            row_window_num = len(tiles) // row_window_width

            sub_op = tiles[0].op
            tile_sizes = fmt[1]
            i_indices = sub_op.idx_values_list[0][0].reshape((-1, tile_sizes[0][1]))[:row_window_num]
            i_indices = np.repeat(i_indices, row_window_width, axis=0)
            indptr = np.cumsum([0]+[t.nnz_when_selected // t.j_num for t in tiles])

            # max_pos_i = len(tiles)-1
            j_lists = [sub_op.idx_values_list[0][2] for i in range(row_window_num)]
            j_indices = np.asarray([ j_lists[i][ i*(tile_sizes[2][1]*row_window_width) : (i+1)*(tile_sizes[2][1]*row_window_width) ] for i in range(row_window_num)])
            j_indices = j_indices.reshape((-1, tile_sizes[2][1]))
            indicesDense = np.concatenate([np.arange(tile_sizes[0][1]*tile_sizes[2][1]) for i in range(row_window_num*row_window_width)])
            indices = np.arange(row_window_num*row_window_width*tile_sizes[0][1]*tile_sizes[2][1])
            # 

            print("input shapes: ", i_indices.shape, j_indices.shape, indptr.shape, indicesDense.shape, indices.shape)

            args = args + [to_tvm_array(arg, "int32", dev) for arg in [i_indices, j_indices, indptr, indicesDense, indices]] #  [a, indptr, indices]

            if os.environ['REMAP']=='False':
                c = np.zeros((row_window_num*row_window_width, tile_sizes[0][1], tile_sizes[2][1]))
                print(c.shape)
                args = args + [to_tvm_array(c, dtype, dev)]
                c_indices[0].append(len(args)-1+params_start) # +3是因为在args头部还有三个parameter未加

            # print(args[-1].shape)


    # 准备公用的参数
    a = ori_op.inps[1].data
    b = ori_op.inps[2].data#.flatten()
    # c = np.zeros( ori_op.inps[0].nnz )

    c = np.zeros( int(sum([ t.nnz_when_selected//t.j_num for tiles in formats.values() for t in tiles ])) )

    args = [to_tvm_array(arg, dtype, dev) for arg in [a, b, c]] + args

    args = args[:3]+args[:2]+args[3:]

    # 将参数转化为tvm array
    # args_nd = [tvm.nd.array(np.array(arg).astype(dtype), device=dev) for arg in args]

    # print("FINISH PREPARE INs")
    # print(args[0].shape)

    return args, c_indices







def prepare_inputs_measure_one_tile(ori_op, formats, dtype, dev):
    '''
    Input:
        formats: the selected formats.
        out_shape: the output shape.
        dtype: the data type for the input values. E.g., the two matrices for spmm.
    Output:
        args_nd: the list of inputs for the scheduled fused kernel.  
        # (max_i, max_j): the output shape.
    '''
    def to_tvm_array(v, dtype, dev):
        return tvm.nd.array(np.array(v).astype(dtype), device=dev)


    def reset_indices(indices):
        tmp_v = np.array(sorted(set(indices)))
        tmp_idx = np.argsort(tmp_v)
        tmp = np.arange(np.amax(indices)+1)
        tmp[tmp_v] = tmp_idx
        indices = tmp[indices]
        return indices

    args = list()
    # max_i, max_j = -1, -1
    op_i_start, op_i_stride, op_k_start, op_k_stride = None, None, None, None

    a_ells = list()

    for fmt in formats:
        tiles = formats[fmt]
        repeat_row = 108*100 // (tiles[0].op.idx_lens[1] // math.prod(tiles[0].tile_sizes[1][1:])) # 108*2
        if fmt[0] == "sparse_template":
            op_K = tiles[0].op.idx_lens[2]
            op_i_start = np.amin(tiles[0].op.idx_values_list[0][0][tiles[0].tile_i_rng[0]:tiles[0].tile_i_rng[1]+1])
            op_i_stride = np.amax(tiles[0].op.idx_values_list[0][0][tiles[0].tile_i_rng[0]:tiles[0].tile_i_rng[1]+1]) - op_i_start + 2
            op_k_start = np.amin(tiles[0].op.idx_values_list[0][2][tiles[0].position_space_when_selected.indices])
            op_k_stride = np.amax(tiles[0].op.idx_values_list[0][2][tiles[0].position_space_when_selected.indices]) - op_k_start + 2


            op_i_start = 0
            op_i_stride = (tiles[0].tile_i_rng[1] - tiles[0].tile_i_rng[0]) * 2 + 2
            op_k_start = 0
            op_k_stride = len(set(tiles[0].op.idx_values_list[0][2][tiles[0].position_space_when_selected.indices])) * 2 + 2


            a = np.concatenate([tile.position_space_when_selected.data for tile in tiles for i in range(repeat_row)])
            indptr_k = np.empty(sum(tile.position_space_when_selected.shape[0] for tile in tiles) * repeat_row + 1, 
                dtype=tiles[0].position_space_when_selected.indptr.dtype)

            indices_k = np.concatenate(
                [tile.op.idx_values_list[0][2][tile.position_space_when_selected.indices] - op_k_start + op_k_stride * i \
                    for tile in tiles for i in range(repeat_row)]
                )


            # reset k completely
            indices_k = tiles[0].op.idx_values_list[0][2][tiles[0].position_space_when_selected.indices]
            indices_k = reset_indices(indices_k)
            indices_k = np.concatenate(
                [indices_k - op_k_start + op_k_stride * i for i in range(repeat_row)]
                )


            # TODO: 此处似乎并没有考虑tile在i轴的range可能超出原input范围的情况，感觉还是得padding，比如有2个tile都需要pad i这种情况。
            indices_reordered_i = np.concatenate(
                [tile.op.idx_values_list[0][0][tile.tile_i_rng[0]:tile.tile_i_rng[1]+1] - op_i_start + op_i_stride * i for tile in tiles for i in range(repeat_row)]
                )


            # reset indices_reordered_i completely
            indices_reordered_i = tiles[0].op.idx_values_list[0][0][tiles[0].tile_i_rng[0]:tiles[0].tile_i_rng[1]+1]
            indices_reordered_i = reset_indices(indices_reordered_i)
            indices_reordered_i = np.concatenate(
                [indices_reordered_i - op_i_start + op_i_stride * i for i in range(repeat_row)]
                )


            # set indices_k
            last_indptr = 0
            sum_dim = 0
            for tile in tiles:
                for i in range(repeat_row):
                    b = tile.position_space_when_selected
                    idxs = slice(sum_dim, sum_dim + b.shape[0])
                    indptr_k[idxs] = b.indptr[:-1]
                    indptr_k[idxs] += last_indptr
                    sum_dim += b.shape[0]
                    last_indptr += b.indptr[-1]
            indptr_k[-1] = last_indptr

            args = args + [to_tvm_array(a, dtype, dev) ]+ [to_tvm_array(arg, "int32", dev) for arg in [indptr_k, indices_k, indices_reordered_i]] # [a, indptr_k, indices_k, indices_reordered_i]

            max_i = np.amax(indices_reordered_i) + 1
            max_k = np.amax(indices_k) + 1

            # print(len(a))
            # print("ELL: ", a.shape, indptr_k.shape, indices_k.shape, indices_reordered_i.shape)

        elif fmt[0] == "sparse_template_ell":
            op_K = tiles[0].op.idx_lens[2]

            best_tile_sizes = fmt[1]
            ilen_blk, klen_blk = math.prod(best_tile_sizes[0][1:]), math.prod(best_tile_sizes[2][1:])
            # 先初始化
            a = np.zeros(( ilen_blk * len(tiles), klen_blk))
            indices_k = np.concatenate([
                np.full((ilen_blk, klen_blk), tile.op.idx_values_list[0][2][0]) 
                for tile in tiles], axis=0)

            indices_reordered_i = np.concatenate([
                np.full(
                    ilen_blk, 
                    tile.i_vals[-1] if tile.tile_i_rng == None else tile.op.idx_values_list[0][0][tile.op.hyb_new_rows[0][-1]],
                    # tile.i_vals[-1],
                    # tile.op.idx_values_list[0][0][tile.op.hyb_new_rows[0][-1]]
                    ) 
                for tile in tiles])

            # set valid values
            for count, tile in enumerate(tiles):
                a_ptr = count * ilen_blk
                covered_csr = tile.position_space_when_selected
                nnzs = covered_csr.getnnz(axis=1)
                idx_values = tile.op.idx_values_list[0]
                hyb_new_rows = tile.op.hyb_new_rows[0]
                for i in range(covered_csr.shape[0]):
                    idxs = slice(0, nnzs[i])
                    a[a_ptr+i][idxs] = covered_csr[i].data
                    indices_k[a_ptr+i][idxs] = idx_values[2][covered_csr[i].indices]
                idxs = slice(a_ptr, a_ptr + covered_csr.shape[0])
                if tile.tile_i_rng==None:
                    indices_reordered_i[idxs] = tile.i_vals
                else:
                    indices_reordered_i[idxs] = idx_values[0][ hyb_new_rows[tile.tile_i_rng[0]: tile.tile_i_rng[1]+1] ]
                # indices_reordered_i[idxs] = tile.i_vals
                # indices_reordered_i[idxs] = idx_values[0][ hyb_new_rows[tile.tile_i_rng[0]: tile.tile_i_rng[1]+1] ]



            op_k_start = np.amin(indices_k)
            op_k_stride = np.amax(indices_k) - op_k_start + 2  
            op_i_start = np.amin(indices_reordered_i)
            op_i_stride = np.amax(indices_reordered_i) - op_i_start + 2


            op_i_start = 0
            # op_i_stride = (tiles[0].tile_i_rng[1] - tiles[0].tile_i_rng[0]) * 2 + 2
            op_i_stride = tiles[0].tile_sizes[0][1] * 2 + 2
            op_k_start = 0
            op_k_stride = len(set(indices_k.flatten())) * 2 + 2


            # indices_k = np.concatenate( [ indices_k - op_k_start + op_k_stride * i for i in range(repeat_row)] )
            # indices_reordered_i = np.concatenate( [ indices_reordered_i - op_i_start + op_i_stride * i for i in range(repeat_row) ] )

            indices_k = reset_indices(indices_k.flatten()).reshape(indices_k.shape)
            indices_k = np.concatenate(
                [indices_k - op_k_start + op_k_stride * i for i in range(repeat_row)]
                )

            indices_reordered_i = reset_indices(indices_reordered_i)
            indices_reordered_i = np.concatenate(
                [indices_reordered_i - op_i_start + op_i_stride * i for i in range(repeat_row)]
                )
            # a = a.flatten()

            a = np.concatenate([a for i in range(repeat_row)])
            a = np.reshape(a, ( len(tiles)*repeat_row, ilen_blk * klen_blk) )
            # indices_k = indices_k.flatten()

            # print(indices_k)
            args = args + [to_tvm_array(a, dtype, dev)] + [to_tvm_array(arg, "int32", dev) for arg in [indices_k, indices_reordered_i]] #  [a, indices_k, indices_reordered_i]
            # print("ELL: ", a.shape, indices_k.shape, indices_reordered_i.shape)

            # a_ells.append(a)

            max_i = np.amax(indices_reordered_i) + 1
            max_k = np.amax(indices_k) + 1

        elif fmt[0] == "TensorCore_template":

            # 首先要对tiles排序，因为计算的时候需要获取每个block的坐标，要和indptr一致
            tiles = sorted(tiles, key=lambda tile: tile.tile_pos)

            best_tile_sizes = fmt[1]
            ilen_blk, klen_blk = math.prod(best_tile_sizes[0][1:]), math.prod(best_tile_sizes[2][1:])
            # mma_n = best_tile_sizes[1][1]
            mma_n = parse_mma_shape(json.loads(fmt[2])["mma_shape_str"])[1]

            # we do not want to modify tile.position_space_when_selected directly
            covered_csrs = [ tile.position_space_when_selected.copy() for tile in tiles ]
            for csr_matrix in covered_csrs:
                csr_matrix.resize((ilen_blk, klen_blk))
            a = np.concatenate(
                [ csr_matrix.toarray(order='C') for csr_matrix in covered_csrs],
                axis = None)
            a = np.concatenate([a for i in range(repeat_row)])
            a = np.reshape(a, (-1, ilen_blk, klen_blk))

            # i_blk_num = math.ceil(ori_op.idx_lens[0] / ilen_blk)
            # poi_is, counts = np.unique([ tile.tile_pos[0] for tile in tiles] + list(range(i_blk_num)), return_counts=True)
            # indptr = np.concatenate( ([0], np.cumsum(counts-1)) )
            # 我们考虑类似dbsrmm中那样，把空行压缩了
            poi_is, counts = np.unique([ tile.tile_pos[0] for tile in tiles], return_counts=True)
            indptr = np.concatenate( ([0], np.cumsum(counts)) )

            indices = np.concatenate([
                np.full(
                    klen_blk, 
                    # tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.tile_k_rng[1]] ]
                    tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.k_vals[-1]] ]
                    )
                for tile in tiles])
            
            indices = np.reshape(indices, (-1, klen_blk))

            for count, tile in enumerate(tiles):
                indices[count][:tile.position_space_when_selected.shape[1] ] = \
                    tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.k_vals] ]
                    # tile.op.idx_values_list[0][2][ tile.op.k_vals[tile.tile_pos[0]][tile.tile_k_rng[0]:tile.tile_k_rng[1]+1] ]


            # print([len(arg) for arg in [a, indptr, indices]], indptr)
            # indices = indices.flatten()

            # 暂时不考虑i的任意reorder
            # i_indices = np.array([t.tile_pos[0] for t in tiles])
            # 考虑了i的任意reorder之后：
            params = json.loads(fmt[-1])
            if params['idx_reordered'][0] or params['real_atomic']:
                i_indices = [
                    np.full(
                        ilen_blk, 
                        tiles[tile_i].op.idx_values_list[0][0][ tiles[tile_i].tile_i_rng[1] ]
                        )
                    for tile_i in indptr[:-1] ]

                for count, tile_i in enumerate(indptr[:-1]):
                    tile = tiles[tile_i]
                    i_indices[count][:tile.position_space_when_selected.shape[0]] = \
                        tile.op.idx_values_list[0][0][ tile.tile_i_rng[0]:tile.tile_i_rng[1]+1 ]

                # i_indices = np.concatenate(i_indices)
                i_indices = np.array(i_indices)
            
            else:
                # 没有reorder i且不使用atomicAdd, i_indices存的其实就是tile 的 position_i
                # i_indices = poi_is
                # 为了方便后面扩展i_indices
                i_indices = np.array([poi_is])

            print(params['idx_reordered'][0] or params['real_atomic'], "i_indices: ", i_indices)

            op_i_start = np.amin(i_indices[0])
            op_i_stride = np.amax(i_indices[0]) - op_i_start + 2
            # i_indices = np.concatenate( [i_indices - op_i_start + op_i_stride * i for i in range(repeat_row)] )

            op_i_start = 0
            op_i_stride = len(set(i_indices[0])) * 2 + 2
            i_indices = np.array([reset_indices(i_indices[0])])
            i_indices = np.concatenate( [i_indices - op_i_start + op_i_stride * i for i in range(repeat_row)] )

            if not (params['idx_reordered'][0] or params['real_atomic']):
                i_indices = np.concatenate(i_indices)


            indptr = np.concatenate([indptr[1:] + indptr[-1] * i for i in range(repeat_row)])
            indptr = np.concatenate( ([0], indptr) )

            op_k_start = np.amin(indices)
            op_k_stride = np.amax(indices) - op_k_start + 2
            # indices = np.concatenate([indices - op_k_start + op_k_stride * i for i in range(repeat_row) ])

            op_k_start = 0
            op_k_stride = len(set(indices.flatten())) * 2 + 2
            indices = reset_indices(indices.flatten()).reshape(indices.shape)
            indices = np.concatenate([indices - op_k_start + op_k_stride * i for i in range(repeat_row) ])


            args = args + [to_tvm_array(a, dtype, dev)] + [to_tvm_array(arg, "int32", dev) for arg in [i_indices, indptr, indices]] #  [a, indptr, indices]

            # print(a.shape, i_indices.shape, indptr.shape, indices.shape)
            print("i_indices: ", i_indices)
            print("indptr: ", indptr)
            print("indices: ", indices)


            max_i = np.amax(i_indices) + 1
            if not (params['idx_reordered'][0] or params['real_atomic']):
                max_i = (np.amax(i_indices)+1) * ilen_blk # tile_sizes[0][1]
            max_k = np.amax(indices) + 1


    # 准备公用的参数
    # max_i, max_j = out_shape

    # max_i = max(op_i_start + op_i_stride - 1, max_i)
    print("op_i_start, op_i_stride, op_k_start, op_k_stride, repeat_row: ", op_i_start, op_i_stride, op_k_start, op_k_stride,repeat_row)
    # max_i = op_i_stride * (repeat_row - 1) + op_i_stride - 1
    # max_k = max(op_k_stride * (repeat_row - 1) + op_k_stride - 1, ori_op.idx_lens[2])
    max_j = max(ori_op.idx_lens[1], math.ceil(ori_op.idx_lens[1] / math.prod(list(formats.keys())[0][1][1][1:])) * math.prod(list(formats.keys())[0][1][1][1:]))

    b = None
    # if max_j - ori_op.idx_lens[1] > 0:
    #     b = np.concatenate(( ori_op.inps[1].data, np.zeros((ori_op.idx_lens[2], max_j - ori_op.idx_lens[1])) ), 
    #         axis=1)#.flatten() #assume in2 is of shape (idx_lens[2]，max_j)
    # else:
    #     b = ori_op.inps[1].data#.flatten()

    b = np.random.rand(max_k,max_j)

    c = np.zeros( (max_i, max_j) )

    # a_ell = np.concatenate(a_ells)

    args = [to_tvm_array(arg, dtype, dev) for arg in [b, c]] + args

    # 将参数转化为tvm array
    # args_nd = [tvm.nd.array(np.array(arg).astype(dtype), device=dev) for arg in args]

    # print("FINISH PREPARE INs")
    # print(args[0].shape)

    return args, (max_i, max_j, max_k)







def measure_latency_BSPMM(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th):
    '''
        INPUT:
            ori_op: the original op to be optimized.
            f: the optimized kernel.
            args_nd: the tvm array parameters.
            out_shape: the output shape of the op after scheduling.
            dtype: the data type used for ori_op.
            dev: the tvm device to run. 
            dev_th: the pytorch device. E.g., cuda2 = torch.device('cuda:2')
        OUTPUT:
            the latency of the operator after optimization.
    '''

    # f.export_library("f_build.tar")
    # f = tvm.runtime.module.load_module("f_build")
    # f.time_evaluator(f.entry_name, dev, number=100)

    def to_tvm_array(v, dtype, dev):
        return tvm.nd.array(np.array(v).astype(dtype), device=dev)

    f(*args_nd)
    print("measure result!: ", profile_tvm_ms(f, args_nd))
    args_nd[1] = to_tvm_array(np.zeros(args_nd[1].shape), dtype, dev)
    f(*args_nd)

    # check accuracy
    try:
        dtype_th = None
        if dtype == 'float16':
            # TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
            dtype_th = torch.float16 # torch.float32 # torch.float16
        elif dtype == 'float32':
            dtype_th = torch.float32

        # A_torch = torch.sparse_csr_tensor(torch.tensor(ori_op.inps[0].indptr, dtype=torch.int32),
        #                                 torch.tensor(ori_op.inps[0].indices, dtype=torch.int32),
        #                                 torch.tensor(ori_op.inps[0].data, dtype=dtype_th), 
        #                                 size=(ori_op.idx_lens[0], ori_op.idx_lens[2]),
        #                                 dtype=dtype_th, device=dev_th)

        
        B_data = np.array(ori_op.inps[1].data).astype(dtype)
        # y_golden = torch.sparse.mm(A_torch, torch.tensor(B_data, dtype=dtype_th, device=dev_th)).cpu().numpy()

        y_golden = ori_op.inps[0].astype(dtype).dot(B_data)

        if op_type == 'batched_spmm':
            y_golden = np.asarray([y_golden for i in range(batch_num)])

        out_nd = args_nd[1]
        if op_type == 'spmm':
            tvm.testing.assert_allclose(
                    out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]], #[:,idx_values[1]], 
                    y_golden, 
                    rtol=1e-2, atol=1e-2
                )
        elif op_type == 'batched_spmm':
            tvm.testing.assert_allclose(
                    out_nd.numpy().reshape([-1,] + list(out_shape))[:, :ori_op.idx_lens[0] ,:][:, :,:ori_op.idx_lens[1]], #[:,idx_values[1]], 
                    y_golden, 
                    rtol=1e-2, atol=1e-2
                )
    except Exception as e:
        print("--------accuracy error")
        print(e)
        # print(list(out_nd.numpy().reshape(-1, op.idx_lens[1])[idx_values[0],:][:,idx_values[1]]))
        # print(list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]]))
        # print(list(y_golden.cpu().numpy()))
        # my_res = list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]])
        # th_res = list(y_golden)
        # for row_i, (i,j) in enumerate(zip(my_res, th_res)):
        #     # if max(i) > 1e9:
        #     if (not (i==j).all()) and (min(i)>0):
        #         print(row_i, i, j)
        #     # if row_i!=0:
        #     #     continue
        #     if list(i) != list(j):
        #         for col_i, (ii, ji) in enumerate(zip(i,j)):
        #             if (abs(1-ji/ii)>0.01): # and (abs(1-ji/ii)>1.01):
        #                 print(f'{(row_i, col_i)}: {ii}, {ji}, {abs(1-ji/ii)}')

    # print("start measuring")
    print("f.entry_name: ", f.entry_name)
    # evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    # cost = evaluator(*args_nd).mean
    # print("tc-spmm time: {:.5f}ms".format(cost * 1000))

    cost = profile_tvm_ms(f, args_nd)

    return cost   




def order_output_BSDDMM(c_indices, args_nd, nnz):
    # 如果我们在运算的时候不对结果remap，那比较正确性的时候就需要remap，这个函数用于在比较正确性的时候remap

    print(c_indices)
    print([i.numpy().shape for i in args_nd])

    c_1d = None
    remap_1d = list()
    if len(c_indices[1])>0:
        c_1d = args_nd[c_indices[1][0]].numpy()
        remap_1d = args_nd[c_indices[1][0]-1].numpy()
    # 
    c_TCs = [args_nd[i].numpy().reshape(( args_nd[i].numpy().shape[0], -1)) for i in c_indices[0]]
    indptrs = [args_nd[i-3].numpy() for i in c_indices[0]]
    indicesDenses = [args_nd[i-2].numpy() for i in c_indices[0]]
    indices_list = [args_nd[i-1].numpy() for i in c_indices[0]]

    print([i.shape for i in c_TCs])
    print([i.shape for i in indptrs])
    print([i.shape for i in indicesDenses])
    print([i.shape for i in indices_list])

    # print(np.count_nonzero(c_TCs[0]))

    # c_out = np.zeros(len(args_nd[2].numpy()))
    c_out = np.zeros(nnz)

    for i in range(len(c_TCs)):
        c_TC = c_TCs[i]
        indptr = indptrs[i]
        indicesDense = indicesDenses[i]
        indices = indices_list[i]
        c_out[indices] = np.concatenate([c_TC[j][ indicesDense[indptr[j]:indptr[j+1]] ] for j in range(len(c_TC))])

    if remap_1d != list():
        c_out[remap_1d] = c_1d # [:len(remap_1d)]

    print(np.count_nonzero(c_out))
    print(c_out)
    print(len(c_out))

    return c_out



def measure_latency_BSDDMM(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th, cuda_i, c_indices):
    '''
        INPUT:
            ori_op: the original op to be optimized.
            f: the optimized kernel.
            args_nd: the tvm array parameters.
            out_shape: the output shape of the op after scheduling.
            dtype: the data type used for ori_op.
            dev: the tvm device to run. 
            dev_th: the pytorch device. E.g., cuda2 = torch.device('cuda:2')
        OUTPUT:
            the latency of the operator after optimization.
    '''

    # f.export_library("f_build.tar")
    # f = tvm.runtime.module.load_module("f_build")
    # f.time_evaluator(f.entry_name, dev, number=100)

    print("start measuring----------")
    f(*args_nd)
    print("measure result!: ", profile_tvm_ms(f, args_nd))
    order_output_BSDDMM(c_indices, args_nd, ori_op.inps[0].nnz)

    # check accuracy
    try:
        dtype_th = None
        if dtype == 'float16':
            # TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
            dtype_th = torch.float16 # torch.float32 # torch.float16
        elif dtype == 'float32':
            dtype_th = torch.float32

        # g = dgl.graph(('csr', (ori_op.inps[0].indptr, ori_op.inps[0].indices, []))).to(cuda_i)
        # A = torch.from_numpy(np.array(ori_op.inps[1].data)).to(dtype_th).to(cuda_i)
        # # 因为我们在准备ori_op.inps[0]的时候对i，j轴的padding不一样，所以此处截取一样的长度，使u_dot_v能正常计算
        # B = torch.from_numpy(np.array(ori_op.inps[2].data)[:ori_op.idx_lens[0]]).to(dtype_th).to(cuda_i)
        # c_golden = dgl.ops.u_dot_v(g, A, B)

        # 此处计算c_golden有问题，因为不一定是B比A更大，所以更为general的解决方案是直接把g pad成长宽一样的样子
        node_num = max(*(ori_op.inps[0].shape))
        indptr = ori_op.inps[0].indptr
        indptr = np.concatenate([indptr, np.full(node_num+1-len(indptr), indptr[-1]) ])
        g = dgl.graph(('csr', (indptr, ori_op.inps[0].indices, []))).to(cuda_i)
        A = np.array(ori_op.inps[1].data)
        A = np.concatenate([A, np.full((node_num-A.shape[0], A.shape[1]), 0)], axis=0)
        A = torch.from_numpy(A).to(dtype_th).to(cuda_i)
        B = np.array(ori_op.inps[2].data)
        B = np.concatenate([B, np.full((node_num-B.shape[0], B.shape[1]), 0)], axis=0)
        B = torch.from_numpy(B).to(dtype_th).to(cuda_i)
        c_golden = dgl.ops.u_dot_v(g, A, B) 

        
        out_nd = args_nd[2].numpy()

        if os.environ['REMAP'] == 'False':
            out_nd = order_output_BSDDMM(c_indices, args_nd, ori_op.inps[0].nnz)

        if op_type == 'sddmm':
            tvm.testing.assert_allclose(
                    out_nd, #[:,idx_values[1]], 
                    c_golden.view(-1).cpu(), 
                    rtol=1e-2, atol=1e-2
                )
    except Exception as e:
        print("--------accuracy error")
        print(e)
        print(c_golden.view(-1).cpu())
        # print(list(out_nd.numpy().reshape(-1, op.idx_lens[1])[idx_values[0],:][:,idx_values[1]]))
        # print(list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]]))
        # print(list(y_golden.cpu().numpy()))
        # my_res = list(out_nd.numpy())
        # th_res = list(c_golden.view(-1).cpu())
        # for row_i, (i,j) in enumerate(zip(my_res, th_res)):
        #     # if row_i!=0:
        #     #     continue
        #     if list(i) != list(j):
        #         for col_i, (ii, ji) in enumerate(zip(i,j)):
        #             if (abs(1-ji/ii)>0.01): # and (abs(1-ji/ii)>1.01):
        #                 print(f'{(row_i, col_i)}: {ii}, {ji}, {abs(1-ji/ii)}')

    # print("start measuring")
    print("f.entry_name: ", f.entry_name)
    # evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    # cost = evaluator(*args_nd).mean
    # print("tc-spmm time: {:.5f}ms".format(cost * 1000))

    cost = profile_tvm_ms(f, args_nd)

    return cost




def measure_latency(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th, cuda_i, c_indices):
    if op_type in ['spmm', 'batched_spmm']:
        return measure_latency_BSPMM(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th)
    elif op_type == 'sddmm':
        return measure_latency_BSDDMM(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th, cuda_i, c_indices)




def measure_latency_one_tile(selected_tile, ori_op, f, args_nd, out_shape, dtype, dev, dev_th):
    '''
    NOTE: this function only measures the latency of one selected tile, 
        so the accuracy checking part is different with the above one.

        INPUT:
            ori_op: the original op to be optimized.
            f: the optimized kernel.
            args_nd: the tvm array parameters.
            out_shape: the output shape of the op after scheduling.
            dtype: the data type used for ori_op.
            dev: the tvm device to run. 
            dev_th: the pytorch device. E.g., cuda2 = torch.device('cuda:2')
        OUTPUT:
            the latency of the operator after optimization.
    '''
    f(*args_nd)

    # check accuracy
    try:
        # get the transfored covered csr from selected_tile
        covered_csr = transform_covered_position_space_to_ori_space(selected_tile).astype(dtype)

        dtype_th = None
        if dtype == 'float16':
            # TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
            dtype_th = torch.float16 # torch.float32 # torch.float16
        elif dtype == 'float32':
            dtype_th = torch.float32

        # A_torch = torch.sparse_csr_tensor(torch.tensor(covered_csr.indptr, dtype=torch.int32),
        #                                 torch.tensor(covered_csr.indices, dtype=torch.int32),
        #                                 torch.tensor(covered_csr.data, dtype=dtype_th), 
        #                                 size=(ori_op.idx_lens[0], ori_op.idx_lens[2]),
        #                                 dtype=dtype_th, device=dev_th)

        
        B_data = np.array(ori_op.inps[1].data).astype(dtype)
        # y_golden = torch.sparse.mm(A_torch, torch.tensor(B_data, dtype=dtype_th, device=dev_th)).cpu().numpy()

        y_golden = covered_csr.dot(B_data)
        
        out_nd = args_nd[1]
        tvm.testing.assert_allclose(
                out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]], #[:,idx_values[1]], 
                y_golden, 
                rtol=1e-2, atol=1e-2
            )
    except Exception as e:
        print("--------accuracy error")
        print(e)
        # print(list(out_nd.numpy().reshape(-1, op.idx_lens[1])[idx_values[0],:][:,idx_values[1]]))
        # print(list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]]))
        # print(list(y_golden.cpu().numpy()))
        my_res = list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]])
        th_res = list(y_golden)
        # print(my_res)
        # print(th_res)
        for row_i, (i,j) in enumerate(zip(my_res, th_res)):
            # if row_i!=0:
            #     continue
            if list(i) != list(j):
                for col_i, (ii, ji) in enumerate(zip(i,j)):
                    if (abs(1-ji/ii)>0.01): # and (abs(1-ji/ii)>1.01):
                        print(f'{(row_i, col_i)}: {ii}, {ji}, {abs(1-ji/ii)}')

    # print("start measuring")
    # evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    # cost = evaluator(*args_nd).mean
    # print("tc-spmm time: {:.5f}ms".format(cost * 1000))

    cost = profile_tvm_ms(f, args_nd)

    return cost   




def measure_latency_one_tile_more_accurate(selected_tile, ori_op, f, args_nd, max_i, max_j, max_k, dtype, dev, dev_th):
    '''
    NOTE: this function only measures the latency of one selected tile, 
        so the accuracy checking part is different with the above one.

        INPUT:
            ori_op: the original op to be optimized.
            f: the optimized kernel.
            args_nd: the tvm array parameters.
            out_shape: the output shape of the op after scheduling.
            dtype: the data type used for ori_op.
            dev: the tvm device to run. 
            dev_th: the pytorch device. E.g., cuda2 = torch.device('cuda:2')
        OUTPUT:
            the latency of the operator after optimization.
    '''
    if get_template_str(selected_tile.op) == 'TensorCore_template':
        b, c, a, i_indices, indptr, indices = args_nd
        print(b.shape, c.shape, a.shape, i_indices.shape, indptr.shape, indices.shape)
    elif get_template_str(selected_tile.op) == 'sparse_template_ell':
        b, c, a, indices_k, indices_reordered_i = args_nd
        print(b.shape, c.shape, a.shape, indices_k.shape, indices_reordered_i.shape)

    # print(profile_tvm_ms(f, args_nd))
    f(*args_nd)
    # print(profile_tvm_ms(f, args_nd))

    # check accuracy
    try:
        # get the transfored covered csr from selected_tile
        # covered_csr = transform_covered_position_space_to_ori_space(selected_tile).astype(dtype)

        covered_csr, B_data, indices_reordered_i = None, None, None
        if get_template_str(selected_tile.op) == 'sparse_template':
            b, c, a, indptr_k, indices_k, indices_reordered_i = args_nd
            # repeat_row = 108*100 // (selected_tile.op.idx_lens[1] // math.prod(selected_tile.tile_sizes[1][1:]))
            covered_csr = scipy.sparse.csr_matrix(
                (a.numpy().flatten(), indices_k.numpy().flatten(), indptr_k.numpy().flatten()), 
                shape=(len(indices_reordered_i.numpy()), max_k)).astype(dtype)
            B_data = b.numpy().reshape((max_k, max_j)).astype(dtype)

            indices_reordered_i = indices_reordered_i.numpy()

        elif get_template_str(selected_tile.op) == 'sparse_template_ell':
            b, c, a, indices_k, indices_reordered_i = args_nd
            I, K = math.prod(selected_tile.tile_sizes[0][1:]), math.prod(selected_tile.tile_sizes[2][1:])

            vals, idx = np.unique(indices_reordered_i.numpy(), return_index=True)
            vals = vals[np.argsort(idx)]
            _, counts = np.unique(indices_reordered_i.numpy(), return_counts=True)
            counts = counts[np.argsort(idx)]
            indptr = np.concatenate( ([0], np.cumsum(counts)) ) * K

            indices_reordered_i = vals

            covered_csr = scipy.sparse.csr_matrix(
                (a.numpy().flatten(), indices_k.numpy().flatten(), 
                    indptr), 
                shape=(len(indices_reordered_i), max_k)).astype(dtype)
            B_data = b.numpy().reshape((max_k, max_j)).astype(dtype)
        elif get_template_str(selected_tile.op) == 'TensorCore_template':
            b, c, a, i_indices, indptr, indices = args_nd
            print(a.shape, i_indices.shape, indptr.shape, indices.shape)
            I, K = math.prod(selected_tile.tile_sizes[0][1:]), math.prod(selected_tile.tile_sizes[2][1:])
            covered_csr = scipy.sparse.csr_matrix(
                (a.numpy().flatten(), 
                    np.tile(indices.numpy().reshape((-1, K)), I).flatten(), 
                    [K * i for i in range(a.numpy().shape[0]*I+1) ]), 
                shape=(a.numpy().shape[0]*I, max_k)).astype(dtype)
            B_data = b.numpy().reshape((max_k, max_j)).astype(dtype)
            indices_reordered_i = i_indices.numpy()
            if len(i_indices.numpy().shape) == 1:
                # i_indices 只是每个block的pos i
                I = math.prod(selected_tile.tile_sizes[0][1:])
                indices_reordered_i = np.concatenate([np.arange(i, i + I) for i in i_indices.numpy()])


        # indices_reordered_i = indices_reordered_i.numpy()
        indices_reordered_i = indices_reordered_i.flatten()

        dtype_th = None
        if dtype == 'float16':
            # TODO: 看看是换别的函数来计算，还是更新pytorch版本，因为目前的版本不支持A100，也不支持half精度的sparse.mm函数
            dtype_th = torch.float16 # torch.float32 # torch.float16
        elif dtype == 'float32':
            dtype_th = torch.float32

        # A_torch = torch.sparse_csr_tensor(torch.tensor(covered_csr.indptr, dtype=torch.int32),
        #                                 torch.tensor(covered_csr.indices, dtype=torch.int32),
        #                                 torch.tensor(covered_csr.data, dtype=dtype_th), 
        #                                 size=(ori_op.idx_lens[0], ori_op.idx_lens[2]),
        #                                 dtype=dtype_th, device=dev_th)

        
        y_golden = covered_csr.dot(B_data)
        


        # B_data = np.array(ori_op.inps[1].data).astype(dtype)
        # # y_golden = torch.sparse.mm(A_torch, torch.tensor(B_data, dtype=dtype_th, device=dev_th)).cpu().numpy()

        # y_golden = covered_csr.dot(B_data)
        
        out_nd = args_nd[1]
        # print(out_nd.numpy(), out_nd.numpy().shape)
        # print(indices_reordered_i.shape)
        # print(out_nd.numpy().reshape(max_i, max_j).shape, out_nd.numpy().reshape(max_i, max_j)[indices_reordered_i ,:].shape)
        tvm.testing.assert_allclose(
                out_nd.numpy().reshape(max_i, max_j)[indices_reordered_i ,:], # [:,:ori_op.idx_lens[1]], #[:,idx_values[1]], 
                y_golden, 
                rtol=1e-2, atol=1e-2
            )
    except Exception as e:
        print("--------accuracy error")
        print(e)
        # # print(list(out_nd.numpy().reshape(-1, op.idx_lens[1])[idx_values[0],:][:,idx_values[1]]))
        # # print(list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]]))
        # # print(list(y_golden.cpu().numpy()))
        # # my_res = list(out_nd.numpy().reshape(out_shape)[:ori_op.idx_lens[0] ,:][:,:ori_op.idx_lens[1]])
        # my_res = list(out_nd.numpy().reshape(max_i, max_j)[indices_reordered_i ,:])
        # th_res = list(y_golden)
        # # print(my_res)
        # # print(th_res)
        # for row_i, (i,j) in enumerate(zip(my_res, th_res)):
        #     # if row_i!=0:
        #     #     continue
        #     if list(i) != list(j):
        #         for col_i, (ii, ji) in enumerate(zip(i,j)):
        #             if (abs(1-ji/ii)>0.01): # and (abs(1-ji/ii)>1.01):
        #                 print(f'{(row_i, col_i)}: {ii}, {ji}, {abs(1-ji/ii)}')

    # print("start measuring")
    # evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    # cost = evaluator(*args_nd).mean
    # print("tc-spmm time: {:.5f}ms".format(cost * 1000))

    cost = profile_tvm_ms(f, args_nd)

    return cost   





def store_op_def_to_file(filename, op_def_str):
    with open(filename, 'w') as f:
        f.write("from tvm.script import tir as T\n")
        f.write(op_def_str)



# 从code中分离出变量声明 和 计算 这两部分的代码，并且直接完成thread y到fuse之后的thread x的替换，这样在nvcc里面就可以直接用了。
def split_good_1D_CUDA(ori_code, params):
    '''
    params: is the parameters used for 1D tiles
    '''
    # print(ori_code)
    ori_code = ori_code.split('\n')

    var_start, var_end = None, None
    for i, line in enumerate(ori_code):
        if 'main_kernel' in line:
            var_start = i+1
        elif ('=' in line) and (var_start != None):
            var_end = i
            break

    assert (var_start != None) and (var_end != None) and (var_start<var_end), "ERROR: did not find the target kernel."

    variables = ori_code[var_start:var_end]

    # we need to get the string to replace "threadIdx.y" and "threadIdx.x".
    assert int(math.log(params['tx'], 2)) == math.log(params['tx'], 2), "tx is not 2**x"
    # *(uint2*)(A_1D + ((mid0[((((int)blockIdx.x) * 32) + ((((int)threadIdx.x) >> 3) * 8))] * 32) + ((((int)threadIdx.x) & 7) * 4)));
    # (((int)threadIdx.x) >> 3)
    ori_ty = '((int)threadIdx.y)'
    new_ty = f"(((int)threadIdx.x) >> {int(math.log(params['tx'], 2))})"
    # (((int)threadIdx.x) & 7)
    ori_tx = '((int)threadIdx.x)'
    new_tx = f"(((int)threadIdx.x) & {int(params['tx'])-1})"

    computation = list()
    for line in ori_code[var_end:]:
        new_line = line.replace(ori_tx, new_tx).replace(ori_ty, new_ty)
        if len(new_line) == 0:
            # 我们不存空行
            continue
        computation.append(new_line)
        # 不管下面的这个写法了，我们还是在生成good CUDA for 1D tiles的时候就加上vectorize。
        # if ('C0' in line) and ('C0_local' in line):
        #     # 这一行是在把结果从local memory中写回 global memory里
        #     if 'for' in computation[-2] and (params['tx'] == params['group_size']):
        #         # 每个线程有多个结果要写回，而且此处没有采用vectorize
        #         computation = computation[:-2]
        #         new_line = f"C0[((((int)blockIdx.x) * {int(params['max_bucket_size'])}) + ((int)threadIdx.x))] = C0_local[(((int)threadIdx.x) & 7)];"

    assert computation[-1] == '}', f'Check the last non empty line again: {computation[-6:]}EndSymbol'
    computation = computation[:-1]


    # print('\n'.join(variables))
    # print('\n'.join(computation))

    return '\n'.join(variables), '\n'.join(computation)






# 用这个函数来获得功能完好的thread y和thread x还没有被fuse在一起的，sparsetir会生成的1D tiles的CUDA代码。
def prepare_good_1D_CUDA(ori_op, op_type, batch_num, selected_tiles, cuda_i, cache_set, dsize,
    dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
    '''
    NOTE: the input selected_tiles are only about 1D tiles.
    '''
    assert ori_op.op_type == 'sddmm', "This is only for SDDMM operators."

    formats = get_formats_from_selected_tiles(selected_tiles, cache_set, dsize)

    # dtype = "float16"
    # dtype_str = '"float16"'
    # zerotype = "T.float16(0)"
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(cuda_i)
    dev_th = torch.device(f'cuda:{cuda_i}') # None # torch.device(f'cuda:{cuda_i}')

    op_def_str = gen_op_definition(formats, dtype_str, zerotype, cache_set, dsize, op_type, batch_num)
    # print(op_def_str)
    store_op_def_to_file(f"tmp_op_def{os.environ['MyFileID']}.py", op_def_str)
    # exec(op_def_str)
    func = None
    if os.environ['MyFileID'] == '0':
        import tmp_op_def0
        tmp_op_def0 = reload(tmp_op_def0)
        func = tmp_op_def0.my_fusedFormats
    elif os.environ['MyFileID'] == '1':
        import tmp_op_def1
        tmp_op_def1 = reload(tmp_op_def1)
        func = tmp_op_def1.my_fusedFormats
    elif os.environ['MyFileID'] == '2':
        import tmp_op_def2
        tmp_op_def2 = reload(tmp_op_def2)
        func = tmp_op_def2.my_fusedFormats
    elif os.environ['MyFileID'] == '3':
        import tmp_op_def3
        tmp_op_def3 = reload(tmp_op_def3)
        func = tmp_op_def3.my_fusedFormats      

    # 以下是原代码---
    # import tmp_op_def
    # tmp_op_def = reload(tmp_op_def)

    # func = tmp_op_def.my_fusedFormats

    sch, out_shape = set_params(func, formats, dsize, op_type)

    # print(sch.mod.script())

    f = schedule_fused_kernel_and_build(op_type, sch, formats, target)
    variables, computation = split_good_1D_CUDA(f.imported_modules[0].get_source(), selected_tiles[0].params)
    with open(f"Good_1D_CUDA{os.environ['MyFileID']}.cuda", 'w') as file:
        file.write(variables)
        file.write("\nNEXT IS COMPUTATION CODE\n")
        file.write(computation)

    # 这个变量会在nvcc里面用到
    os.environ['has_1d_tile'] = 'True'

    # print(f.imported_modules[0].get_source())
    # with open("Check_accuracy.py", 'w') as file:
    #     file.write(f.imported_modules[0].get_source())










# 接下来是整体的函数
def measure_seleted_formats(ori_op, op_type, batch_num, selected_tiles, cuda_i, cache_set, dsize, gened_inputs = list(),
    dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
    '''
    input里面有op_type, batch_num, 其中ori_op.op_type 可能和op_type不一样, 比如我们在做batched spmm的实验的时候, 
    ori_op.op_type=spmm, op_type=batched spmm。这是因为我们目前的tuning阶段只支持spmm, 暂时先这样实现batched spmm的功能支持。
    '''
    # 这个变量会在nvcc里面用到，会在prepare_good_1D_CUDA 获得了good cuda之后变为True
    os.environ['has_1d_tile'] = 'False'
    # os.environ['has_32thread_SDDMM_cuda'] = 'False'

    tiles_1d = [t for t in selected_tiles if get_template_str(t.op) == '1D_sddmm']
    ori_param_1D = None
    if len(tiles_1d)>0:
        ori_param_1D = tiles_1d[0].params
    fake_param_1D = {'tx': 8, 'ty': 4, 'vec_size': 4, 'group_size': 8, 'max_bucket_size': 32}
    if (len(tiles_1d) > 0) and (len(tiles_1d) < len(selected_tiles)):
        prepare_good_1D_CUDA(ori_op, op_type, batch_num, tiles_1d, cuda_i, cache_set, dsize,
            dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
        # prepare_good_SDDMM_CUDA_both_TC_1D(ori_op, op_type, batch_num, selected_tiles, cuda_i, cache_set, dsize,
        #     dtype = dtype, dtype_str = dtype_str, zerotype = zerotype)
        # for t in tiles_1d:
        #     t.params = fake_param_1D
        #     t.best_params = fake_param_1D


    formats = get_formats_from_selected_tiles(selected_tiles, cache_set, dsize)

    # dtype = "float16"
    # dtype_str = '"float16"'
    # zerotype = "T.float16(0)"
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(cuda_i)
    dev_th = torch.device(f'cuda:{cuda_i}') # None # torch.device(f'cuda:{cuda_i}')

    op_def_str = gen_op_definition(formats, dtype_str, zerotype, cache_set, dsize, op_type, batch_num)
    # print(op_def_str)
    store_op_def_to_file(f"tmp_op_def{os.environ['MyFileID']}.py", op_def_str)
    # exec(op_def_str)
    func = None
    if os.environ['MyFileID'] == '0':
        import tmp_op_def0
        tmp_op_def0 = reload(tmp_op_def0)
        func = tmp_op_def0.my_fusedFormats
    elif os.environ['MyFileID'] == '1':
        import tmp_op_def1
        tmp_op_def1 = reload(tmp_op_def1)
        func = tmp_op_def1.my_fusedFormats
    elif os.environ['MyFileID'] == '2':
        import tmp_op_def2
        tmp_op_def2 = reload(tmp_op_def2)
        func = tmp_op_def2.my_fusedFormats
    elif os.environ['MyFileID'] == '3':
        import tmp_op_def3
        tmp_op_def3 = reload(tmp_op_def3)
        func = tmp_op_def3.my_fusedFormats      

    # 以下是原代码---
    # import tmp_op_def
    # tmp_op_def = reload(tmp_op_def)

    # func = tmp_op_def.my_fusedFormats

    sch, out_shape = set_params(func, formats, dsize, op_type)

    # print(sch.mod.script())

    f = schedule_fused_kernel_and_build(op_type, sch, formats, target)
    # print(f.imported_modules[0].get_source())
    # with open("Check_accuracy.py", 'w') as file:
    #     file.write(f.imported_modules[0].get_source())

    # FOR DEBUG 暂时删掉了
    args_nd, c_indices = None, None
    print("len(gened_inputs)", len(gened_inputs))
    if len(gened_inputs) == 0:
        args_nd, c_indices = prepare_inputs(ori_op, op_type, batch_num, formats, out_shape, dtype, dev)
        gened_inputs.append(args_nd)
        gened_inputs.append(c_indices)
        print("FINISH PREPARE INPUTS")
    else:
        prepare_inputs_based_on_gened_inputs(gened_inputs, tiles_1d, op_type, dtype, dev)
        args_nd, c_indices = gened_inputs

    return ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th, cuda_i, c_indices

    cost = measure_latency(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th, cuda_i, c_indices)

    # if len(tiles_1d) > 0:
    #     for t in tiles_1d:
    #         t.params = ori_param_1D
    #         t.best_params = ori_param_1D

    return cost





def Benchmark_TC_BSDDMM(ori_op, op_type, batch_num, selected_tiles, cuda_i, cache_set, dsize, row_window_num, row_window_width, 
    dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
    '''
    input里面有op_type, batch_num, 其中ori_op.op_type 可能和op_type不一样, 比如我们在做batched spmm的实验的时候, 
    ori_op.op_type=spmm, op_type=batched spmm。这是因为我们目前的tuning阶段只支持spmm, 暂时先这样实现batched spmm的功能支持。
    '''

    # !!NOTE: there will be only one tile in selected_tiles
    os.environ['has_1d_tile'] = 'False'

    formats = get_formats_from_selected_tiles(selected_tiles, cache_set, dsize)
    fmt = list(formats.keys())[0]
    formats[fmt] = [selected_tiles[0] for i in range(row_window_num*row_window_width)]

    target = tvm.target.Target("cuda")
    dev = tvm.cuda(cuda_i)
    dev_th = torch.device(f'cuda:{cuda_i}') # None # torch.device(f'cuda:{cuda_i}')

    op_def_str = gen_op_definition(formats, dtype_str, zerotype, cache_set, dsize, op_type, batch_num)
    # print(op_def_str)
    store_op_def_to_file(f"tmp_op_def{os.environ['MyFileID']}.py", op_def_str)
    # exec(op_def_str)
    func = None
    if os.environ['MyFileID'] == '0':
        import tmp_op_def0
        tmp_op_def0 = reload(tmp_op_def0)
        func = tmp_op_def0.my_fusedFormats
    elif os.environ['MyFileID'] == '1':
        import tmp_op_def1
        tmp_op_def1 = reload(tmp_op_def1)
        func = tmp_op_def1.my_fusedFormats
    elif os.environ['MyFileID'] == '2':
        import tmp_op_def2
        tmp_op_def2 = reload(tmp_op_def2)
        func = tmp_op_def2.my_fusedFormats
    elif os.environ['MyFileID'] == '3':
        import tmp_op_def3
        tmp_op_def3 = reload(tmp_op_def3)
        func = tmp_op_def3.my_fusedFormats      

    # 以下是原代码---
    # import tmp_op_def
    # tmp_op_def = reload(tmp_op_def)

    # func = tmp_op_def.my_fusedFormats

    sch, out_shape = set_params(func, formats, dsize, op_type)

    # print(sch.mod.script())

    f = schedule_fused_kernel_and_build(op_type, sch, formats, target)
    # print(f.imported_modules[0].get_source())
    # with open("Check_accuracy.py", 'w') as file:
    #     file.write(f.imported_modules[0].get_source())
    print(f.imported_modules[0].get_source())

    # FOR DEBUG 暂时删掉了
    args_nd, c_indices = prepare_inputs_BSDDMM_one_tile(ori_op, op_type, batch_num, formats, out_shape, dtype, dev, row_window_width)
    cost = measure_latency(ori_op, op_type, batch_num, f, args_nd, out_shape, dtype, dev, dev_th, cuda_i, c_indices)

    return cost







# 对于单个tile的measure，我们目前只支持spmm这个算子，所以对于batch spmm，我们也会当做spmm来对待。
def measure_seleted_tile(ori_op, selected_tile, cuda_i, cache_set, dsize,
    dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
    selected_tiles = [selected_tile]
    formats = get_formats_from_selected_tiles(selected_tiles, cache_set, dsize)

    # dtype = "float16"
    # dtype_str = '"float16"'
    # zerotype = "T.float16(0)"
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(cuda_i)
    dev_th = torch.device(f'cuda:{cuda_i}') # None # torch.device(f'cuda:{cuda_i}')

    # op_def_str = gen_op_definition(formats, dtype_str, zerotype, cache_set, dsize)
    op_def_str = gen_op_definition(formats, dtype_str, zerotype, cache_set, dsize, ori_op.op_type, 1)
    # print(op_def_str)
    store_op_def_to_file(f"tmp_op_def{os.environ['MyFileID']}.py", op_def_str)
    # exec(op_def_str)
    func = None
    if os.environ['MyFileID'] == '0':
        import tmp_op_def0
        tmp_op_def0 = reload(tmp_op_def0)
        func = tmp_op_def0.my_fusedFormats
    elif os.environ['MyFileID'] == '1':
        import tmp_op_def1
        tmp_op_def1 = reload(tmp_op_def1)
        func = tmp_op_def1.my_fusedFormats
    elif os.environ['MyFileID'] == '2':
        import tmp_op_def2
        tmp_op_def2 = reload(tmp_op_def2)
        func = tmp_op_def2.my_fusedFormats
    elif os.environ['MyFileID'] == '3':
        import tmp_op_def3
        tmp_op_def3 = reload(tmp_op_def3)
        func = tmp_op_def3.my_fusedFormats   


    # 以下是原代码---
    # import tmp_op_def
    # tmp_op_def = reload(tmp_op_def)

    # # print("before load")
    # func = tmp_op_def.my_fusedFormats

    # print("before set params")

    sch, out_shape = set_params(func, formats, dsize, ori_op.op_type)

    # f = schedule_fused_kernel_and_build(sch, formats, target)
    f = schedule_fused_kernel_and_build(ori_op.op_type, sch, formats, target)
    # print(f.imported_modules[0].get_source())
    # with open("Check_accuracy.py", 'w') as file:
    #     file.write(f.imported_modules[0].get_source())

    # args_nd = prepare_inputs(ori_op, formats, out_shape, dtype, dev)
    args_nd, _ = prepare_inputs(ori_op, ori_op.op_type, 1, formats, out_shape, dtype, dev)
    cost = measure_latency_one_tile(selected_tile, ori_op, f, args_nd, out_shape, dtype, dev, dev_th)

    return cost





def measure_seleted_tile_more_accurate(ori_op, selected_tile, cuda_i, cache_set, dsize, set_atomic,
    dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
    selected_tiles = [selected_tile]
    formats = get_formats_from_selected_tiles(selected_tiles, cache_set, dsize, set_atomic = set_atomic)

    target = tvm.target.Target("cuda")
    dev = tvm.cuda(cuda_i)
    dev_th = torch.device(f'cuda:{cuda_i}') # None # torch.device(f'cuda:{cuda_i}')

    op_def_str = gen_op_definition(formats, dtype_str, zerotype, cache_set, dsize, ori_op.op_type, 1)
    # print(op_def_str)
    store_op_def_to_file(f"tmp_op_def{os.environ['MyFileID']}.py", op_def_str)
    # exec(op_def_str)
    func = None
    if os.environ['MyFileID'] == '0':
        import tmp_op_def0
        tmp_op_def0 = reload(tmp_op_def0)
        func = tmp_op_def0.my_fusedFormats
    elif os.environ['MyFileID'] == '1':
        import tmp_op_def1
        tmp_op_def1 = reload(tmp_op_def1)
        func = tmp_op_def1.my_fusedFormats
    elif os.environ['MyFileID'] == '2':
        import tmp_op_def2
        tmp_op_def2 = reload(tmp_op_def2)
        func = tmp_op_def2.my_fusedFormats
    elif os.environ['MyFileID'] == '3':
        import tmp_op_def3
        tmp_op_def3 = reload(tmp_op_def3)
        func = tmp_op_def3.my_fusedFormats   


    args_nd, (max_i, max_j, max_k) = prepare_inputs_measure_one_tile(ori_op, formats, dtype, dev)

    sch, _ = set_params_measure_one_tile(func, formats, dsize, max_i, max_j, max_k)
    # out_shape = (max_i, max_j)

    # f = schedule_fused_kernel_and_build(sch, formats, target)
    f = schedule_fused_kernel_and_build(ori_op.op_type, sch, formats, target)
    # print(f.imported_modules[0].get_source())
    # with open("Check_accuracy.py", 'w') as file:
    #     file.write(f.imported_modules[0].get_source())

    # cost = measure_latency_one_tile(selected_tile, ori_op, f, args_nd, out_shape, dtype, dev, dev_th)
    cost = measure_latency_one_tile_more_accurate(selected_tile, ori_op, f, args_nd, max_i, max_j, max_k, dtype, dev, dev_th)

    return cost


# ====================================================================
# about tuning tiles faster
# 虽然根据我们的猜想，一个很快速的tuning方案（对于32 bit的数据来说），固定j轴thread 数量为32，然后变化i轴thread数量使其为128倍数。
# 但是此处还是决定使用cost based的方法。
# 找到更有希望成为好的implementation的具体的tile_sizes
def fast_tile_tuner_BSPMM(tile, dsize, max_bucket_size):
    sub_op = tile.op
    tile_sizes = tile.tile_sizes
    params = tile.params

    all_tile_sizes = [tile_sizes]
    is_tb_tile = False
    if max([len(tile_size) for tile_size in tile_sizes]) == 2:
        all_tile_sizes = gen_tile_sizes_given_tile_tb(tile, max_bucket_size)
        is_tb_tile = True

    mem_cost_dict = dict()
    for tile_sizes in all_tile_sizes:
        thread_num = tile_sizes[0][2]*tile_sizes[1][2]
        if thread_num not in mem_cost_dict:
            mem_cost_dict[thread_num] = list()
        mem_cost = my_cost_model.memory_cost_tb_impl(tile, tile_sizes, dsize)
        mem_cost_dict[thread_num].append((tile_sizes, mem_cost))

    ret = list()
    for thread_num in sorted(mem_cost_dict.keys(), reverse=True):
        costs = mem_cost_dict[thread_num]
        costs = sorted(costs, key=lambda tmp:tmp[1])
        ret.append(costs[0][0])

    return ret



def fast_tile_tuner_BSDDMM(tile, dsize, max_bucket_size):
    return []



def fast_tile_tuner(tile, dsize, max_bucket_size):
    # 这个函数其实没啥用了，所以我们不再维护这个函数，再搜索tile的过程中也不再单独measure每个tile。
    if tile.op.op_type == 'spmm':
        return fast_tile_tuner_BSPMM(tile, dsize, max_bucket_size)
    elif tile.op.op_type == 'sddmm':
        return fast_tile_tuner_BSDDMM(tile, dsize, max_bucket_size)

# ====================================================================



def measure_tiles(ori_op, tiles, tuning_hub, dsize, cache_set,
    max_bucket_size, use_faster_tuner = False, 
    log_file="log_hub/log.py", cuda_i = 3, 
    dtype = "float16", dtype_str = '"float16"', zerotype = "T.float16(0)"):
    '''
    NOTE: the input tiles do not have concrete tile sizes and parameters. 
    This function is to find the best tile sizes and the best params for the given tiles.
    '''
    # if ori_op.op_type == 'sddmm':
    # <jingzhi>@revision: also do not tune ELL tiles: tile.best_tile_sizes = tile.tile_sizes is wrong (wrong tile sizes) for spmm. Anyway, delete it for no improvement on logsparse spmm.
    if ori_op.op_type in ('sddmm', 'spmm'):
        for tile in tiles:
            tile.cost = 1
            tile.best_tile_sizes = tile.tile_sizes
            if ori_op.op_type == 'spmm':
                tile.best_tile_sizes = ([None, tile.tile_sizes[0][1], 1], [None, tile.tile_sizes[1][1]//32, 32], (1, tile.tile_sizes[2][1]))
            tile.best_params = tile.params
            tile.set_avg_cost()
        return


    target = tvm.target.Target("cuda")

    tensor_core_cost_dict = dict()

    tot_kernels_to_measure = 0
    for tile in tiles:
        if tile.cost != None:
            # already measured
            print("already measured")
            continue

        sub_op = tile.op
        sub_op_id = sub_op.op_id
        tile_sizes = tile.tile_sizes
        tile_pos = tile.tile_pos
        params = tile.params
        area_i = sub_op.this_area_i


        all_tile_sizes = None
        is_tb_tile = None
        if not use_faster_tuner:
            all_tile_sizes = [tile_sizes]
            is_tb_tile = False
            if max([len(tile_size) for tile_size in tile_sizes]) == 2:
                all_tile_sizes = gen_tile_sizes_given_tile_tb(tile, max_bucket_size)
                is_tb_tile = True
        else:
            all_tile_sizes = fast_tile_tuner(tile, dsize, max_bucket_size)
            is_tb_tile = True

        best_cost = float('inf')
        best_tile_sizes = None
        best_params = None

        # Preprocess tile. Pretend that it has been selected.
        gen_updated_position_space_for_tile(tile)
        tile.position_space_when_selected = tile.uncovered_position_space
        tile.nnz_when_selected = tile.nnz_uncovered


        # 判断一下这个tile是不是有同样shape 同样sub_op的其他tile已经被measure过了
        can_reuse = False
        reuse_key = json.dumps((sub_op_id, tile_sizes))
        if reuse_key in tuning_hub:
            all_tile_sizes = [tuning_hub[reuse_key][0]]


        for tile_sizes in all_tile_sizes:
            all_params = [params]
            if is_tb_tile:
                all_params = get_params_list(sub_op, tile_sizes, max_bucket_size, is_tb_tile = False)
            
            if reuse_key in tuning_hub:
                all_params = [tuning_hub[reuse_key][1]]

            for params in all_params:
                try:
                    # pass
                    if tot_kernels_to_measure % 1 == 0:
                        print(sub_op_id, sub_op.loop_protocals[area_i], sub_op.idx_reordered[area_i])
                        print(f"{tile_sizes, tile_pos, params}")


                    # 在测量cost之前还得预处理一下，比如把假装已经选择了这个tile且得到了其best tile sizes 和 best params
                    if (sub_op.loop_protocals[area_i] == 'uuu') and (params['mma_shape_str'] in tensor_core_cost_dict):
                        cost = tensor_core_cost_dict[params['mma_shape_str']]
                    else:
                        # need to measure
                        tile.best_tile_sizes = tile_sizes
                        tile.best_params = params
                        cost = 100
                        if reuse_key not in tuning_hub:
                            cost = measure_seleted_tile(ori_op, tile, cuda_i, cache_set, dsize, dtype, dtype_str, zerotype)
                        if sub_op.loop_protocals[area_i] == 'uuu':
                            tensor_core_cost_dict[params['mma_shape_str']] = cost

                    print(f"cost: {cost}   {tile_sizes, tile_pos, params}")
                    if (cost != -1) and (cost < best_cost):
                        best_cost = cost
                        best_tile_sizes = tile_sizes
                        best_params = params

                    if cost != -1:
                        with open(log_file, "a") as f:
                            f.write(f"cost_dict[{sub_op_id}, {tuple(tile_sizes)}, {tuple(tile_pos)}, {json.dumps(params)}] = {cost}\n")
                    # =============================================

                    # if cost == -1:
                    #   tile.cost = float('inf')
                    # else:
                    # # cost_dict[sub_op_id, tuple(tile_sizes), tuple(tile_pos), json.dumps(params)] = cost
                    # # if cost != -1:
                    #   tile.cost = cost
                    #   with open(log_file, "a") as f:
                    #       f.write(f"cost_dict[{sub_op_id}, {tuple(tile_sizes)}, {tuple(tile_pos)}, {json.dumps(params)}] = {cost}\n")
                    
                    # tile.set_avg_cost()
                    # 
                except Exception as e:
                    print(e)
                    print("error")
                    # return tile
                tot_kernels_to_measure += 1

        tile.cost = best_cost
        tile.best_tile_sizes = best_tile_sizes
        tile.best_params = best_params
        tile.set_avg_cost()
        if tile.cost == float('inf'):
            tile.pred_cost = float('inf')
            tile.set_pred_avg_cost()
        else:
            # 当找到valid implementation时，才把结果存到tuning hub中
            if reuse_key not in tuning_hub:
                tuning_hub[reuse_key] = [best_tile_sizes, best_params]

    print(f"tot_kernels_measured: {tot_kernels_to_measure}")






