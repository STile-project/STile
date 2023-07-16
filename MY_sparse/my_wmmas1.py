import tvm
import tvm.tir as tir
from tvm.script import tir as T


@T.prim_func
def wmma_m16n16k16_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, k, j in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vkk, vjj = T.axis.remap("SRS", [i, k, j])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]


@T.prim_func
def wmma_m16n16k16_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )






@T.prim_func
def wmma_m16n16k16_sync_ijk_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vjj, vkk]


@T.prim_func
def wmma_m16n16k16_sync_ijk_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )








@T.prim_func
def wmma_m8n32k16_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, k, j in T.grid(8, 16, 32):
            with T.block("update"):
                vii, vkk, vjj = T.axis.remap("SRS", [i, k, j])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]


@T.prim_func
def wmma_m8n32k16_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:8, 0:32],
                A_frag[0:8, 0:16],
                B_frag[0:16, 0:32],
            ]
        )
        T.writes(C_frag[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    A_frag.data,
                    A_frag.elem_offset // 128 + T.floordiv(T.floormod(A_frag.elem_offset, 128), 16),
                    B_frag.data,
                    B_frag.elem_offset // 512 + T.floordiv(T.floormod(B_frag.elem_offset, 512), 32),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="global")
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
def wmma_m16n16k16_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
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
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )





@T.prim_func
def wmma_m16n16k16_load_a_shared_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
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
def wmma_m16n16k16_load_a_shared_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
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
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )





@T.prim_func
def wmma_m8n32k16_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (8, 16), "float16", align=128, offset_factor=16, scope="global")
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:8, 0:16])
        T.writes(A_frag[0:8, 0:16])
        for i, j in T.grid(8, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_m8n32k16_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (8, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:8, 0:16])
        T.writes(A_frag[0:8, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    A_frag.data,
                    8,
                    32,
                    16,
                    A_frag.elem_offset // 128 + T.floordiv(T.floormod(A_frag.elem_offset, 128), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
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
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m16n16k16_load_b_shared_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_b_shared_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
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
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )






@T.prim_func
def wmma_m16n16k16_load_b_shared_colmajor_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_b_shared_colmajor_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
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
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "col_major",
                    dtype="handle",
                )
            )






@T.prim_func
def wmma_m8n32k16_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 32), "float16", align=128, offset_factor=16, scope="shared")
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 32):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m8n32k16_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 32), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        T.reads(B[0:16, 0:32])
        T.writes(B_frag[0:16, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B_frag.data,
                    8,
                    32,
                    16,
                    B_frag.elem_offset // 512 + T.floordiv(T.floormod(B_frag.elem_offset, 512), 32),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_m16n16k16_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m8n32k16_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(8, 32):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_m8n32k16_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    8,
                    32,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m16n16k16_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m16n16k16_store_shared_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m16n16k16_store_shared_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m8n32k16_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (8, 32), "float16", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(8, 32):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m8n32k16_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (8, 32), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:8, 0:32])
        T.writes(C[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    8,
                    32,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m8n32k16_store_shared_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (8, 32), "float16", align=128, offset_factor=16, scope="shared")
    with T.block("root"):
        for i, j in T.grid(8, 32):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m8n32k16_store_shared_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (8, 32), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:8, 0:32])
        T.writes(C[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    8,
                    32,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m16n16k16_sync_desc_fp32(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, k, j in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vkk, vjj = T.axis.remap("SRS", [i, k, j])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]


@T.prim_func
def wmma_m16n16k16_sync_impl_fp32(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m8n32k16_sync_desc_fp32(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float32", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float32", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, k, j in T.grid(8, 16, 32):
            with T.block("update"):
                vii, vkk, vjj = T.axis.remap("SRS", [i, k, j])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]


@T.prim_func
def wmma_m8n32k16_sync_impl_fp32(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float32", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:8, 0:32],
                A_frag[0:8, 0:16],
                B_frag[0:16, 0:32],
            ]
        )
        T.writes(C_frag[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    A_frag.data,
                    A_frag.elem_offset // 128 + T.floordiv(T.floormod(A_frag.elem_offset, 128), 16),
                    B_frag.data,
                    B_frag.elem_offset // 512 + T.floordiv(T.floormod(B_frag.elem_offset, 512), 32),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_load_a_desc_fp32(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_a_impl_fp32(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
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
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )





@T.prim_func
def wmma_m16n16k16_load_a_shared_desc_fp32(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="shared")
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_a_shared_impl_fp32(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float32", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
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
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )





@T.prim_func
def wmma_m8n32k16_load_a_desc_fp32(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (8, 16), "float32", align=128, offset_factor=16, scope="global")
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:8, 0:16])
        T.writes(A_frag[0:8, 0:16])
        for i, j in T.grid(8, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_m8n32k16_load_a_impl_fp32(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (8, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    A_frag = T.match_buffer(
        a_frag, (8, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:8, 0:16])
        T.writes(A_frag[0:8, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    A_frag.data,
                    8,
                    32,
                    16,
                    A_frag.elem_offset // 128 + T.floordiv(T.floormod(A_frag.elem_offset, 128), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_load_b_desc_fp32(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float32", align=128, offset_factor=16, scope="shared")
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_b_impl_fp32(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float32", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.matrix_b"
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
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m8n32k16_load_b_desc_fp32(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 32), "float32", align=128, offset_factor=16, scope="shared")
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float32", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 32):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m8n32k16_load_b_impl_fp32(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 32), "float32", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    B_frag = T.match_buffer(
        b_frag, (16, 32), "float32", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        T.reads(B[0:16, 0:32])
        T.writes(B_frag[0:16, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B_frag.data,
                    8,
                    32,
                    16,
                    B_frag.elem_offset // 512 + T.floordiv(T.floormod(B_frag.elem_offset, 512), 32),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_fill_desc_fp32(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float32(0)


@T.prim_func
def wmma_m16n16k16_fill_impl_fp32(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float32(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m8n32k16_fill_desc_fp32(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(8, 32):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float32(0)


@T.prim_func
def wmma_m8n32k16_fill_impl_fp32(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    8,
                    32,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    T.float32(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_m16n16k16_store_desc_fp32(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m16n16k16_store_impl_fp32(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s0, s1]
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m16n16k16_store_shared_desc_fp32(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="shared")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m16n16k16_store_shared_impl_fp32(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
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
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m8n32k16_store_desc_fp32(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (8, 32), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(8, 32):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m8n32k16_store_impl_fp32(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (8, 32), "float32", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:8, 0:32])
        T.writes(C[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    8,
                    32,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )




@T.prim_func
def wmma_m8n32k16_store_shared_desc_fp32(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (8, 32), "float32", align=128, offset_factor=16, scope="shared")
    with T.block("root"):
        for i, j in T.grid(8, 32):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m8n32k16_store_shared_impl_fp32(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (8, 32), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (8, 32), "float32", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:8, 0:32])
        T.writes(C[0:8, 0:32])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    8,
                    32,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 32),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )







WMMA_M16N16K16_SYNC = tir.TensorIntrin.register(
    "wmma_m16n16k16_sync",
    wmma_m16n16k16_sync_desc,
    wmma_m16n16k16_sync_impl,
)


WMMA_M16N16K16_SYNC_ijk = tir.TensorIntrin.register(
    "wmma_m16n16k16_sync_ijk",
    wmma_m16n16k16_sync_ijk_desc,
    wmma_m16n16k16_sync_ijk_impl,
)



WMMA_M8N32K16_SYNC = tir.TensorIntrin.register(
    "wmma_m8n32k16_sync",
    wmma_m8n32k16_sync_desc,
    wmma_m8n32k16_sync_impl,
)

WMMA_M16N16K16_LOAD_A = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_a",
    wmma_m16n16k16_load_a_desc,
    wmma_m16n16k16_load_a_impl,
)

WMMA_M16N16K16_LOAD_A_shared = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_a_shared",
    wmma_m16n16k16_load_a_shared_desc,
    wmma_m16n16k16_load_a_shared_impl,
)

WMMA_M8N32K16_LOAD_A = tir.TensorIntrin.register(
    "wmma_m8n32k16_load_a",
    wmma_m8n32k16_load_a_desc,
    wmma_m8n32k16_load_a_impl,
)

WMMA_M16N16K16_LOAD_B = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_b",
    wmma_m16n16k16_load_b_desc,
    wmma_m16n16k16_load_b_impl,
)

WMMA_M8N32K16_LOAD_B = tir.TensorIntrin.register(
    "wmma_m8n32k16_load_b",
    wmma_m8n32k16_load_b_desc,
    wmma_m8n32k16_load_b_impl,
)


WMMA_M16N16K16_LOAD_B_shared = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_b_shared",
    wmma_m16n16k16_load_b_shared_desc,
    wmma_m16n16k16_load_b_shared_impl,
)


WMMA_M16N16K16_LOAD_B_shared_colmajor = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_b_shared_colmajor",
    wmma_m16n16k16_load_b_shared_colmajor_desc,
    wmma_m16n16k16_load_b_shared_colmajor_impl,
)

WMMA_M16N16K16_FILL = tir.TensorIntrin.register(
    "wmma_m16n16k16_fill",
    wmma_m16n16k16_fill_desc,
    wmma_m16n16k16_fill_impl,
)

WMMA_M8N32K16_FILL = tir.TensorIntrin.register(
    "wmma_m8n32k16_fill",
    wmma_m8n32k16_fill_desc,
    wmma_m8n32k16_fill_impl,
)

WMMA_M16N16K16_STORE = tir.TensorIntrin.register(
    "wmma_m16n16k16_store",
    wmma_m16n16k16_store_desc,
    wmma_m16n16k16_store_impl,
)


WMMA_M16N16K16_STORE_shared = tir.TensorIntrin.register(
    "wmma_m16n16k16_store_shared",
    wmma_m16n16k16_store_shared_desc,
    wmma_m16n16k16_store_shared_impl,
)

WMMA_M8N32K16_STORE = tir.TensorIntrin.register(
    "wmma_m8n32k16_store",
    wmma_m8n32k16_store_desc,
    wmma_m8n32k16_store_impl,
)

WMMA_M8N32K16_STORE_shared = tir.TensorIntrin.register(
    "wmma_m8n32k16_store_shared",
    wmma_m8n32k16_store_shared_desc,
    wmma_m8n32k16_store_shared_impl,
)




WMMA_M16N16K16_SYNC_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_sync_fp32",
    wmma_m16n16k16_sync_desc_fp32,
    wmma_m16n16k16_sync_impl_fp32,
)

WMMA_M8N32K16_SYNC_FP32 = tir.TensorIntrin.register(
    "wmma_m8n32k16_sync_fp32",
    wmma_m8n32k16_sync_desc_fp32,
    wmma_m8n32k16_sync_impl_fp32,
)

WMMA_M16N16K16_LOAD_A_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_a_fp32",
    wmma_m16n16k16_load_a_desc_fp32,
    wmma_m16n16k16_load_a_impl_fp32,
)

WMMA_M16N16K16_LOAD_A_shared_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_a_shared_fp32",
    wmma_m16n16k16_load_a_shared_desc_fp32,
    wmma_m16n16k16_load_a_shared_impl_fp32,
)

WMMA_M8N32K16_LOAD_A_FP32 = tir.TensorIntrin.register(
    "wmma_m8n32k16_load_a_fp32",
    wmma_m8n32k16_load_a_desc_fp32,
    wmma_m8n32k16_load_a_impl_fp32,
)

WMMA_M16N16K16_LOAD_B_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_b_fp32",
    wmma_m16n16k16_load_b_desc_fp32,
    wmma_m16n16k16_load_b_impl_fp32,
)

WMMA_M8N32K16_LOAD_B_FP32 = tir.TensorIntrin.register(
    "wmma_m8n32k16_load_b_fp32",
    wmma_m8n32k16_load_b_desc_fp32,
    wmma_m8n32k16_load_b_impl_fp32,
)

WMMA_M16N16K16_FILL_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_fill_fp32",
    wmma_m16n16k16_fill_desc_fp32,
    wmma_m16n16k16_fill_impl_fp32,
)

WMMA_M8N32K16_FILL_FP32 = tir.TensorIntrin.register(
    "wmma_m8n32k16_fill_fp32",
    wmma_m8n32k16_fill_desc_fp32,
    wmma_m8n32k16_fill_impl_fp32,
)

WMMA_M16N16K16_STORE_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_store_fp32",
    wmma_m16n16k16_store_desc_fp32,
    wmma_m16n16k16_store_impl_fp32,
)


WMMA_M16N16K16_STORE_shared_FP32 = tir.TensorIntrin.register(
    "wmma_m16n16k16_store_shared_fp32",
    wmma_m16n16k16_store_shared_desc_fp32,
    wmma_m16n16k16_store_shared_impl_fp32,
)

WMMA_M8N32K16_STORE_FP32 = tir.TensorIntrin.register(
    "wmma_m8n32k16_store_fp32",
    wmma_m8n32k16_store_desc_fp32,
    wmma_m8n32k16_store_impl_fp32,
)

WMMA_M8N32K16_STORE_shared_FP32 = tir.TensorIntrin.register(
    "wmma_m8n32k16_store_shared_fp32",
    wmma_m8n32k16_store_shared_desc_fp32,
    wmma_m8n32k16_store_shared_impl_fp32,
)

