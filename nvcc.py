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
# pylint: disable=invalid-name
"""Utility to invoke nvcc compiler in the system"""
from __future__ import absolute_import as _abs

import subprocess
import os
import warnings

import tvm._ffi
from tvm.target import Target

from . import utils
from .._ffi.base import py_str

import json
import math


def compile_cuda(code, target_format="ptx", arch=None, options=None, path_target=None):
    """Compile cuda code with NVCC from env.

    Parameters
    ----------
    code : str
        The cuda code.

    target_format : str
        The target format of nvcc compiler.

    arch : str
        The cuda architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "sm_xx", or a list, such as
        # [
        #   "-gencode", "arch=compute_52,code=sm_52",
        #   "-gencode", "arch=compute_70,code=sm_70"
        # ]
        compute_version = "".join(
            get_target_compute_version(Target.current(allow_none=True)).split(".")
        )
        arch = ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]

    temp = utils.tempdir()
    if target_format not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target_format must be in cubin, ptx, fatbin")
    temp_code = temp.relpath("my_kernel.cu")
    temp_target = temp.relpath("my_kernel.%s" % target_format)

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = ["nvcc"]
    cmd += ["--%s" % target_format, "-O3"]
    if isinstance(arch, list):
        cmd += arch
    elif isinstance(arch, str):
        cmd += ["-arch", arch]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    # NOTE: ccbin option can be used to tell nvcc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require nvcc is configured
    # correctly by default.
    cxx_compiler_path = os.environ.get("CUDAHOSTCXX", "")
    if cxx_compiler_path != "":
        cmd += ["-ccbin", cxx_compiler_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def find_cuda_path():
    """Utility function to find cuda path

    Returns
    -------
    path : str
        Path to cuda root.
    """
    if "CUDA_PATH" in os.environ:
        return os.environ["CUDA_PATH"]
    cmd = ["which", "nvcc"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        return os.path.realpath(os.path.join(str(out).strip(), "../.."))
    cuda_path = "/usr/local/cuda"
    if os.path.exists(os.path.join(cuda_path, "bin/nvcc")):
        return cuda_path
    raise RuntimeError("Cannot find cuda path")


def get_cuda_version(cuda_path=None):
    """Utility function to get cuda version

    Parameters
    ----------
    cuda_path : Optional[str]

        Path to cuda root.  If None is passed, will use
        `find_cuda_path()` as default.

    Returns
    -------
    version : float
        The cuda version

    """
    if cuda_path is None:
        cuda_path = find_cuda_path()

    version_file_path = os.path.join(cuda_path, "version.txt")
    if not os.path.exists(version_file_path):
        # Debian/Ubuntu repackaged CUDA path
        version_file_path = os.path.join(cuda_path, "lib", "cuda", "version.txt")
    try:
        with open(version_file_path) as f:
            version_str = f.read().strip().split()[-1]
            return tuple(int(field) for field in version_str.split("."))
    except FileNotFoundError:
        pass

    cmd = [os.path.join(cuda_path, "bin", "nvcc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [l for l in out.split("\n") if "release" in l][0]
        release_fields = [s.strip() for s in release_line.split(",")]
        version_str = [f[1:] for f in release_fields if f.startswith("V")][0]
        return tuple(int(field) for field in version_str.split("."))
    raise RuntimeError("Cannot read cuda version file")




def fix_atomic_BSPMM(ori_code):
    # we should count the sub-kernel id to only fix the atomic add when need, 
    # or make it easier, we just count the index of each store of C and check whether we need to make it atomic.

    # print(ori_code)

    C_store_atomics = None
    with open(f"C_store_atomics{os.environ['MyFileID']}.json", 'r') as f:
        C_store_atomics = json.load(f)

    A_ell_max_shared, TC_C_SMEM = None, None
    with open(f"A_ell_max_shared{os.environ['MyFileID']}.json", 'r') as f:
        A_ell_max_shared, TC_C_SMEM = json.load(f)

    SMEM_pad_patterns = None
    with open(f"SMEM_pad_patterns{os.environ['MyFileID']}.json", 'r') as f:
        SMEM_pad_patterns = json.load(f)

    # print(ori_code)
    # print(C_store_atomics)

    ori_code = ori_code.split('\n')
    new_code = list()
    # 
    C_store_count = 0
    look_for_atomic_store = False
    C_local_vars_lines = list()
    C_local_cnt = 0
    read_SMEM_A_cnt = 0
    find_read_SMEM_A = False
    # alloc_TC_C_SMEM = False
    alloc_SMEM = False
    for line in ori_code:
        if (find_read_SMEM_A) and ((('C_local' in line) and ('=' in line)) or ('C[' in line)):
            read_SMEM_A_cnt += 1
            find_read_SMEM_A = False

        # ==================================================
        if ('__shared__' in line):
            if not alloc_SMEM:
                new_line = f"__shared__ {os.environ['MydtypeIn']} In_shared[{A_ell_max_shared}];\n"
                new_code.append(new_line)
                if TC_C_SMEM!=None:
                    new_line = f"__shared__ {os.environ['MydtypeOut']} C_shared[{TC_C_SMEM}];\n"
                    new_code.append(new_line)
                alloc_SMEM = True
            continue


        # ==================================================

        if ('B +' in line) and ('shared' in line):
            pos0 = line.find('_shared')
            pos1 = line.find('+')
            new_line = f'{line[:pos0 - 1]}In_shared {line[pos1:]}\n'
            new_code.append(new_line)
            continue

        if ('nvcuda::wmma::load_matrix_sync' in line) and ('B_shared' in line):
            pos0 = line.find('&')
            pos1 = line[pos0:].find('_shared') + pos0
            pos2 = line[pos0:].find('[') + pos0
            new_line = f'{line[:pos1-1]}In_shared{line[pos2:]}\n'
            new_code.append(new_line)
            continue

        if ('nvcuda::wmma::store_matrix_sync((&(' in line):
            # new_line = line
            # if 'C_shared' not in line:
            #     # pos0 = line.find('nvcuda::wmma::store_matrix_sync((&(') + len('nvcuda::wmma::store_matrix_sync((&(B_shared')
            #     pos1 = line.find('[')
            #     new_line = f'nvcuda::wmma::store_matrix_sync((&(C_shared{line[pos1:]}\n'
            # 
            pos1 = line.find('[')
            if 'shared' in line[:pos1]:        
                new_line = f'nvcuda::wmma::store_matrix_sync((&(C_shared{line[pos1:]}\n'            
                new_code.append(new_line)
            else:
                new_code.append(line+'\n')
                assert not C_store_atomics[C_store_count], "Should use atomic but directly write back to GMEM."
                C_store_count += 1
            continue


        if ('C[' in line) and ('shared' in line) and ('B[' not in line):
            # new_line = 'atomicAdd(C + ((((((I_indices0[0] * 8192) + (ax1_ax2_fused_0 * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ax1_ax2_fused_2), C_shared[(((ax1_ax2_fused_0 * 64) + (((int)threadIdx.x) * 2)) + ax1_ax2_fused_2)]);'
            # new_code.append(new_line)
            # continue
            if not C_store_atomics[C_store_count]:     
                pos0 = line.find('=')
                pos1 = line[pos0:].find('[') + pos0
                new_line = f'{line[:pos0]} = C_shared{line[pos1:]}\n'
                new_code.append(new_line)
                # continue
            else:
                # find the position of '='
                pos0 = line.find('C[')
                pos1 = line.find('=')
                pos2 = line.find(';')
                # 
                start = line[:pos0]
                address = line[ pos0+len('C['):pos1-2 ]
                val = line[ pos1+1:pos2 ]
                end = line[pos2:]
                pos4 = val.find('[')
                new_val = f'C_shared{val[pos4:]}'
                # 
                new_line = f"{start}atomicAdd(C + {address}, {new_val}){end}\n"
                new_code.append(new_line) 
                # continue 
            C_store_count += 1
            continue          
        # elif ('atomicAdd(C' in line):
        #     assert 'B_shared' in line
        #     pos0 = line.find('B_shared')
        #     pos1 = pos0 + len('B_shared')
        #     new_line = f'{line[:pos0]}C_shared{line[pos1:]}\n'
        #     new_code.append(new_line)
        #     continue

        # ==================================================
        if ('shared' in line) and ('=' in line) and ('C' not in line):
            # load data A from input to shared memory
            # e.g., A_shared[((int)threadIdx.x)] = A0[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))];
            find_read_SMEM_A = True
            # pos0 = line.find('_shared[') + len('_shared[')
            pos0 = line.find('[') + 1
            pos1 = line.find('] =')
            idx = line[pos0:pos1]
            # offset = idx / nncols / pattern[0] * pattern[1], nncols = pattern[2]
            # new_idx = idx+offset
            # print(f"*****     {line}")
            pattern = SMEM_pad_patterns[read_SMEM_A_cnt]
            new_idx = None
            if pattern[1] == 0:
                # no padding
                new_idx = f'{idx}'
            else:
                if math.log(pattern[1], 2) == int(math.log(pattern[1], 2)):
                    rmove_bit = int(math.log(pattern[2] * pattern[0], 2) - math.log(pattern[1], 2))
                    if rmove_bit > 0:
                        new_idx = f'{idx} + (({idx}) >> {rmove_bit})'
                    elif rmove_bit == 0:
                        new_idx = f'{idx} + {idx}'
                    else:
                        new_idx = f'{idx} + (({idx}) << {-rmove_bit})'
                else:
                    new_idx = f'{idx} + ((({idx}) >> {int(math.log(pattern[2] * pattern[0], 2))}) * {pattern[1]})'
            # new_line = f'{ line[:pos0] }{ new_idx }{ line[pos1:] }\n'
            new_line = f'In_shared[{ new_idx }{ line[pos1:] }\n'
            new_code.append(new_line)

        elif ('C_local' in line) and ('=' not in line):
            # declare C_local
            C_local_vars_lines.append(line)

        elif ('C_local' in line) and ('=' in line) and ('C[' not in line) and ('shared' not in line) and ('B' not in line):
            # initialize C_local
            # print(C_local_vars_lines, C_local_cnt)
            new_code.append(C_local_vars_lines[C_local_cnt]+'\n')
            C_local_cnt += 1
            new_code.append(line+'\n')

        elif ('C_local' in line) and ('=' in line) and ('C[' not in line) and ('B' in line): # and ('shared' in line):
            # compute C_local
            look_for_atomic_store = True
            if 'shared' in line:
                pos0 = line.find('+')
                pos2 = line[pos0:].find('[') + pos0
                new_line = f'{line[:pos0]}+ (In_shared{line[pos2:]}\n'
                new_code.append(new_line)
                # print("find one C local")
            else:
                new_code.append(line+'\n')
        
        elif 'C[' in line:
            # store C_local back to C
            if look_for_atomic_store:
                if C_store_atomics[C_store_count]:
                    # find the position of '='
                    pos0 = line.find('C[')
                    pos1 = line.find('=')
                    pos2 = line.find(';')
                    # 
                    start = line[:pos0]
                    address = line[ pos0+len('C['):pos1-2 ]
                    val = line[ pos1+1:pos2 ]
                    end = line[pos2:]
                    # 
                    new_line = f"{start}atomicAdd(C + {address}, {val}){end}\n"
                    new_code.append(new_line)
                else:
                    new_code.append(line+'\n')
                C_store_count += 1
                look_for_atomic_store = False
            else:
                if ('B' in line) and ('_shared' in line):
                    pos0 = line.find('_shared')
                    pos1 = line[pos0:].find('[') + pos0
                    new_line = f'{line[:pos0-1]}In_shared{line[pos1:]}\n'
                    new_code.append(new_line)
                else:
                    new_code.append(line+'\n')
        else:
            new_code.append(line+'\n')

    assert C_store_count == len(C_store_atomics), print(C_store_count, len(C_store_atomics))

    # print(''.join(new_code))

    return ''.join(new_code)




def deal_with_blockid_offset(code):
    new_code = list()
    start = None
    for i, line in enumerate(code):
        if 'main_kernel' in line:
            start = i
            break
    new_code = code[:start]
    
    for line in code[start:]:
        if '-' in line:
           pos0 = line.find('-')
            pos1 = line[pos0:].find(')')+pos0
            pos2 = line.find('((int)blockIdx.x)')
            pos3 = line[pos2+len('((int)blockIdx.x)'):].find('*')+pos2+len('((int)blockIdx.x)')
            pos4 = line[pos2+len('((int)blockIdx.x)'):].find(')')+pos2+len('((int)blockIdx.x)')
            if pos2 < pos0:
                # print(line[pos0:])
                # print(pos0, pos1, line)
                offset = int(line[pos0+1:pos1])
                unit = int(line[pos3+1:pos4])
                new_offset = offset // unit
                assert new_offset == offset/unit, f'Wrong offset and unit: {offset, unit} in LINE {line}'
                new_line = f"{line[:pos2]}(((int)blockIdx.x) - {new_offset}){line[pos2+len('((int)blockIdx.x)'):pos0 ]}{line[pos1:]}"
                new_code.append(new_line)
            else:
                itername = None
                if 'for' in new_code[-1]:
                    pos0 = new_code[-1].find('int') + len('int')
                    pos1 = new_code[-1].find('=')
                    itername = new_code[-1][pos0:pos1]
                else:
                    itername = '0'
                # print(f"itername: {itername}")
                pos0 = line.find('wmma_accumulator')+len('wmma_accumulator')+1
                pos1 = line[pos0:].find(']')+pos0
                new_line = f"{line[:pos0]}{itername}{line[pos1:]}"
                # print(line)
                # print(new_line)
                new_code.append(new_line)
        else:
            new_code.append(line)
    return new_code



def fix_atomic_BSDDMM(ori_code):

    # # FOR DEBUG-------------------------
    # return ori_code

    # if os.environ['use_code_from_file'] == 'True':
    #     code = None
    #     with open(f"Good_1D_CUDA{os.environ['MyFileID']}.cuda", 'r') as f:
    #         code = f.read()
    #     return code

    # if os.environ['has_32thread_SDDMM_cuda'] == 'True':
    #     with open(f"Good_32thread_hyb_SDDMM_CUDA{os.environ['MyFileID']}.cuda", 'r') as f:
    #         ori_code = f.read()


    # print(ori_code)
    ori_code = ori_code.split('\n')
    new_code = list()


    var_start, var_end, comp_start, comp_end = None, None, None, None
    variables, computation = None, None
    if os.environ['has_1d_tile'] == 'True':
        with open(f"Good_1D_CUDA{os.environ['MyFileID']}.cuda", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line == "NEXT IS COMPUTATION CODE\n":
                    variables = ''.join(lines[:i])
                    computation = ''.join(lines[i+1:])
                    break

        for i, line in enumerate(ori_code):
            if 'main_kernel' in line:
                var_start = i+1
            elif (var_start!=None) and ('nvcuda' in line):
                var_end = i
                break
        for i, line in enumerate(ori_code[var_end:]):
            if (comp_start==None) and ('blockIdx.x' in line):
                comp_start = var_end+i+1
            elif (comp_start!=None) and ('} else {' in line):
                comp_end = var_end+i
                break

    block_i = 0
    declare_sharemem = False
    for i, line in enumerate(ori_code):

        if os.environ['has_1d_tile'] == 'True':
            if i == var_start:
                new_code.append(variables)
                continue
            elif i == comp_start:
                new_code.append(computation)
                continue
            elif ((i > var_start) and (i < var_end)) or ((i > comp_start) and (i < comp_end)):
                continue

        # ===========================================================================
        if ('__shared__' in line) and (not declare_sharemem):
            if int(os.environ['A_SMEM'])>0:
                new_code.append(f"__shared__ {os.environ['MydtypeIn']} A_shared[{os.environ['A_SMEM']}];\n")
            if int(os.environ['B_SMEM'])>0:
                new_code.append(f"__shared__ {os.environ['MydtypeIn']} B_shared[{os.environ['B_SMEM']}];\n")
            if int(os.environ['C_SMEM'])>0:
                new_code.append(f"__shared__ {os.environ['MydtypeOut']} C_shared[{os.environ['C_SMEM']}];\n")
            declare_sharemem = True
            continue

        if (('B[' in line) or ('B +' in line)) and ('=' in line) and ('shared' in line) and ('C' not in line) and ('B_shared_wmma_matrix_b' not in line):
            assert 'A_shared' in line, line
            pos0 = line.find('A_shared')
            pos1 = pos0 + len('A_shared')
            new_line = f"{line[:pos0]}B_shared{line[pos1:]}\n"
            new_code.append(new_line)
            continue

        if ('load_matrix_sync' in line) and ('B_shared_wmma_matrix_b' in line):
            pos0 = line.find('A_shared')
            pos1 = pos0 + len('A_shared')
            new_line = f"{line[:pos0]}B_shared{line[pos1:]}\n"
            new_code.append(new_line)
            continue

        if ('store_matrix_sync' in line) and ('A_shared' in line):
            pos0 = line.find('A_shared')
            pos1 = pos0 + len('A_shared')
            new_line = f"{line[:pos0]}C_shared{line[pos1:]}\n"
            new_code.append(new_line)
            continue

        if ('C' in line) and ('A_shared' in line) and ('mma_sync' not in line):
            pos0 = line.find('A_shared')
            pos1 =  pos0 + len('A_shared')
            new_line = f"{line[:pos0]}C_shared{line[pos1:]}\n"
            new_code.append(new_line)
            continue

        new_code.append(line+'\n')

    
    new_code = deal_with_blockid_offset(new_code)

    # print(''.join(new_code))

    # with open(f"Good_32thread_hyb_SDDMM_CUDA{os.environ['MyFileID']}.cuda", 'w') as f:
    #     f.write(''.join(new_code))

    with open(f"mycuda.cuda", 'a') as f:
        f.write("New Code\n")
        f.write(''.join(new_code))


    return ''.join(new_code)





def fix_atomic(ori_code):
    if os.environ['op_type'] in ['spmm', 'batched_spmm']:
        return fix_atomic_BSPMM(ori_code)
    elif os.environ['op_type'] == 'sddmm':
        return fix_atomic_BSDDMM(ori_code)





@tvm._ffi.register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate fatbin code for better optimization"""
    # print("fix atomic bug---type of code: ", type(code))
    if 'MyFileID' in os.environ:
        code = fix_atomic(code)

    ptx = compile_cuda(code, target_format="fatbin")
    return ptx


@tvm._ffi.register_func("tvm_callback_libdevice_path")
def find_libdevice_path(arch):
    """Utility function to find libdevice

    Parameters
    ----------
    arch : int
        The compute architecture in int

    Returns
    -------
    path : str
        Path to libdevice.
    """
    cuda_path = find_cuda_path()
    lib_path = os.path.join(cuda_path, "nvvm/libdevice")
    if not os.path.exists(lib_path):
        # Debian/Ubuntu repackaged CUDA path
        lib_path = os.path.join(cuda_path, "lib/nvidia-cuda-toolkit/libdevice")
    selected_ver = 0
    selected_path = None
    cuda_ver = get_cuda_version(cuda_path)
    major_minor = (cuda_ver[0], cuda_ver[1])
    if major_minor in (
        (9, 0),
        (9, 1),
        (10, 0),
        (10, 1),
        (10, 2),
        (11, 0),
        (11, 1),
        (11, 2),
        (11, 3),
    ):
        path = os.path.join(lib_path, "libdevice.10.bc")
    else:
        for fn in os.listdir(lib_path):
            if not fn.startswith("libdevice"):
                continue

            try:
                # expected pattern: libdevice.${ARCH}.10.bc
                #             e.g., libdevice.compute_20.10.bc
                ver = int(fn.split(".")[-3].split("_")[-1])
                if selected_ver < ver <= arch:
                    selected_ver = ver
                    selected_path = fn
            except ValueError:
                # it can just be `libdevice.10.bc` in CUDA 10
                selected_path = fn

        if selected_path is None:
            raise RuntimeError("Cannot find libdevice for arch {}".format(arch))
        path = os.path.join(lib_path, selected_path)
    return path


def callback_libdevice_path(arch):
    try:
        return find_libdevice_path(arch)
    except RuntimeError:
        warnings.warn("Cannot find libdevice path")
        return ""


def get_target_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "8.6")
    """
    # 1. input target object
    # 2. Target.current()
    target = target or Target.current()
    if target and target.arch:
        major, minor = target.arch.split("_")[1]
        return major + "." + minor

    # 3. GPU compute version
    if tvm.cuda(0).exist:
        return tvm.cuda(0).compute_version

    raise ValueError(
        "No CUDA architecture was specified or GPU detected."
        "Try specifying it by adding '-arch=sm_xx' to your target."
    )


def parse_compute_version(compute_version):
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error: " + str(err))


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, minor = parse_compute_version(compute_version)
    # fp 16 support in reference to:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions
    if major == 5 and minor == 3:
        return True
    if major >= 6:
        return True

    return False


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, _ = parse_compute_version(compute_version)
    if major >= 6:
        return True

    return False


def have_tensorcore(compute_version=None, target=None):
    """Either TensorCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    target : tvm.target.Target, optional
        The compilation target, will be used to determine arch if compute_version
        isn't specified.
    """
    if compute_version is None:
        if tvm.cuda(0).exist:
            compute_version = tvm.cuda(0).compute_version
        else:
            if target is None or "arch" not in target.attrs:
                warnings.warn(
                    "Tensorcore will be disabled due to no CUDA architecture specified."
                    "Try specifying it by adding '-arch=sm_xx' to your target."
                )
                return False
            compute_version = target.attrs["arch"]
            # Compute version will be in the form "sm_{major}{minor}"
            major, minor = compute_version.split("_")[1]
            compute_version = major + "." + minor
    major, _ = parse_compute_version(compute_version)
    if major >= 7:
        return True

    return False


def have_cudagraph():
    """Either CUDA Graph support is provided"""
    try:
        cuda_ver = get_cuda_version()
        if cuda_ver < (10, 0):
            return False
        return True
    except RuntimeError:
        return False


def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    major, _ = parse_compute_version(compute_version)
    if major >= 8:
        return True

    return False
