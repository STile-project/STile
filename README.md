# How to run the experiments:

### Pull a Singularity image from the Docker image of SparseTIR  
```bash
singularity pull sparsetir-ae.sif docker://expye/sparsetir-ae:latest
singularity shell --nv sparsetir-ae.sif
export PS1='Singularity:\u@\h:\w>'
```
### Compile STile
```bash
export HOME=path/of/current_directory
git clone -b artifact https://github.com/STile-project/STile.git --recursive
cd $HOME/STile/3rdparty/SparseTIR
bash docker/install/install_sparsetir_gpu.sh
```

### Prepare environment
```bash
cd $HOME/STile
mkdir my_python_libs

export PYTHONPATH=$HOME/STile/python:$PYTHONPATH
export PYTHONPATH=$HOME/STile/MY_sparse:$PYTHONPATH
export PYTHONPATH=$HOME/STile/SPMM:$PYTHONPATH
export PYTHONPATH=$HOME/STile/my_python_libs:$PYTHONPATH

pip3 install --target=$HOME/STile/my_python_libs numpy==1.24.3
pip3 install --target=$HOME/STile/my_python_libs dgl==1.1.0+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
```

### Prepare dataset for VectorSparse
```bash
cd $HOME/STile/SPMM
mkdir data_for_Magicube
python3 my_gen_data_for_baselines.py
```

### Example commands to run experiments
```bash
# run STile for SpMM experiments
cd $HOME/sparsetir-artifact/SPMM
python3 SPMMbench_our_method.py > our_method_fp16.log 2> our_method_fp16.err

# run STile for SDDMM experiments
cd $HOME/sparsetir-artifact/SDDMM
python3 SDDMMbench_our_method.py > our_method_fp16.log 2> our_method_fp16.err

# run VectorSparse for SDDMM experiments
cd $HOME/sparsetir-artifact/my_3rdparty/Magicube/Magicube1/baselines
python3 my_launch_spmm_vectorSparse.py > my_spmm_vectorSparse.txt 2> my_spmm_vectorSparse.err
python3 my_launch_sddmm_vectorSparse.py > my_sddmm_vectorSparse.txt 2> my_sddmm_vectorSparse.err

# commands to run other baselines are similar to the ones to run STile
```

~~1. Enter the Docker container of [sparsetir-artifact](https://github.com/uwsampl/sparsetir-artifact/tree/main).~~

~~2. Replace "nvcc.py" and "codegen_c.cc" with the corresponding files in this repository, and recompile SparseTIR.~~ (We have replaced these two files in this branch.)
  
~~3. Run the scripts in SPMM and SDDMM.~~ 






# The code organization:

## MY_sparse: the source code of STile

> - gen_formats_v2.py:        about sparse tile generation
>  
> - my_search_formats.py:     about the greedy algorithm
>  
> - my_cost_model.py:         about the cost model
>  
> - my_branch_and_bound.py:	  about local search and the withdraw technique
>  
> - my_fuse_formats.py:       about code generation
>  
> - my_wmmas1.py:             help functions for tensor core computation


## SPMM: the script to run experiments on SpMM

> - utils.py:                   	help functions
>
> - SPMMbench_our_method.py: 	test different versions of our method.
>
> - SPMM*.py files, SPMM_NotreDame_run_sparsetir.sh:	test different baselines.

> - ~~my_run_graph.py:            	run experiment on graph adjacency matrices~~
>
> - ~~my_run_prunedbert.py:       	run experiment on structured and unstructured matrices~~
> 
> - ~~my_run_pureformat.py:       	run experiment when only considering one basic format~~
> 
> - ~~my_run_localsearch.py:      	do ablation study of the local search influence~~
> 
> - ~~my_run_withdraw.py:         	do ablation study of the withdraw technique influence~~
> 
> - ~~my_run_cost_model_PERF.py:  	collect cost model performance data~~

 


## SDDMM: the script to run experiments on SDDMM

> - SDDMMbench_our_method.py: test different versions of our method.
> 
> - SDDMM*.py files: test different baselines.

> - ~~bench_our_method.py:		run experiment on graph adjacency matrices, structured matrices, and unstructured matrices~~
>  
> - ~~bench_sddmm_for_test.py:	help functions~~




~~nvcc.py:	replace the file sparsetir-artifact/3rdparty/SparseTIR/python/tvm/contrib/nvcc.py in SparseTIR with this nvcc.py to generate CUDA code correctly~~

~~codegen_c.cc:   replace the file sparsetir-artifact/3rdparty/SparseTIR/src/target/source/codegen_c.cc with this codegen_c.cc file and recompile SparseTIR in the container~~





