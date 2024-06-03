## The code organization:

MY_sparse: the source code of STile

- gen_formats_v2.py:        about sparse tile generation
 
- my_search_formats.py:     about the greedy algorithm
 
- my_cost_model.py:         about the cost model
 
- my_branch_and_bound.py:	  about local search and the withdraw technique
 
- my_fuse_formats.py:       about code generation
 
- my_wmmas1.py:             help functions for tensor core computation


SPMM: the script to run experiments on SpMM

- ~~my_run_graph.py:            	run experiment on graph adjacency matrices~~
 
- ~~my_run_prunedbert.py:       	run experiment on structured and unstructured matrices~~
 
- ~~my_run_pureformat.py:       	run experiment when only considering one basic format~~
 
- ~~my_run_localsearch.py:      	do ablation study of the local search influence~~
 
- ~~my_run_withdraw.py:         	do ablation study of the withdraw technique influence~~
 
- ~~my_run_cost_model_PERF.py:  	collect cost model performance data~~
 
- utils.py:                   	help functions

- SPMMbench_our_method.py: 	test different versions of our method.

- SPMM*.py files, SPMM_NotreDame_run_sparsetir.sh:	test different baselines.
 


SDDMM: the script to run experiments on SDDMM

- ~~bench_our_method.py:		run experiment on graph adjacency matrices, structured matrices, and unstructured matrices~~
 
- ~~bench_sddmm_for_test.py:	help functions~~

- SDDMMbench_our_method.py: test different versions of our method.

- SDDMM*.py files: test different baselines.



nvcc.py:	replace the nvcc.py in SparseTIR with this nvcc.py to generate CUDA code correctly

codegen_c.cc:   replace the file sparsetir-artifact/3rdparty/SparseTIR/src/target/source/codegen_c.cc with this codegen_c.cc file and recompile SparseTIR in the container


## How to run the experiments:

1. Enter the Docker container of [sparsetir-artifact](https://github.com/uwsampl/sparsetir-artifact/tree/main).

2. Replace "nvcc.py" and "codegen_c.cc" with the corresponding files in this repository, and recompile SparseTIR.
  
3. Run the scripts in SPMM and SDDMM.


