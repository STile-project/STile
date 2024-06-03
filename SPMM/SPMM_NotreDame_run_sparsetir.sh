printf "\n\n\n\nNew Round---------\n"  >>  Mem_SparseTIR_fp16.json

# python3 bench_spmm_hyb.py -d ppi > Mem_SparseTIR_fp16_1.log 2> Mem_SparseTIR_fp16_1.err
# python3 bench_spmm_hyb.py -d cora >> Mem_SparseTIR_fp16_1.log 2>> Mem_SparseTIR_fp16_1.err
# python3 bench_spmm_hyb.py -d citeseer >> Mem_SparseTIR_fp16_1.log 2>> Mem_SparseTIR_fp16_1.err
# python3 bench_spmm_hyb.py -d pubmed >> Mem_SparseTIR_fp16_1.log 2>> Mem_SparseTIR_fp16_1.err
# python3 bench_spmm_hyb.py -d arxiv >> Mem_SparseTIR_fp16_1.log 2>> Mem_SparseTIR_fp16_1.err
# python3 bench_spmm_hyb.py -d proteins >> Mem_SparseTIR_fp16_1.log 2>> Mem_SparseTIR_fp16_1.err
# python3 bench_spmm_hyb.py -d reddit >> Mem_SparseTIR_fp16_1.log 2>> Mem_SparseTIR_fp16_1.err

for name in out.web-NotreDame strided logsparse
do
	for size in 4 8 16 32 64 128 256 512
	do
		for cols in 1 2 4 8 16
		do
			echo ${name} ${size} ${cols}
			echo "python3 bench_spmm_hyb.py -d ${name} -s ${size} -c ${cols} >> Mem_SparseTIR_fp16_2.log 2>> Mem_SparseTIR_fp16_2.err"
			python3 bench_spmm_hyb.py -d ${name} -s ${size} -c ${cols} >> Mem_SparseTIR_fp16_2.log 2>> Mem_SparseTIR_fp16_2.err
		done
	done
done
