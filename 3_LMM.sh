#!bin/bash

for J in {3..99}; do
	python3 3_LMM.py --nbs results/settings/permutations/neighborhoods/nbs_ --kernel lin --odir results/llr/permuted --j $J --ofile llr_ 
done