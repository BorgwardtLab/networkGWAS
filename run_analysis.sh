#!/bin/bash

# Inputs definition
INFOLD='data/' # folder where all the input data is stored (genotype, phenotype, etc.)
MAPPING='gene_snps_index.pkl' # name of the gene-snps mapping file
NETWORK='PPI_adj.pkl'		  # name of the PPI network file
GENOTYPE='genotype'			  # name of the genotype file
PHENO="y_50.pheno"			  # name of the phenotype file

# Parameters definition
NPERM=100 # number of permutations
KERNEL='lin' # type of kernel. Either linear or polynomial
Q1=0.05 # FDR level to control for

# Output definitions
SETTINGS='results/settings/' # where to store the aggregated neighborhoods on the original setting
PERMNW='results/settings/permutations/networks/' # where to store the permuted networks
PERMNB='results/settings/permutations/neighborhoods/' # where to store the aggregated neighborhoods on the permuted settings
ORIGINAL="results/llr/" # where to store the lrt statistic(s) on the original setting(s)
PERMUTED='results/llr/permuted/' # where to store the lrt statistics on the permuted settings
NULLDIR='results/null_distr/' # where to store the distribution of the statistics under the null hypothesis
PVALDIR='results/pvals/' # where to store the p-values


echo "1. Neighborhood aggregation..."

python3 1_nb_aggregation.py \
--i $INFOLD \
--o $SETTINGS \
--g2s $MAPPING \
--bim "$GENOTYPE".bim \
--nw  $NETWORK \
--nbs neighborhoods.txt


echo "2. Permuted settings computation..."

python3 2_circPerm_nwPerm.py \
--i $INFOLD \
--g2s $MAPPING \
--bim "$GENOTYPE".bim \
--nw $NETWORK \
--perm $NPERM \
--alpha 0.5 \
--seed 42 \
--onwdir $PERMNW \
--onbdir $PERMNB \
--onw nw_ \
--onb nbs_


echo "3. Lrt calculation on the non-permuted setting..."

python3 3_LMM.py \
--genotype "$INFOLD""$GENOTYPE" \
--phenotype "$INFOLD""$PHENO" \
--nbs "$SETTINGS"neighborhoods.txt \
--kernel $KERNEL \
--odir "$ORIGINAL" \
--ofile llr.pkl

echo "4. Lrt calculation on the permuted settings..."

python3 3_LMM.py \
--genotype "$INFOLD""$GENOTYPE" \
--phenotype "$INFOLD""$PHENO" \
--nbs "$PERMNB"nbs_ \
--kernel $KERNEL \
--j 0 $NPERM \
--odir $PERMUTED \
--ofile llr_


echo "5. p-values calculation..."

python3 4_obtain_pvals.py \
--inpath "$ORIGINAL"llr.pkl \
--inpathperm "$PERMUTED"llr_ \
--nperm $NPERM \
--dirnd $NULLDIR \
--dirpv $PVALDIR \
--fignd null_distr.png \
--figpv qqplot.png \
--outpathnd null_distr.pkl \
--outpathpv pvals.pkl


echo "6. Associated neighborhoods..."

python3 5_associated_neighborhoods.py \
--nw "$INFOLD""$NETWORK" \
--pv "$PVALDIR"pvals.pkl \
--q1 $Q1 
