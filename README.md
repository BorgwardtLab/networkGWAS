# neighborGWAS
## METHOD
This repository contains the python implementation of neighborGWAS method, which foresees the following steps:
- calculation of original p-values capturing the association score of each neighbourhood and the phenotype of interest; this is done by using Fast-LMM snp-set function [1] (obtain_original_pvalues.py)
- swapping of the genes according to degree-preserving permutation strategy (gene_swapping.py)
- generation of the neighbourhoods aggregation files on the permuted networks; the structure of the network remains the same, what changes is the mapping of the SNPs, which is defined in the previous step (new_neighbourhood.py)
- obtaining the statistics by using Fast-LMM snp-set function on the permuted data (obtain_pvalues_permuted_network.py); this script accepts as command-line argument the index of the permutation. This structure allows to parallelise the computation when possible. An acceptable number of permutations is 1000.
- calculation of the null distribution and adjusted pvalues from the original p-values and the p-values obtained on the permuted networks; reporting of the contingency table, precision and recall (null_distr_adj_pvals.py)
Available here a toy-dataset on which try the method. Detail below.

## DATA
The toy dataset is comprised of:
- a SNP matrix with 200 columns (corresponding to the SNPs) and 200 rows (corresponding to the samples, e.g. individuals)
- a ppi network in form of a binary adjacency matrix (e.g. 1 when there is an interaction between 2 genes (proteins), and 0 otherwise)
- a mapping between the genes (involved in the PPI network) and the SNPs
- the same network expressed in form of interactions between the SNPs

In particular:
- gene_name.pkl contains the names of the genes
- mapping.pkl contains the array of snps names with the corresponding gene onto which each snp is mapped
- ppi.pkl is a pandas dataframe with the binary adjacency matrix of the PPI network
- causal.pkl is a numpy array containing the names of the causal genes in our simulation setting
- neighborhood_list.txt is the .txt file containing the SNPs representing the 1-degree neighbourhoods on the real network. This is the result of the 1-degree neighbourhood aggregation per each gene, e.g. the union of the SNPs of the gene and the SNPs of its 1-degree neighbours.

The folder named "plink" contains:
- snp_matrix in ped/map format
- snp_matrix in bed/bim/fam format
- phenotype in pheno format
These formats, which are the classic plink formats (for more details, see https://www.cog-genomics.org/plink/2.0/formats)

## DEPENDENCIES
The code only supports python3 and requires the following packages and submodules:
+ numpy (tested on 1.18.1)
+ pandas (tested on 1.0.1)
+ fastlmm 


## REFERENCES

[1] Listgarten, J. et al, A powerful and efficient set test for genetic markers that handles confounders.
Bioinformatics, 2013.
