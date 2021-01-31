# neighborGWAS
## METHOD
This repository contains the python implementation of neighborGWAS method, which foresees the following steps:
- calculation of original p-values by using Fast-LMM snp-set function [1] (1_obtain_original_pvalues.py)
- swapping of the genes according to degree-preserving permutation strategy (2_1_gene_swapping.py)
- generation of the neighbourhoods aggregation files on the permuted networks; the structure of the network remains the same, what changes is the mapping of the SNPs, which is defined in the previous step (2_2_new_neighbourhood.py)
- obtaining the statistics by using Fast-LMM snp-set function on the permuted data (3_obtain_pvalues_permuted_network.py)
- calculation of the null distribution and adjusted pvalues

## DATA
All the above methods are implemented in python3, and are tested on a toy dataset comprised of:
- a SNP matrix with 200 columns (corresponding to the SNPs) and 200 rows (corresponding to the samples, e.g. individuals)
- a ppi network in form of a binary adjacency matrix (e.g. 1 when there is an interaction between 2 genes (proteins), and 0 otherwise)
- a mapping between the genes (involved in the PPI network) and the SNPs
- the same network expressed in form of interactions between the SNPs

In particular:
- gene_name.pkl contains the names of the genes
- mapping.pkl contains the array of snps names with the corresponding gene onto which each snp is mapped
- ppi.pkl is a pandas dataframe with the binary adjacency matrix of the PPI network

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
