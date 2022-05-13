# networkGWAS
## METHOD
This repository contains the python implementation of networkGWAS method, which foresees the following steps:
- neighborhood aggregation of the SNPs according to the biological network ([**1_nb_aggregation.py**](1_nb_aggregation.py));
- 2-level permutation procedure, which combines a circular permutation of the SNPs and degree-preserving permutation of the network ([**2_circPerm_nwPerm.py**](2_circPerm_nwPerm.py));
- calculation of the log-likelihood ratio (lrt) test statistics on both the non-permuted setting and the permuted settings by employing the modified FaST-LMM snp-set function ([**3_LMM.py**](3_LMM.py));
- calculation of the p-values. Per each neighborhood, this is done by: (i) obtaining the null distribution through the pooling of the lrt statitistics on the permuted settings, (ii) calculating the ratio of statistics in the null distribution that are larger than or equal to the statistic for the neighborhood obtained on the non-permuted setting ([**4_obtain_pvals.py**](4_obtain_pvals.py)).
Available here a toy-dataset on which try the method. Detail below.

## TOY DATA
The toy dataset is comprised of:
- genotype file in Plink .bed/.bim/.fam format ("data/genotype.bed", "data/genotype.bim", "data/genotype.fam")
- phenotype in Plink .pheno format ("data/y_50.pheno")
- the adjacency matrix of the biological network, i.e., the protein-protein interaction network in Pickle format ("data/PPI_adj.pkl"). In particular, the adjacency matrix is saved as a pandas dataframe, where the index and columns are the names of the proteins, which represents the nodes of the PPI network.

For more details on the Plink formats, please refer to https://www.cog-genomics.org/plink/2.0/formats.

## 


## DEPENDENCIES
The code only supports python3 and requires the following packages and submodules:
+ numpy (tested on 1.18.1)
+ pandas (tested on 1.0.1)
+ fastlmm (https://github.com/fastlmm/FaST-LMM)


## REFERENCES

[1] G. Muzio, L. O’Bray, L. Meng-Papaxanthos, J. Klatt, K. Borgwardt, networkGWAS: A network-based approach for genome-wide association studies in structured populations, bioRxiv, https://doi.org/10.1101/2021.11.11.468206.

[2] C. Lippert, J. Xiang, D. Horta, C. Widmer, C. Kadie, D. Heckerman, J. Listgarten, Greater power and computational efficiency for kernel-based association testing of sets of genetic variants, Bioinformatics, Volume 30, Issue 22, Pages 3206–3214, 2014.

## CONTACT
giulia.muzio@bsse.ethz.ch
