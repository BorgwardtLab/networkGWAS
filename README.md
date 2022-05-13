# networkGWAS
## METHOD
This repository contains the python implementation of networkGWAS method, which foresees the following steps:
- ([**1_nb_aggregation.py**](1_nb_aggregation.py))
- ([**2_circPerm_nwPerm.py**](2_circPerm_nwPerm.py))
- ([**3_LMM.py**](3_LMM.py))
- ([**4_obtain_pvals.py**](4_obtain_pvals.py))
Available here a toy-dataset on which try the method. Detail below.

## DATA
The toy dataset is comprised of:
- 

In particular:
-

- snp_matrix in bed/bim/fam format
- phenotype in pheno format
These formats, which are the classic plink formats (for more details, see https://www.cog-genomics.org/plink/2.0/formats)

## DEPENDENCIES
The code only supports python3 and requires the following packages and submodules:
+ numpy (tested on 1.18.1)
+ pandas (tested on 1.0.1)
+ fastlmm (https://github.com/fastlmm/FaST-LMM)


## REFERENCES

[1] Listgarten, J. et al, A powerful and efficient set test for genetic markers that handles confounders.
Bioinformatics, 2013.

## CONTACT
giulia.muzio@bsse.ethz.ch
