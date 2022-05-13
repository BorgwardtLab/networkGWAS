# networkGWAS
## METHOD
This repository contains the python implementation of networkGWAS method, which foresees the following steps:
1) neighborhood aggregation of the SNPs according to the biological network ([**1_nb_aggregation.py**](1_nb_aggregation.py));
2) 2-level permutation procedure, which combines a circular permutation of the SNPs and degree-preserving permutation of the network ([**2_circPerm_nwPerm.py**](2_circPerm_nwPerm.py));
3) calculation of the log-likelihood ratio (lrt) test statistics on both the non-permuted setting and the permuted settings by employing the modified FaST-LMM snp-set function, which is available in the folder "LMM" ([**3_LMM.py**](3_LMM.py));
4) calculation of the p-values. Per each neighborhood, this is done by: (i) obtaining the null distribution through the pooling of the lrt statitistics on the permuted settings, (ii) calculating the ratio of statistics in the null distribution that are larger than or equal to the statistic for the neighborhood obtained on the non-permuted setting ([**4_obtain_pvals.py**](4_obtain_pvals.py)).
5)

Available here a toy-dataset on which try the method. Detail below.

## TOY DATA
The toy dataset is comprised of:
- genotype file in Plink .bed/.bim/.fam format ("data/genotype.bed", "data/genotype.bim", "data/genotype.fam")
- simulated phenotype in Plink .pheno format ("data/y_50.pheno")
- the numpy array of the causal genes set as causal in the simulation of the phenotype ("data/genes_50.pkl")
- the adjacency matrix of the biological network, i.e., the protein-protein interaction (PPI) network in Pickle format ("data/PPI_adj.pkl"). In particular, the adjacency matrix is saved as a pandas dataframe, where the index and columns are the names of the genes, which represents the nodes of the PPI network.
- the dictionary presenting the nodes (gene) as keys with as values a numpy boolean vector that presents True where the SNPs (ordered according to the .bim file) are mapped onto that particular gene ("data/gene_snps_index.pkl")

For more details on the Plink formats, please refer to https://www.cog-genomics.org/plink/2.0/formats.

## EXAMPLES

## DATA AVAILABILITY
In order to reproduce the results presented in the manuscript, here a list of the data availabilities:

### Semi-simulated common-variant setting
- **genotype**: the full imputed version of the _A. thaliana_ genotype is available on the AraGWAS database (https://aragwas.1001genomes.org/#/download-center);
- **phenotype**: simulated according to the procedure detailed in [1];
- **PPI network**: the PPI network is downloaded from the The Arabidopsis Information Resource (TAIR) database (https://www.arabidopsis.org/).

### Fully synthetic rare-variant setting
- **genotype**: simulated using sim1000G package [4] giving as input the VCF from Phase III 1000 genomes sequencing data;
- **phenotype**: simulated according to the procedure detailed in [1];
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for the model organism _H. sapiens_.

### _A. thaliana_ 
- **genotype**: the full imputed version of the _A. thaliana_ genotype is available on the AraGWAS database (https://aragwas.1001genomes.org/#/download-center);
- **phenotype**: the natural phenotypes for _A. thaliana_ have been downloaded from the AraPheno database (https://arapheno.1001genomes.org/phenotypes/);
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for the model organism _A. thaliana_.

### _S. cerevisiae_
- **genotype & phenotype**: genotype and phenotype for the model organism _S. cerevisiae_ are available at http://1002genomes.u-strasbg.fr/files/, which is the repository of the publication [3]; 
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for the model organism _S. cerevisiae_.

## DEPENDENCIES
The code only supports python3 and requires the following packages and submodules:
+ numpy (tested on 1.18.1)
+ pandas (tested on 1.0.1)
+ fastlmm (https://github.com/fastlmm/FaST-LMM)


## REFERENCES

[1] G. Muzio, L. O’Bray, L. Meng-Papaxanthos, J. Klatt, K. Borgwardt, networkGWAS: A network-based approach for genome-wide association studies in structured populations, bioRxiv, https://doi.org/10.1101/2021.11.11.468206.

[2] C. Lippert, J. Xiang, D. Horta, C. Widmer, C. Kadie, D. Heckerman, J. Listgarten, Greater power and computational efficiency for kernel-based association testing of sets of genetic variants, Bioinformatics 30(22):3206–3214, 2014.

[3] J. Peter, M. De Chiara, A. Friedrich et al., Genome evolution across 1,011 Saccharomyces cerevisiae isolates. Nature 556, 339–344, 2018.

[4] A. Dimitromanolakis, J. Xu, A. Krol, L. Briollais, sim1000g: a user-friendly genetic variant simulator in R for unrelated individuals and family-based designs. BMC Bioinformatics 20(26), 2019.

## CONTACT
giulia.muzio@bsse.ethz.ch
