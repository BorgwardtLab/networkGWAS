# networkGWAS

This repository contains the python implementation of networkGWAS method, which foresees the following steps:
1) neighborhood aggregation of the SNPs according to the biological network ([**1_nb_aggregation.py**](1_nb_aggregation.py));
2) 2-level permutation procedure, which combines a circular permutation of the SNPs and degree-preserving permutation of the network ([**2_circPerm_nwPerm.py**](2_circPerm_nwPerm.py));
3) calculation of the log-likelihood ratio (lrt) test statistics on both the non-permuted setting and the permuted settings by employing the modified FaST-LMM snp-set function, which is available in the folder "LMM" ([**3_LMM.py**](3_LMM.py));
4) calculation of the _p_-values. Per each neighborhood, this is done by: (i) obtaining the null distribution through the pooling of the lrt statitistics on the permuted settings, (ii) calculating the ratio of statistics in the null distribution that are larger than or equal to the statistic for the neighborhood obtained on the non-permuted setting ([**4_obtain_pvals.py**](4_obtain_pvals.py)).
5) identifying the statistically associated neighborhood by means of the Benjamini-Hochberg (B-H) procedure in case of analysing one phenotype only or by using the hierarchical procedure based on B-H procedure in case of multiple phenotypes ([**5_associated_neighborhoods.py**](5_associated_neighborhoods.py)).

Available here a toy-dataset on which try the method. Detail below.

## TOY DATA
The toy dataset is comprised of:
- genotype file in Plink .bed/.bim/.fam format ("data/genotype.bed", "data/genotype.bim", "data/genotype.fam")
- simulated phenotype in Plink .pheno format ("data/y_50.pheno")
- the adjacency matrix of the biological network, i.e., the protein-protein interaction (PPI) network in Pickle format ("data/PPI_adj.pkl"). In particular, the adjacency matrix is saved as a pandas dataframe, where the index and columns are the names of the genes, which represents the nodes of the PPI network.
- the dictionary presenting the nodes (gene) as keys with as values a numpy boolean vector that presents True where the SNPs (ordered according to the .bim file) are mapped onto that particular gene ("data/gene_snps_index.pkl")

For more details on the Plink formats, please refer to https://www.cog-genomics.org/plink/2.0/formats.

In case the user has a PPI network in two columns format (without header), that is:

```
gene1	gene4
gene1	gene2
...	...
```
it is possible to convert it to the required dataframe (i.e., adjacency matrix) format using the command:
```
python3 convert_ppi.py --i data/ppi_edges.txt --o data/PPI_adj.pkl
```


## DEPENDENCIES
The code only supports python3 and requires the following packages and submodules:
+ numpy (tested on 1.18.1)
+ pandas (tested on 1.0.1)
+ fastlmm (https://github.com/fastlmm/FaST-LMM) (tested on 0.4.8)
+ pysnotools (tested on 0.4.19)

## EXAMPLE 1
#### 1) running the neighborhood aggregation operation:
```
python3 1_nb_aggregation.py \
--i data \
--o results/settings \
--g2s gene_snps_index.pkl \
--bim genotype.bim \
--nw PPI_adj.pkl \
--nbs neighborhoods.txt
```
#### 2) running the generation of the permuted settings:
```
python3 2_circPerm_nwPerm.py \
--i data \
--g2s gene_snps_index.pkl \
--bim genotype.bim \
--nw PPI_adj.pkl \
--perm 100 \
--alpha 0.5 \
--seed 42 \
--onwdir results/settings/permutations/networks/ \
--onbdir results/settings/permutations/neighborhoods/ \
--onw nw_ \
--onb nbs_
```
Since this command runs the generation of 100 permuted settings, it might take a few minutes, depending on the resources available.
#### 3.1) running the lrt calculation on the non-permuted setting:
```
python3 3_LMM.py \
--genotype data/genotype \
--phenotype data/y_50.pheno \
--nbs results/settings/neighborhoods.txt \
--kernel lin \
--odir results/llr/ \
--ofile llr.pkl
```
#### 3.2) running the lrt calculation on _one_ of the permuted settings, i.e., the permuted setting with index (or permutation id) j = 0:
```
python3 3_LMM.py \
--genotype data/genotype \
--phenotype data/y_50.pheno \
--nbs results/settings/permutations/neighborhoods/nbs_ \
--kernel lin \
--j 0 \
--odir results/llr/permuted \
--ofile llr_
```

**IMPORTANT:** This command has to be run for enough permutations, e.g., in this example we use 100 permutations. This can be done by changing the value of the input parameter ```--j``` with a value from 0 to the maximum permutation id available, which in this example would be 99. Note that since the permutation id (```--j```) is a command line argument, [**3_LMM.py**](3_LMM.py) for the different permuted settings can be run in parallel. When enough resources should not be available to run multiple permutations in parallel, it is possible to run the lrt calculation sequentially by specifying a range of permutations. The following command calculates the lrt values on the permuted settings from 0 to 99:

```
python3 3_LMM.py \
--genotype data/genotype \
--phenotype data/y_50.pheno \
--nbs results/settings/permutations/neighborhoods/nbs_ \
--kernel lin \
--j 0 99 \
--odir results/llr/permuted \
--ofile llr_
```

#### 4) obtaining the _p_-values:
```
python3 4_obtain_pvals.py \
--inpath results/llr/llr.pkl \
--inpathperm results/llr/permuted/llr_ \
--nperm 100 \
--dirnd results/null_distr/ \
--dirpv results/pvals/ \
--fignd null_distr.png \
--figpv qqplot.png \
--outpathnd null_distr.pkl \
--outpathpv pvals.pkl
```
#### 5) identifying the statistically associated neighborhoods in case of one phenotype:
```
python3 5_associated_neighborhoods.py \
--nw 'data/PPI_adj.pkl' \
--pv 'results/pvals/pvals.pkl' \
--q1 0.05 
```
or  identifying the statistically associated neighborhoods in case of multiple phenotypes that are from the same study:
```
python3 5_associated_neighborhoods.py \
--nw 'data/PPI_adj.pkl' \
--pv 'results/pvals/pvals_pheno1.pkl' 'results/pvals/pvals_pheno2.pkl' 'results/pvals/pvals_pheno3.pkl' \
--q1 0.05 \
--q2 0.05
```
Note that this last command would work only after having obtained the _p_-values for "pheno1'', "pheno2'', and "pheno3''. 

## EXAMPLE 2
It is also possible to run the entire analysis using one script, namely:

```
bash run_analysis.sh
```
The user would need to change the input files and parameters in the file according to their data. 

## DATA AVAILABILITY
In order to reproduce the results presented in the manuscript, here a list of the data availabilities:

### Semi-simulated common-variant setting
- **genotype**: the full imputed version of the _A. thaliana_ genotype is available on the AraGWAS database (https://aragwas.1001genomes.org/#/download-center); genetic variants presenting MAF < 0.05 have been filtered out;
- **phenotype**: simulated according to the procedure detailed in [1];
- **PPI network**: the PPI network is downloaded from the The Arabidopsis Information Resource (TAIR) database (https://www.arabidopsis.org/).

### Fully synthetic rare-variant setting
- **genotype**: simulated using sim1000G package [4] giving as input the VCF from Phase III 1000 genomes sequencing data; the simulated SNPs present MAF < 0.1;
- **phenotype**: simulated according to the procedure detailed in [1];
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for the model organism _H. sapiens_.

### _A. thaliana_ natural phenotypes
- **genotype**: the full imputed version of the _A. thaliana_ genotype is available on the AraGWAS database (https://aragwas.1001genomes.org/#/download-center); genetic variants presenting MAF < 0.05 have been filtered out;
- **phenotype**: the natural phenotypes for _A. thaliana_ have been downloaded from the AraPheno database (https://arapheno.1001genomes.org/phenotypes/);
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for the model organism _A. thaliana_.

### _S. cerevisiae_ natural phenotypes
- **genotype & phenotype**: genotype and phenotype for the model organism _S. cerevisiae_ are available at http://1002genomes.u-strasbg.fr/files/, which is the repository of the publication [3]; 
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for the model organism _S. cerevisiae_.

### Estonian BioBank phenotypes
- **genotype & phenotype**: genotype and phenotype are obtained from the Estonian BioBank dataset (https://genomics.ut.ee/en/content/estonian-biobank); 
- **PPI network**: the PPI network is downloaded from the STRING database (https://string-db.org/) by selecting the high confidence PPIs (score >= 700) for _H. sapiens_.

## REFERENCES

[1] G. Muzio, L. O’Bray, L. Meng-Papaxanthos, J. Klatt, K. Borgwardt, networkGWAS: A network-based approach for genome-wide association studies in structured populations, bioRxiv, https://doi.org/10.1101/2021.11.11.468206.

[2] C. Lippert, J. Xiang, D. Horta, C. Widmer, C. Kadie, D. Heckerman, J. Listgarten, Greater power and computational efficiency for kernel-based association testing of sets of genetic variants, Bioinformatics 30(22):3206–3214, 2014.

[3] J. Peter, M. De Chiara, A. Friedrich et al., Genome evolution across 1,011 Saccharomyces cerevisiae isolates. Nature 556, 339–344, 2018.

[4] A. Dimitromanolakis, J. Xu, A. Krol, L. Briollais, sim1000g: a user-friendly genetic variant simulator in R for unrelated individuals and family-based designs. BMC Bioinformatics 20(26), 2019.

## CONTACT
giulia.muzio@bsse.ethz.ch

## FUNDING
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 813533. This study was supported in part by the Alfried Krupp Prize for Young University Teachers of the Alfried Krupp von Bohlen und Halbach-Stiftung (Borgwardt).
