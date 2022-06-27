'''
Compared to the original implementation at
https://github.com/fastlmm/FaST-LMM/
this file has been modified by Giulia Muzio
'''
import io
import os
import stat
import time
import logging
import warnings
import scipy as sp
import numpy as np
import util.preprocess as util

from .Result import *
from pyplink.plink import *
from pyplink.altset_list.SnpAndSetNameCollection import SnpAndSetNameCollection
from pysnptools.util.pheno import *
from association.tests import *
from pyplink.snpreader.Bed import *




class FastLmmSet:
    '''
    A class for specifying a FastLmmSet and then running it.
    '''
    def __init__(self, **entries):
        '''
        outfile         : string of the filename to which to write out results (two files will be generated from this, on a record of the run)
        phenofile       : string of the filename containing phenotype
        alt_snpreader   : A snpreader for the SNPs for alternative kernel. If just a file name is given, the Bed reader (with standardization) is used.
        bedfilealt      : (deprecated) same as alt_snpreader
        filenull        : string of the filename (with .bed or .ped suffix) containing SNPs for null kernel. Should be Bed format if autoselect is true.
        extractSim      : string of a filename containing a list of whitespace-delmited SNPs. Only these SNPs will be used when filenull is read.
                               By default all SNPs in filenull are used.
                               It is an error to specify extractSim and not filenull.
                               Currently assumes that filenull is in bed format and that autoselect is not being used.
        altset_list     : list of the altsets
                               By default this is a file that will be read by SnpAndSetNameCollection
                               but a file of nuc ranges can be given again via 'NucRangeSet(filename)'
        covarfile       : string of the filename containing covariate
        mpheno          : integer representing the index of the testing phenotype (starting at 1)
        mindist         : SNPs within mindist from the alternative SNPs will be removed from
                          the null kernel computation
        idist           : the index of the position index to use with mindist
                            1 : genomic distance
                            2 : base-pair distance
        test            :'lrt' or 'lrt_up'
        nperm           : number of pemutations per test
        calseed         : the int that gets added to the random seed 34343 used to generate permutations for lrt null fitting.
        forcefullrank   : If true, the covariance is always computed for lrt (default False)
        log             : (Defaults to not changing logging level) Level of log messages, e.g. logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO
        '''
        # member variables with default values, or those not defined in the input file    
        self.mpheno = 1 
        #self.autoselect = True
        self.extractSim = None
        self.calseed = None # was 0 # the int that gets added to the random seed 34343 used to generate permutations for lrt null fitting
        self.sets = None
        self.forcefullrank = False
        self.covarimp = 'standardize' # use 'standardize' for mean and variance standardization, with mean imputation
        self.nullModel = None
        self.altModel = None
        self.scoring = None
        self.greater_is_better = None
        self.log = None
        self.detailed_table = False
        self.signal_ratio = True
        self._synthphenfile = None
        self.alt_snpreader = None
        self.kernel = None
        self.__dict__.update(entries)          
        self._ran_once = False
        
        if isinstance(self.alt_snpreader, str):
            self.alt_snpreader = Bed(self.alt_snpreader)

        # convert deprecated "verbose" into "log"        
        if self.hasNonNoneAttr("log") and self.hasNonNoneAttr("verbose") :
            raise Exception("log or verbose may be given, but not both")
        if self.hasNonNoneAttr("verbose"):
            if self.verbose:
                self.log = logging.INFO
            else:
                self.log = logging.CRITICAL
            delattr(self, "verbose")

        if self.hasNonNoneAttr("log"): # If neither were set
            logger = logging.getLogger()
            logger.setLevel(self.log)

        if not hasattr(self, "test") : raise Exception("FastLmmSet must have 'test' set")


    def __str__(self): # OK!
        '''
        This method is called when print() or str() function is invoked on an object.
        It should contain a readable description of the class.
        '''
        if self.outfile == None:
            return self.__class__.__name__
        else:
            return "{0} {1}".format(self.__class__.__name__, self.outfile)


    def __repr__(self): # OK!
        '''
        This method is the "official" string representation of our object.
        '''
        fp = io.StringIO() if sys.version_info >= (3,0) else io.BytesIO()
        fp.write("{0}(\n".format(self.__class__.__name__))
        varlist = []
        for f in dir(self):
            if f.startswith("_"): # remove items that start with '_'
                continue
            if type(self.__class__.__dict__.get(f,None)) is property: # remove @properties
                continue
            if callable(getattr(self, f)): # remove methods
                continue
            varlist.append(f)


        for var in varlist[:-1]: #all but last
            fp.write("\t{0} = {1},\n".format(var, getattr(self, var).__repr__()))
        

        var = varlist[-1] # last
        fp.write("\t{0} = {1})\n".format(var, getattr(self, var).__repr__()))
        result = fp.getvalue()
        fp.close()
        return result

    
    def hasNonNoneAttr(self, attr): # USED!!
        '''
        Function for checking it an attribute of the class exists 
        and if it's different from None
        '''
        return hasattr(self, attr) and getattr(self, attr) != None


    def work_sequence(self): # USING IT!
        '''
        This is the function where:
        1) the SNPs for the alternative kernel are loaded
        2) it's called the function for running the test

        OTHER DETAILS:
        Enumerates a sequence of work items, i.e. generator.
        Each work item is a lambda expression (i.e. function pointer) that calls 'run_test', returning a 
        list of Results (often just one)
        '''

        self.run_once() # load files, etc. -- stuff we only want to do once per task (e.g. on the cluster)
                        # no matter how many times we call 'run_test' (and then 'reduce' which of course
                        # only gets run one time, and calls this too, but doesn't actually do the work).
        ttt0 = time.time()
        y = None                                        
        for iset, altset in enumerate(self.altsetlist_filtbysnps):
            SNPsalt = altset.read()
            SNPsalt['snps'] = util.standardize(SNPsalt['snps'])
            G1 = SNPsalt['snps']
            
            # setting remaining parameters
            ichrm =  ",".join(sp.array(sp.unique(SNPsalt['pos'][:, 0]), dtype = str)) 
            
            minpos = str(np.min(SNPsalt['pos'][:, 2]))
            maxpos = str(sp.max(SNPsalt['pos'][:, 2]))
            
            iposrange = minpos + "-" + maxpos

            y = self.__y
            yield self.run_test(SNPs1 = SNPsalt, G1 = G1, y = y, altset = altset, iset = iset + 1, 
                             ichrm = ichrm, iposrange = iposrange, kernel = self.kernel)
   

        ttt1 = time.time()
        logging.info("---------------------------------------------------")
        logging.info("Elapsed time for all tests is %.2f seconds" % (ttt1 - ttt0))
        logging.info("---------------------------------------------------")


    def run_test(self, SNPs1, G1, y, altset, iset, ichrm, iposrange, kernel, iperm = -1, varcomp_test = None):
        '''
        This function does the main work of the class, and also reads in the SNPs for the alternative model.
        It is called (via a lambda) inside the loops found in 'generate_sequence'.

        Input:
            altset - a set of snps
            iset - index to altset
            iperm - index to permutation (-1 means no permutation)
            varcomp -if not None, assume that it is the correct one for this test, and does not re-compute anything
        Output:
            a list (often just one) of
                instances of the Result class, varcomp (for caching)


        Input
        -----------------------------------
        SNPs1:
        G1:
        altset:
        iset:            ID of the set
        ichrm:           ID of the chromosome
        iposrange:       position range of the SNPs in the set to test
        iperm:           permutation number
        varcomp_test:    object containing the model
        '''
        t0 = time.time()            

        self.run_once() #load files, etc. -- stuff we only want to do once no matter how many times we call 'run_test' and then 'reduce.        
        
        logging.info("\taltset {0}, {1} of {2}  ".format(altset, iset, len(self.altsetlist_filtbysnps)))
        
        result = Result(iperm = iperm, iset = iset, setname = str(altset), ichrm = ichrm, iposrange = iposrange)
   
            
        # 1) SET ANALYSIS
        if G1.shape[1] == 0:
            logging.info( "no SNPS in set " + setname )
            result = None
            return [result]
        if sp.isnan(G1.sum()): raise Exception("found missing values in test SNPs that remain after intersection for " + str(altset))

        # Calculation of the test size
        result.setsize = SNPs1['snps'].shape[1]
        logging.info(" (" + str(result.setsize) + " SNPs)")
        varcomp_test = self.__varcomp_test                
        
        # 4) RUNNING THE ACTUAL TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # <association.lrt.lrt at 0x7fb963c63440> is varcomp_test in case of NOT using the GSM
        # <association.tests.LRT_up.lrt at 0x7f69ac1020e0> is varcomp_test when using the GSM
        result.test = varcomp_test.testG(G1, self.kernel, self.test) # pass center            

        # HERE!!!!!
        logging.info("p=%.2e",result.test['pv'])
                                               
        t1 = time.time()
        logging.info("%.2f seconds elapsed in run_test" % (t1 - t0))
             
        return [result]


    @property # USED
    def mainseed(self):
        '''used to xor indexes with to get seeds for other purposes'''
        return 34343
       

    def run_once(self): # USED!!!
        '''
        Loading datasets (but not SNPs for alternative), etc. that we want to do only 
        once no mater how many times we call 'run_test'.        
        '''         
        # 1) IMPOSING THE SEED FOR THE REPRODUCIBILITY OF RESULTS
        self._seed = self.mainseed


        # 2) CHECKING IF THE DATASET HAS ALREADY BEEN READ
        # If so, then don't load anything
        if (self._ran_once):
            return
        

        self._ran_once = True # modify this parameter to True, meaning that we are now reading the files,
        # and if this function will be called again, then the files won't be read twice.
        logging.info("Running Once")


        # 3) CHECKING THE TEST TO PERFORM
        # Now it's imposing the correct file to use for calculating the LRT association scores
        if self.test == "lrt":
            self.test = Lrt() # CHECK THIS FILE 


        # 4) READING THE PHENOTYPE
        phenhead, phentail = os.path.split(self.phenofile)
        logging.info("Reading Pheno: " +  phentail + "")        
        pheno = loadPhen(filename = self.phenofile, missing = 'NaN')
        pheno['vals'] = pheno['vals'][:, self.mpheno - 1 ] # use -1 so compatible with C++ version
        goodind = sp.logical_not(sp.isnan(pheno['vals']))
        pheno['vals'] = pheno['vals'][goodind]
        pheno['iid']  = pheno['iid'][goodind,:]        


        # 6) COVARIATES CHECK & LOADING
        if self.covarfile == None:
            covar = None
        else: # if present, then reading them
            covar = loadPhen(self.covarfile,missing='NaN')
            if self.covarimp == 'standardize':
                covar['vals'], fracmissing = utilx.standardize_col(covar['vals'])
            elif self.covarimp is None:
                pass
            else:
                raise Exception("covarimp=" + self.covarimp + " not implemented")          

        # filtering the covariate to keep only the samples/individuals for which we have the 
        # phenotype value
        covar, pheno, indarr = self.intersect_data(covar, pheno)
        N = pheno['vals'].shape[0];
        if self.covarfile == None:
            self.__X = sp.ones((N, 1))
        else:
            # check for covariates which are constant, as the lmm code crashes on these            
            badind = utilx.indof_constfeatures(covar['vals'], axis = 0)
            if len(badind) > 0:
                raise Exception("found constant covariates with indexes: %s: please remove, or modify code here to do so" % badind)
            self.__X = sp.hstack((sp.ones((N, 1)), covar['vals']))
        

        self.__y = pheno['vals']

        if not covar is None and sp.isnan(covar['vals'].sum()): 
            raise Exception("found missing values in covariates file that remain after intersection")
        if sp.isnan(self.__y.sum()): 
            raise Exception("found missing values in phenotype file that remain after intersection")
        

        # creating sets from set defn files. Looks at bed file to filter down the SNPs to only those present in the bed file
        self.altset_list, self.altsetlist_filtbysnps = self.create_altsetlist_filtbysnps(self.altset_list, self.alt_snpreader)
        

        # 7) SHOWING THE INFO ABOUT THE INPUT DATA
        logging.info("------------------------------------------------")
        logging.info("Found " + str(self.__X.shape[0]) +  " individuals")
        logging.info("Found " + str(self.__X.shape[1]) +  " covariates (including bias)")
        if (len(self.__y.shape)>1): nPhen=self.__y.shape[1]
        else: nPhen = 1
        logging.info("Found " + str(nPhen) +  " phenotypes")
        
        logging.info("(running single set tests)")        

        if self.mpheno is not None: logging.info("mpheno=%i", self.mpheno)
        if self.extractSim is not None: logging.info("extractSim=%s",self.extractSim)
        logging.info("------------------------------------------------")

        logging.info("Creating null model... ")
        #cache whatever is needed for each case (sometimes likelihood, sometimes rotated items, sometimes nothing)
        t0 = time.time()
        
        # 8) ACTUALLY RUNNING THE PROCEDURE (here it's actually the null model)
        # for some models, this caches important and expensive information
        # as appropriate, gets re-computed in run_test()  
        self.__varcomp_test = self.varcomp_test_setup(self.__y)
        
        t1 = time.time()
        logging.info("done. %.2f seconds elapsed" % (t1 - t0))
        return 0


    def varcomp_test_setup(self, y): # USED!!!
        '''
        This function is used for contructing the model, in particular to see whether it
        is possible to include a background kernel (i.e. the GSM), besides the alternative 
        kernel (constructed from the SNPs to test).

        Input
        -----------------------------------
        y:             phenotype
        SNPs0:         SNPs to include in the GSM, as dictionary ('data', 'filename', 
                       'original_iids', 'num_snps')
        i_exclude:     the SNPs to exclude from the GSM computation. Those SNPs are the ones 
                       which didn't fulfill the criterion about the distance (idist, ...)

        Output
        -----------------------------------
        varcomp_test:  object which represents the model, with everything defined.
        '''
        # in case we exclude all SNPs (needed for score, not for lrt, not sure about lrt_up)
        # this first part is for checking if all the SNPs are excluded from the GSM
        nullModelTmp = self.nullModel.copy()           
        excluded_all_snps = False         
        
        varcomp_test = self.test.construct_no_backgound_kernel(y, self.__X,
                             forcefullrank = self.forcefullrank, nullModel = nullModelTmp,
                             altModel = self.altModel, scoring = self.scoring,
                             greater_is_better = self.greater_is_better)

        return varcomp_test


    def intersect_data(self, covar, pheno): # USED!!!
        '''
        This function intersects the covariates and the phenotype
        in order to have the same individuals, and to put them in 
        the same order.
        '''
        if self.check_id_order(covar, pheno): # if they match
            indarr = SP.arange(pheno['iid'].shape[0])
            logging.info(str(indarr.shape[0]) + " IDs match up across data sets")
        else:            
            logging.info("IDs do not match up, so intersecting the data over individuals")            
            
            nullsnpids = None
            if self.covarfile is not None:
                covarids = covar['iid']
            else:
                covarids = None
            if hasattr(self, "__varBackNullSnpsGen") and self.__varBackNullSnpsGen is not None:
                varbacknullsnp = self.__varBackNullSnpsGen['iid']
            else:
                varbacknullsnp = None

            # the order of inputs here is reflected in the reordering indexes below of 0,1,2,3,4
            indarr = utilx.intersect_ids([pheno['iid'], self.alt_snpreader.original_iids, covarids, nullsnpids, varbacknullsnp])            
            assert indarr.shape[0] > 0, "no individuals remain after intersection, check that ids match in files"
            # sort the indexes so that SNPs ids in their original order (and 
            # therefore we have to move things around in memory the least amount)
            # [indarr[:, 3] holds the SNP order    
            sortind = sp.argsort(indarr[:,3])
            indarr = indarr[sortind]

            pheno['iid'] = pheno['iid'][indarr[:, 0]]
            pheno['vals'] = pheno['vals'][indarr[:, 0]]

            self.alt_snpreader.ind_used = indarr[:, 1]

            if self.covarfile is not None:
                covar['iid'] = covar['iid'][indarr[:, 2]]
                covar['vals'] = covar['vals'][indarr[:, 2]]


            if hasattr(self,"_varBackNullSnpsGen") and self.__varBackNullSnpsGen is not None:
                self.__varBackNullSnpsGen['iid'] = self.__varBackNullSnpsGen['iid'][indarr[:, 4]]
                self.__varBackNullSnpsGen['snps'] = self.__varBackNullSnpsGen['snps'][indarr[:, 4]]

            
            logging.info(str(indarr.shape[0]) + " ids left")
        
        return covar, pheno, indarr


    def check_id_order(self, covar, pheno): # USED!!!
        '''
        Function for checking if the covariates and the phenotype
        have the same individuals ordering. Returns a boolean.
        '''
        ids_pheno = pheno['iid']
        ids_SNPs = self.alt_snpreader.original_iids
        if (len(ids_pheno) != len(ids_SNPs)) or (not sp.all(ids_pheno == ids_SNPs)):
          return False


        if self.covarfile is not None:
            ids_X = covar['iid']
            if (len(ids_pheno)!=len(ids_X)) or (not sp.all(ids_pheno == ids_X)): 
                return False


        if hasattr(self,"__varBackNullSnpsGen") and self.__varBackNullSnpsGen is not None: #for synthetic data generation            
            ids_SNPgen = self.__varBackNullSnpsGen["iid"]
            if not sp.all(ids_pheno == ids_SNPgen): 
                return False


        return True


    def create_altsetlist_filtbysnps(self, altset_list, altsetbed): # USED!!!
        '''
        This function allows to create an object containing the lists of names of SNPs belonging to 
        each set. The lists are contained in altset_list's attribute "bigToSmall". It's called "bigToSmall"
        because the sets are ordered according to the number of SNPs each of them contains, from the bigger
        to the smaller set (in terms of SNPs cardinality).

        If it's specified, then it's also applied a filter on the sets, i.e. via Subset function.

        returns: altset_list, altsetlist_filtbysnps
        'altset_list' is the raw set defintion read in from file
        'altsetlist_filtbysnps' is the result of filtering altset_list by SNPsin the specified altsetbed SNP file
        Additionally, only allows sets specified by sets are allowed (if this is not None)
        '''
        if isinstance(altset_list, str):
            altset_list = SnpAndSetNameCollection.SnpAndSetNameCollection(altset_list)
            # the function SnpAndSetNameCollection can be found in pyplink/altset_list

        if self.sets is not None:  #filter by those sets specified on the command line, stored in self.sets
            altset_list = Subset(altset_list, self.sets)


        altsetlist_filtbysnps = altset_list.addbed(altsetbed)
        
        if len(altsetlist_filtbysnps) == 0: raise Exception("Expect altset_list to contain at least one set")
        # the length of altsetlist_filtbysnps is the number of sets of SNPs.
        return altset_list, altsetlist_filtbysnps



    def reduce(self, result_list_sequence):
        '''
        Given a sequence of results from 'run_test', create the output report.
        '''
        
        self.run_once() #load files, etc. -- doesn't actually do the work if it's already been done
                         #note, however, that before reduce is called, work_count calls run_once() anyhow

        # results can come in any order, so we have to use iperm and iset to put them in the right place
        # there is one result instance for each combination of test and permutation, and here we are just gathering them
        # into the arrays from above
        
        #for result_list in result_list_sequence:
        observed_statistics = {}
        null_distribution   = {}
        for result in result_list_sequence:
            if result.iperm < 0: # observed statistics on the not permuted setting
                observed_statistics[result.setname] = self.test.lrt_method(result)
                null_distribution.update({result.setname: {}})              
            else:
                null_distribution[result.setname][result.iperm] = self.test.lrt_method(result)
            if result.alteqnull is None and str(self.test)[0:3]=="lrt":
                raise Exception("self.alteqnull is None")
        
        return observed_statistics, null_distribution



def CreateSnpSetReaderForFileNull(snp_set):
    if (snp_set is None):
        return AllSnps()
    if isinstance(snp_set, str):        
        return SnpsFromFile(snp_set)
    return snp_set