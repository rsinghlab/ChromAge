#load required packages
import pyBigWig
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from os import listdir
from os.path import isfile, join
from IPython.display import clear_output
from progressbar import ProgressBar, Bar, Percentage, AnimatedMarker, AdaptiveETA
import re
import math
import pickle
from scipy.stats import lognorm
from gtfparse import read_gtf
import gc

#hg38 numbers
chrom_info_hg38 = {
    'chr1':248956422,
    'chr2':242193529,
    'chr3':198295559,
    'chr4':190214555,
    'chr5':181538259,
    'chr6':170805979,
    'chr7':159345973,
    'chr8':145138636,
    'chr9':138394717,
    'chr10':133797422,
    'chr11':135086622,
    'chr12':133275309,
    'chr13':114364328,
    'chr14':107043718,
    'chr15':101991189,
    'chr16':90338345,
    'chr17':83257441,
    'chr18':80373285,
    'chr19':58617616,
    'chr20':64444167,
    'chr21':46709983,
    'chr22':50818468,
    'chrX':156040895
}

#mm10 numbers
chrom_info_mm10 = {
    'chr1':195471971,
    'chr2':82113224,
    'chr3':160039680,
    'chr4':156508116,
    'chr5':151834684,
    'chr6':149736546,
    'chr7':145441459,
    'chr8':129401213,
    'chr9':124595110,
    'chr10':130694993,
    'chr11':122082543,
    'chr12':120129022,
    'chr13':120421639,
    'chr14':124902244,
    'chr15':104043685,
    'chr16':98207768,
    'chr17':94987271,
    'chr18':90702639,
    'chr19':61431566,
    'chrX':171031299,
}

class histone_data:
    
#------------------------------------------------------------------------------------------------------
    def __init__(self, histone, organism, data_type, chrom_info):
        self.histone = histone                 #name of the histone mark
        self.organism = organism               #human or mouse
        self.data_type = data_type             #type of data (tissue, cell line, primary cell, etc.)
        self.chrom_info = chrom_info           #dictionary with the number of bps per chromosome
        self.features = None                   #df with names of features (eg. "chr1:123132-123460") 
        self.evolution_cycle = 0               #evolution cycle for feature augmentation
        self.df = None                         #df for the main compressed fold enrichment data
        self.file_names = None                 #bigWig files
        self.corrupted_files = []              #bigWig files that raise an error when opening
        self.window_size = None                #window size for compression
        self.function = None                   #function used to compress data
        self.max_resolution = None             #max resolution for greedy zoom feature augmentation
        self.gene_annotation = None            #gene annotation for each transcript

#------------------------------------------------------------------------------------------------------
    def add_file_names(self):
        directory = 'histone_data/' + self.organism + '/' + self.data_type + '/' + self.histone + '/raw_data/'
        histone_files = [f for f in listdir(directory) if isfile(join(directory, f))]
        histone_files.sort()
        histone_files.pop(0)
        histone_files.pop(-1)
        self.directory = directory
        self.file_names = histone_files
        
#------------------------------------------------------------------------------------------------------
    def check_files(self, verbose = False): #makes sure that files are not corrupted, and if so, removes from file_names
                
        #loop through all the files
        for file in self.file_names:
            try: 
                bw = pyBigWig.open(self.directory + file)
                #loop through all chromosomes to check if each one can be opened
                for chrom in list(self.chrom_info.keys()):
                    chrom_bases = bw.values(chrom, 0, 42, numpy = True) 
            
            except: 
                self.file_names.remove(file)
                self.corrupted_files.append(file)
                if verbose == True:
                    print(file)
                continue

            bw.close()

#------------------------------------------------------------------------------------------------------
    def subdivide(self, by = 'bin', window_size = 100000, gene_filter = None): #make the segments for each feature
        
        if by == 'gene':
        
            self.gene_annotation = read_gtf('histone_metadata/' + self.organism + '/annotation/gene_annotation.gtf')
            self.gene_annotation = self.gene_annotation[self.gene_annotation["feature"] == "gene"]
            self.gene_annotation = self.gene_annotation[self.gene_annotation.seqname.apply(lambda x: x in self.chrom_info.keys())]
            if gene_filter != None:
                self.gene_annotation = gene_filter(self.gene_annotation)

            chrom = np.array(self.gene_annotation.seqname)
            start = np.array(self.gene_annotation.start)
            end = np.array(self.gene_annotation.end)
            length = end - start
            former_na = [0]*self.gene_annotation.shape[0]
            zero_masked = [0]*self.gene_annotation.shape[0]

            self.features = pd.DataFrame(np.array([chrom, start, end, length, former_na, zero_masked]).T, columns = ['chrom', 'start', 'end', 'length', 'former_na', 'zero_masked'], index = np.array(self.gene_annotation.gene_id))
            self.features[['start', 'end', 'length', 'former_na', 'zero_masked']] = self.features[['start', 'end', 'length', 'former_na', 'zero_masked']].apply(pd.to_numeric, axis = 1)
            self.features = self.features.sort_values(['chrom', 'start', 'end'])
            
        elif by == 'bin':
            
            self.window_size = int(window_size)
            self.features = np.empty([0,7])

            #loop through all chromosomes but chrY (as some samples are from woman)
            for chrom in list(self.chrom_info.keys()):

                #slide across the chromosome to get the feature names and positions
                bases_to_end = self.chrom_info[chrom]
                while bases_to_end > 0:

                    start = self.chrom_info[chrom] - bases_to_end
                    end = start + self.window_size if bases_to_end > self.window_size else self.chrom_info[chrom]
                    length = end - start
                    index = chrom + ':' + str(start + 1) + '-' + str(end)

                    self.features = np.vstack([self.features, [index, chrom, start + 1, end, length, 0, 0]])

                    bases_to_end -= self.window_size

            self.features = pd.DataFrame(self.features, columns = ['', 'chrom', 'start', 'end', 'length', 'former_na', 'zero_masked'])
            self.features[['start', 'end', 'length', 'former_na', 'zero_masked']] = self.features[['start', 'end', 'length', 'former_na', 'zero_masked']].apply(pd.to_numeric, axis = 1)
            self.features = self.features.set_index('')
        
        #empty df for the main compressed fold enrichment data
        self.df = np.empty([0,self.features.shape[0]])
        
#------------------------------------------------------------------------------------------------------
    def process(self, function): #compresses bigWig data with a function
        
        #function used to compress the data
        self.function = function              
        
        #code to get a progress bar
        widgets = ['Progress for ' + self.histone + ':', Percentage(), '[', Bar(marker=AnimatedMarker()), ']', ' ', AdaptiveETA(), ' ']
        pbar_maxval = len(self.file_names) * len(self.chrom_info.keys())
        pbar = ProgressBar(widgets=widgets, maxval = pbar_maxval).start()
        count = 0
        
        #loop through all the files
        for file in self.file_names:
            
            #first open file and create empty array to store all compressed variables across samples
            bw = pyBigWig.open(self.directory + file)
            all_vars = np.empty([0,])
            
            #loop through all chromosomes but chrY (as some samples are from woman)
            for chrom in list(self.chrom_info.keys()):
                
                #load entire chromosome ChIP-Seq values
                chrom_bases = bw.values(chrom, 0, self.chrom_info[chrom], numpy = True)
                
                #beginning and end of chromosome are zero for the interpolation below
                na_indexes = np.isnan(chrom_bases)
                if np.isnan(chrom_bases[0]):
                    chrom_bases[0] = 0
                if np.isnan(chrom_bases[-1]):
                    chrom_bases[-1] = 0
                    
                #Linear Interpolation for imputation
                chrom_bases = np.array(pd.Series(chrom_bases).interpolate(method='linear'))
                
                #due to the interpolation, it is possible that values are really small or negative. Just set them to 0
                zero_mask = np.array(chrom_bases < 0.01) & np.array(chrom_bases > 0.00)
                chrom_bases[zero_mask] = 0
                
                #features by chromossome
                features_chrom = self.features[self.features.chrom == chrom]
                
                #slide across the chromosome compressing the original bigWig file
                for index in features_chrom.index:
                    start = features_chrom.loc[index].start
                    end = features_chrom.loc[index].end
                    bin_bases = chrom_bases[start-1:end]
                    var = self.function(bin_bases)
                    all_vars = np.append(all_vars, var)
                                        
                    if np.sum(np.array(na_indexes[start:end])) > 0:
                        self.features.loc[np.array(self.features.chrom == chrom) & np.array(self.features.start == start), 'former_na'] += 1
                    if np.sum(np.array(zero_mask[start:end])) > 0:
                        self.features.loc[np.array(self.features.chrom == chrom) & np.array(self.features.start == start), 'zero_masked'] += 1
                    
                #update progress bar
                pbar.update(count+1)
                count+=1
                
                #collect garbage files
                gc.collect()
            
            bw.close()
                
            self.df = np.vstack([self.df, all_vars])
        
        #create pandas dataframe with indexes as file accession names
        self.df = pd.DataFrame(self.df, index = [feature_name[0:11] for feature_name in self.file_names], columns = self.features.index)
        
        #stop progress bar
        pbar.finish()
        
#------------------------------------------------------------------------------------------------------
    def save(self, name):
        filehandler = open('histone_data/' + self.organism + '/' + self.data_type + '/' + self.histone + '/processed_data/' + name + '.pkl', 'wb') 
        pickle.dump(self, filehandler)

#------------------------------------------------------------------------------------------------------

#entropy calculation uses a zero-inflated log-normal distribution
def continuous_normal_pdf(x, mu = 0, sd = 1, zero = 0.7):
    all_zeros = np.where(x == 0)
    f = (1 - zero) * lognorm.pdf(x, s = 1/np.sqrt(6), loc= mu, scale=sd)
    f[all_zeros] = zero + (1 - zero) * lognorm.cdf(x[all_zeros], s = 0.7, loc= mu, scale=sd)
    return f

def my_entropy(bases):
    ps = continuous_normal_pdf(bases)
    shannon_entropy = -(ps * np.log(ps)).sum()
    norm_entropy = shannon_entropy/len(bases)
    return norm_entropy

# Human tissue
# mean bins
# mean genes
# mean entropy


# H3K4me3
H3K4me3_mean_bins = histone_data('H3K4me3', 'human', 'tissue', chrom_info_hg38)
H3K4me3_mean_bins.add_file_names()
H3K4me3_mean_bins.check_files(verbose = True)
H3K4me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K4me3_mean_bins.process(np.mean)
H3K4me3_mean_bins.save('H3K4me3_mean_bins')

H3K4me3_mean_genes = histone_data('H3K4me3', 'human', 'tissue', chrom_info_hg38)
H3K4me3_mean_genes.add_file_names()
H3K4me3_mean_genes.check_files(verbose = True)
H3K4me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K4me3_mean_genes.process(np.mean)
H3K4me3_mean_genes.save('H3K4me3_mean_genes')

H3K4me3_entropy_bins = histone_data('H3K4me3', 'human', 'tissue', chrom_info_hg38)
H3K4me3_entropy_bins.add_file_names()
H3K4me3_entropy_bins.check_files(verbose = True)
H3K4me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K4me3_entropy_bins.process(my_entropy)
H3K4me3_entropy_bins.save('H3K4me3_entropy_bins')


# H3K27ac
H3K27ac_mean_bins = histone_data('H3K27ac', 'human', 'tissue', chrom_info_hg38)
H3K27ac_mean_bins.add_file_names()
H3K27ac_mean_bins.check_files(verbose = True)
H3K27ac_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K27ac_mean_bins.process(np.mean)
H3K27ac_mean_bins.save('H3K27ac_mean_bins')

H3K27ac_mean_genes = histone_data('H3K27ac', 'human', 'tissue', chrom_info_hg38)
H3K27ac_mean_genes.add_file_names()
H3K27ac_mean_genes.check_files(verbose = True)
H3K27ac_mean_genes.subdivide(by = 'gene')
clear_output()
H3K27ac_mean_genes.process(np.mean)
H3K27ac_mean_genes.save('H3K27ac_mean_genes')

H3K27ac_entropy_bins = histone_data('H3K27ac','human', 'tissue', chrom_info_hg38)
H3K27ac_entropy_bins.add_file_names()
H3K27ac_entropy_bins.check_files(verbose = True)
H3K27ac_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K27ac_entropy_bins.process(my_entropy)
H3K27ac_entropy_bins.save('H3K27ac_entropy_bins')

# H3K4me1
H3K4me1_mean_bins = histone_data('H3K4me1', 'human', 'tissue', chrom_info_hg38)
H3K4me1_mean_bins.add_file_names()
H3K4me1_mean_bins.check_files(verbose = True)
H3K4me1_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K4me1_mean_bins.process(np.mean)
H3K4me1_mean_bins.save('H3K4me1_mean_bins')

H3K4me1_mean_genes = histone_data('H3K4me1', 'human', 'tissue', chrom_info_hg38)
H3K4me1_mean_genes.add_file_names()
H3K4me1_mean_genes.check_files(verbose = True)
H3K4me1_mean_genes.subdivide(by = 'gene')
clear_output()
H3K4me1_mean_genes.process(np.mean)
H3K4me1_mean_genes.save('H3K4me1_mean_genes')

H3K4me1_entropy = histone_data('H3K4me1','human', 'tissue', chrom_info_hg38)
H3K4me1_entropy.add_file_names()
H3K4me1_entropy.check_files(verbose = True)
H3K4me1_entropy.subdivide(by = 'bin', window_size = 100000)
H3K4me1_entropy.process(my_entropy)
H3K4me1_entropy.save('H3K4me1_entropy')

# H3K9me3
H3K9me3_mean_bins = histone_data('H3K9me3', 'human', 'tissue', chrom_info_hg38)
H3K9me3_mean_bins.add_file_names()
H3K9me3_mean_bins.check_files(verbose = True)
H3K9me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K9me3_mean_bins.process(np.mean)
H3K9me3_mean_bins.save('H3K9me3_mean_bins')

H3K9me3_mean_genes = histone_data('H3K9me3', 'human', 'tissue', chrom_info_hg38)
H3K9me3_mean_genes.add_file_names()
H3K9me3_mean_genes.check_files(verbose = True)
H3K9me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K9me3_mean_genes.process(np.mean)
H3K9me3_mean_genes.save('H3K9me3_mean_genes')

H3K9me3_entropy_bins = histone_data('H3K9me3','human', 'tissue', chrom_info_hg38)
H3K9me3_entropy_bins.add_file_names()
H3K9me3_entropy_bins.check_files(verbose = True)
H3K9me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K9me3_entropy_bins.process(my_entropy)
H3K9me3_entropy_bins.save('H3K9me3_entropy_bins')

# H3K27me3
H3K27me3_mean_bins = histone_data('H3K27me3', 'human', 'tissue', chrom_info_hg38)
H3K27me3_mean_bins.add_file_names()
H3K27me3_mean_bins.check_files(verbose = True)
H3K27me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K27me3_mean_bins.process(np.mean)
H3K27me3_mean_bins.save('H3K27me3_mean_bins')

H3K27me3_mean_genes = histone_data('H3K27me3', 'human', 'tissue', chrom_info_hg38)
H3K27me3_mean_genes.add_file_names()
H3K27me3_mean_genes.check_files(verbose = True)
H3K27me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K27me3_mean_genes.process(np.mean)
H3K27me3_mean_genes.save('H3K27me3_mean_genes')

H3K27me3_entropy_bins = histone_data('H3K27me3','human', 'tissue', chrom_info_hg38)
H3K27me3_entropy_bins.add_file_names()
H3K27me3_entropy_bins.check_files(verbose = True)
H3K27me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K27me3_entropy_bins.process(my_entropy)
H3K27me3_entropy_bins.save('H3K27me3_entropy_bins')

# H3K36me3
H3K36me3_mean_bins = histone_data('H3K36me3', 'human', 'tissue', chrom_info_hg38)
H3K36me3_mean_bins.add_file_names()
H3K36me3_mean_bins.check_files(verbose = True)
H3K36me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K36me3_mean_bins.process(np.mean)
H3K36me3_mean_bins.save('H3K36me3_mean_bins')

H3K36me3_mean_genes = histone_data('H3K36me3', 'human', 'tissue', chrom_info_hg38)
H3K36me3_mean_genes.add_file_names()
H3K36me3_mean_genes.check_files(verbose = True)
H3K36me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K36me3_mean_genes.process(np.mean)
H3K36me3_mean_genes.save('H3K36me3_mean_genes')

H3K36me3_entropy_bins = histone_data('H3K36me3','human', 'tissue', chrom_info_hg38)
H3K36me3_entropy_bins.add_file_names()
H3K36me3_entropy_bins.check_files(verbose = True)
H3K36me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K36me3_entropy_bins.process(my_entropy)
H3K36me3_entropy_bins.save('H3K36me3_entropy_bins')

# Human primary_cell
# mean bins
# mean genes
# mean entropy


# H3K4me3
H3K4me3_mean_bins = histone_data('H3K4me3', 'human', 'primary_cell', chrom_info_hg38)
H3K4me3_mean_bins.add_file_names()
H3K4me3_mean_bins.check_files(verbose = True)
H3K4me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K4me3_mean_bins.process(np.mean)
H3K4me3_mean_bins.save('H3K4me3_mean_bins')

H3K4me3_mean_genes = histone_data('H3K4me3', 'human', 'primary_cell', chrom_info_hg38)
H3K4me3_mean_genes.add_file_names()
H3K4me3_mean_genes.check_files(verbose = True)
H3K4me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K4me3_mean_genes.process(np.mean)
H3K4me3_mean_genes.save('H3K4me3_mean_genes')

H3K4me3_entropy_bins = histone_data('H3K4me3', 'human', 'primary_cell', chrom_info_hg38)
H3K4me3_entropy_bins.add_file_names()
H3K4me3_entropy_bins.check_files(verbose = True)
H3K4me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K4me3_entropy_bins.process(my_entropy)
H3K4me3_entropy_bins.save('H3K4me3_entropy_bins')


# H3K27ac
H3K27ac_mean_bins = histone_data('H3K27ac', 'human', 'primary_cell', chrom_info_hg38)
H3K27ac_mean_bins.add_file_names()
H3K27ac_mean_bins.check_files(verbose = True)
H3K27ac_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K27ac_mean_bins.process(np.mean)
H3K27ac_mean_bins.save('H3K27ac_mean_bins')

H3K27ac_mean_genes = histone_data('H3K27ac', 'human', 'primary_cell', chrom_info_hg38)
H3K27ac_mean_genes.add_file_names()
H3K27ac_mean_genes.check_files(verbose = True)
H3K27ac_mean_genes.subdivide(by = 'gene')
clear_output()
H3K27ac_mean_genes.process(np.mean)
H3K27ac_mean_genes.save('H3K27ac_mean_genes')

H3K27ac_entropy_bins = histone_data('H3K27ac','human', 'primary_cell', chrom_info_hg38)
H3K27ac_entropy_bins.add_file_names()
H3K27ac_entropy_bins.check_files(verbose = True)
H3K27ac_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K27ac_entropy_bins.process(my_entropy)
H3K27ac_entropy_bins.save('H3K27ac_entropy_bins')

# H3K4me1
H3K4me1_mean_bins = histone_data('H3K4me1', 'human', 'primary_cell', chrom_info_hg38)
H3K4me1_mean_bins.add_file_names()
H3K4me1_mean_bins.check_files(verbose = True)
H3K4me1_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K4me1_mean_bins.process(np.mean)
H3K4me1_mean_bins.save('H3K4me1_mean_bins')

H3K4me1_mean_genes = histone_data('H3K4me1', 'human', 'primary_cell', chrom_info_hg38)
H3K4me1_mean_genes.add_file_names()
H3K4me1_mean_genes.check_files(verbose = True)
H3K4me1_mean_genes.subdivide(by = 'gene')
clear_output()
H3K4me1_mean_genes.process(np.mean)
H3K4me1_mean_genes.save('H3K4me1_mean_genes')

H3K4me1_entropy = histone_data('H3K4me1','human', 'primary_cell', chrom_info_hg38)
H3K4me1_entropy.add_file_names()
H3K4me1_entropy.check_files(verbose = True)
H3K4me1_entropy.subdivide(by = 'bin', window_size = 100000)
H3K4me1_entropy.process(my_entropy)
H3K4me1_entropy.save('H3K4me1_entropy')

# H3K9me3
H3K9me3_mean_bins = histone_data('H3K9me3', 'human', 'primary_cell', chrom_info_hg38)
H3K9me3_mean_bins.add_file_names()
H3K9me3_mean_bins.check_files(verbose = True)
H3K9me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K9me3_mean_bins.process(np.mean)
H3K9me3_mean_bins.save('H3K9me3_mean_bins')

H3K9me3_mean_genes = histone_data('H3K9me3', 'human', 'primary_cell', chrom_info_hg38)
H3K9me3_mean_genes.add_file_names()
H3K9me3_mean_genes.check_files(verbose = True)
H3K9me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K9me3_mean_genes.process(np.mean)
H3K9me3_mean_genes.save('H3K9me3_mean_genes')

H3K9me3_entropy_bins = histone_data('H3K9me3','human', 'primary_cell', chrom_info_hg38)
H3K9me3_entropy_bins.add_file_names()
H3K9me3_entropy_bins.check_files(verbose = True)
H3K9me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K9me3_entropy_bins.process(my_entropy)
H3K9me3_entropy_bins.save('H3K9me3_entropy_bins')

# H3K27me3
H3K27me3_mean_bins = histone_data('H3K27me3', 'human', 'primary_cell', chrom_info_hg38)
H3K27me3_mean_bins.add_file_names()
H3K27me3_mean_bins.check_files(verbose = True)
H3K27me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K27me3_mean_bins.process(np.mean)
H3K27me3_mean_bins.save('H3K27me3_mean_bins')

H3K27me3_mean_genes = histone_data('H3K27me3', 'human', 'primary_cell', chrom_info_hg38)
H3K27me3_mean_genes.add_file_names()
H3K27me3_mean_genes.check_files(verbose = True)
H3K27me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K27me3_mean_genes.process(np.mean)
H3K27me3_mean_genes.save('H3K27me3_mean_genes')

H3K27me3_entropy_bins = histone_data('H3K27me3','human', 'primary_cell', chrom_info_hg38)
H3K27me3_entropy_bins.add_file_names()
H3K27me3_entropy_bins.check_files(verbose = True)
H3K27me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K27me3_entropy_bins.process(my_entropy)
H3K27me3_entropy_bins.save('H3K27me3_entropy_bins')

# H3K36me3
H3K36me3_mean_bins = histone_data('H3K36me3', 'human', 'primary_cell', chrom_info_hg38)
H3K36me3_mean_bins.add_file_names()
H3K36me3_mean_bins.check_files(verbose = True)
H3K36me3_mean_bins.subdivide(by = 'bin', window_size = 100000)
H3K36me3_mean_bins.process(np.mean)
H3K36me3_mean_bins.save('H3K36me3_mean_bins')

H3K36me3_mean_genes = histone_data('H3K36me3', 'human', 'primary_cell', chrom_info_hg38)
H3K36me3_mean_genes.add_file_names()
H3K36me3_mean_genes.check_files(verbose = True)
H3K36me3_mean_genes.subdivide(by = 'gene')
clear_output()
H3K36me3_mean_genes.process(np.mean)
H3K36me3_mean_genes.save('H3K36me3_mean_genes')

H3K36me3_entropy_bins = histone_data('H3K36me3','human', 'primary_cell', chrom_info_hg38)
H3K36me3_entropy_bins.add_file_names()
H3K36me3_entropy_bins.check_files(verbose = True)
H3K36me3_entropy_bins.subdivide(by = 'bin', window_size = 100000)
H3K36me3_entropy_bins.process(my_entropy)
H3K36me3_entropy_bins.save('H3K36me3_entropy_bins')









