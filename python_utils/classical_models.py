import numpy as np
import pandas as pd
import random
import pickle
import gc
from gtfparse import read_gtf
import pyBigWig

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import seaborn as sns

from progressbar import ProgressBar, Bar, Percentage, AnimatedMarker, AdaptiveETA
from IPython.display import clear_output

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer

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

class Blank(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return X
    
    def inverse_transform(self, X, y = None):
        return X

class LogLinearTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, adult_age = 20):
        self.adult_age = adult_age

    def fit(self, target):
        return self

    def transform(self, target):
        target_ = target.copy().astype(float)
        for i in range(len(target_)):
            if target_[i] < self.adult_age:
                target_[i] = np.log((target_[i] + 1)/(self.adult_age + 1))
            else:
                target_[i] = (target_[i] - self.adult_age)/(1 + self.adult_age)
        return target_
    
    def inverse_transform(self, target):
        target_ = target.copy().astype(float)
        for i in range(len(target_)):
            if target_[i] < 0:
                target_[i] = (1 + self.adult_age)*(np.exp(target_[i]))-1
            else:
                target_[i] = (1 + self.adult_age)*target_[i] + self.adult_age
        return target_

class SigmoidTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_lifespan = 122, alpha = 1):
        self.max_lifespan = max_lifespan
        self.alpha = alpha
        self.gestation_time = 40*7/365

    def fit(self, target):
        return self

    def transform(self, target):
        target_ = target.copy().astype(float)
        target_ = target_ + self.gestation_time #make sure that all ages are positive
        target_ = target_/self.max_lifespan
        target_ = np.log(target_/(1-target_))/self.alpha
        return target_
    
    def inverse_transform(self, target):
        target_ = target.copy().astype(float)
        target_ = 1/(1 + np.exp(-target_*self.alpha))
        target_ = target_ * self.max_lifespan - self.gestation_time
        return target_

def filter_metadata(metadata, cancer = False, biological_replicates = False):
    
    #keep or remove cancer samples
    cancer_indexes = []
    for i in metadata[metadata.Description.notnull()].index:
        description = metadata.loc[i].Description
        if 'cancerous' in description or 'oma' in description:
            cancer_indexes.append(i)  
    if cancer == True: 
        metadata = metadata.loc[cancer_indexes]
    else:
        metadata = metadata.drop(cancer_indexes)
    
    #keep or remove biological replicates
    biological_replicate_experiments = metadata.groupby(['Experiment accession']).count()[metadata.groupby(['Experiment accession']).count()['Biological replicate(s)']>2].index
    if biological_replicates == True:
        metadata = metadata[metadata['Experiment accession'].isin(biological_replicate_experiments)]
    else:
        metadata = metadata[~metadata['Experiment accession'].isin(biological_replicate_experiments)]
    
    return metadata

def validate_classical_models(histone, organism, data_type, model_list, scaler_list, age_transform_list, folds = 5):
    
    #summary
    summary_df = pd.DataFrame(columns = ['Histone', 'Function', 'Scaler', 'Age Transform', 'Model', 'Mean MAE', 'Std MAE', 'Mean MSE', 'Std MSE', 'Mean R2', 'Std R2'])
    
    #get file names for all the compressed histone_data_objects
    directory = '/users/masif/data/masif/ChromAge/encode_histone_data/' + organism + '/' + data_type + '/' + histone +'/processed_data/'
    histone_files = [f for f in listdir(directory) if isfile(join(directory, f))]
    histone_files = [f for f in histone_files if histone in f]
    
    #load metadata without the cancer samples
    metadata = pd.read_pickle('/users/masif/data/masif/ChromAge/encode_histone_data/' + organism + '/' + data_type + '/metadata_summary.pkl') 
    metadata = filter_metadata(metadata)
    
    #code to get a progress bar
    widgets = ['Progress: ', Percentage(), '[', Bar(marker=AnimatedMarker()), ']', ' ', AdaptiveETA(), ' ']
    pbar_maxval = len(histone_files)*len(model_list)*len(scaler_list)*len(age_transform_list)
    pbar = ProgressBar(widgets=widgets, maxval = pbar_maxval).start()
    count = 0
    
    #X-fold CV
    cv = KFold(n_splits=folds, random_state=42, shuffle=True)
    
    for file in histone_files:
        histone_data_object = pickle.load(open(directory + file, 'rb'))
        
        #ensures both X and y have same samples
        X = histone_data_object.df
        samples = np.intersect1d(metadata.index, X.index)
        X = X.loc[samples]
        y = metadata.loc[X.index].age
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)        
        
        #loop through the different age transformers
        for age_transform in age_transform_list:

            #loop through the different scaling methods
            for scaler in scaler_list:

                #loop through the models doing a 10-fold CV
                for model_name in model_list:

                    pipeline = Pipeline(steps = [('imputer', KNNImputer()), scaler, model_name])
                    model = TransformedTargetRegressor(regressor = pipeline, transformer = age_transform[1], check_inverse = False)
                    results = cross_validate(model, X_train, y_train, cv = cv, scoring = {'mae':'neg_median_absolute_error', 'mse':'neg_mean_squared_error', 'r2':'r2'})
                    
                    #evaluation metrics
                    mae = np.mean(np.abs(results['test_mae']))
                    std_mae = np.std(np.abs(results['test_mae']))
                    mse = np.mean(np.abs(results['test_mse']))
                    std_mse = np.std(np.abs(results['test_mse']))
                    r2 = np.mean(results['test_r2'])
                    std_r2 = np.std(results['test_r2'])
                    
                    #update progress bar
                    clear_output()
                    pbar.update(count+1)
                    count+=1
                    
                    #add all the results to one row of the 
                    summary_row =  pd.Series([histone, histone_data_object.function, scaler[0], age_transform[0], model_name[0], mae, std_mae, mse, std_mse, r2, std_r2], index = summary_df.columns)
                    summary_df = summary_df.append(summary_row, ignore_index = True)

    #stop progress bar
    pbar.finish()

    return summary_df

def plot(histone_mark):
    a = histone_mark

    df = a
    df.Function = df.Function.apply(lambda x: str(x)[10:14])
    fig, axs = plt.subplots(1,4, tight_layout = True, figsize=(20,5))
    axs[0].set_title('Validation MSE')
    sns.boxplot(data = df, x = 'Model', y = 'Mean MSE', ax = axs[0], showfliers = False)
    axs[1].set_title('Validation MSE')
    sns.boxplot(data = df, x = 'Age Transform', y = 'Mean MSE', ax = axs[1], showfliers = False)
    axs[2].set_title('Validation MSE')
    sns.boxplot(data = df, x = 'Function', y = 'Mean MSE', ax = axs[2], showfliers = False)
    axs[3].set_title('Validation MSE')
    sns.boxplot(data = df, x = 'Scaler', y = 'Mean MSE', ax = axs[3], showfliers = False)
    fig.show()

model_list = [
    ('elastic_net', ElasticNetCV(n_alphas = 10, max_iter=1000, random_state = 42)),
    ('svr', SVR()),
    ('knn', KNeighborsRegressor()),
    ('random_forest', RandomForestRegressor(random_state = 42)),
    ('gbr', GradientBoostingRegressor(random_state = 42)),
]

scaler_list = [
    ('no', Blank()),
    ('standard', StandardScaler()),
    ('robust', RobustScaler()),
    ('quantile', QuantileTransformer(output_distribution='normal', random_state=42)),
]

age_transform_list = [
    ('no', Blank()),
    ('loglinear', LogLinearTransformer()),
]

results_H3K4me3 = validate_classical_models('H3K4me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)
results_H3K4me3.to_csv("classic_results_H3K4me3.csv")

results_H3K27ac = validate_classical_models('H3K27ac', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)
results_H3K27ac.to_csv("classic_results_H3K27ac.csv")

results_H3K4me1 = validate_classical_models('H3K4me1', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)
results_H3K4me1.to_csv("classic_results_H3K4me1.csv")

results_H3K9me3 = validate_classical_models('H3K9me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)
results_H3K9me3.to_csv("classic_results_H3K9me3.csv")

results_H3K27me3 = validate_classical_models('H3K27me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)
results_H3K27me3.to_csv("classic_results_H3K27me3.csv")

results_H3K36me3 = validate_classical_models('H3K36me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)
results_H3K36me3.to_csv("classic_results_H3K36me3.csv")

# plot(results_H3K36me3)