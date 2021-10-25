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

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from histone_data import histone_data

#random seed for reproducibility
tf.random.set_seed(42)
random.seed(42)

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
    biological_replicate_experiments = metadata.groupby(['Experiment accession']).count()[metadata.groupby(['Experiment accession']).count()['Biological replicate(s)']>1].index
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
    print("meta shape: ", metadata.shape)
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

results_H3K27ac = validate_classical_models('H3K27ac', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)

results_H3K4me1 = validate_classical_models('H3K4me1', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)

results_H3K9me3 = validate_classical_models('H3K9me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)

results_H3K27me3 = validate_classical_models('H3K27me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)

results_H3K36me3 = validate_classical_models('H3K36me3', 'human', 'tissue', model_list, scaler_list, age_transform_list, folds = 4)

# plot(results_H3K36me3)