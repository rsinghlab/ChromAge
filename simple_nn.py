from collections import defaultdict
from sched import scheduler
import numpy as np
import pandas as pd
import random
import pickle
import gc
import ast
import json
import os
import re

from tensorflow.python.ops.gen_nn_ops import Selu

from gtfparse import read_gtf
import pyBigWig

from os import listdir
from os.path import isfile, join

from progressbar import ProgressBar, Bar, Percentage, AnimatedMarker, AdaptiveETA
from IPython.display import clear_output
import shap

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import regularizers, datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional,Conv1DTranspose, ActivityRegularization, Input, LSTM, ReLU, GRU, multiply, Lambda, PReLU, SimpleRNN, Dense, Activation, BatchNormalization, Conv2D, Conv1D, Flatten, LeakyReLU, Dropout, GaussianNoise, MaxPooling1D, Reshape, UpSampling1D
import tensorflow_probability as tfp
import keras.backend as K
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from keras_visualizer import visualizer

#random seed for reproducibility
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

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

def split_data(metadata, histone_data_object, split = 0.2, histone_str = None, GEO = False):
    X = histone_data_object.df
    # ####### GEO DATA PROCESSING
    if GEO:
        metadata = metadata.dropna(subset=[histone_str])

        metadata.loc[:,[histone_str]] = metadata[histone_str].apply(lambda x: re.search('SRR\d*',x)[0])
        metadata = metadata.dropna(subset=[histone_str])
        metadata = metadata.dropna(subset=["Age"])

        metadata_temp = metadata[metadata[histone_str].apply(lambda x: x in X.index)]

        y = metadata_temp["Age"]
        X = X.loc[metadata_temp[histone_str]]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = split, random_state = 42)
    
    else:
    ##### ENCODE DATA PROCESSING
        samples = np.intersect1d(metadata.index, X.index)

        metadata_temp = metadata.loc[samples, :]

        experiment_training, experiment_testing = train_test_split(metadata_temp.groupby(['Experiment accession']).count().index, test_size = split, random_state = 42)

        training_list = [i in experiment_training for i in np.array(metadata_temp['Experiment accession'])]
        training_metadata = metadata_temp.loc[training_list, :]

        X_train = X.loc[training_metadata.index]
        y_train = training_metadata.loc[X_train.index].age
        
        testing_list = [i in experiment_testing for i in np.array(metadata_temp['Experiment accession'])]
        testing_metadata = metadata_temp.loc[testing_list, :]

        X_test = X.loc[testing_metadata.index]
        y_test = testing_metadata.loc[X_test.index].age

    return X_train, X_test, y_train, y_test

def filter_metadata(metadata, cancer = False, biological_replicates = False):
    
    #keep or remove cancer samples
    cancer_indexes = []
    for i in metadata[metadata.Description.notnull()].index:
        description = metadata.loc[i].Description
        if 'cancerous' in description or 'oma' in description:
            cancer_indexes.append(i)  
    if cancer: 
        metadata = metadata.loc[cancer_indexes]
    else:
        metadata = metadata.drop(cancer_indexes)
    
    biological_replicate_experiments = metadata.groupby(['Experiment accession']).count()[metadata.groupby(['Experiment accession']).count()['Biological replicate(s)']>2].index

    if not(biological_replicates):
        metadata = metadata[~metadata['Experiment accession'].isin(biological_replicate_experiments)]
    
    return metadata

def k_cross_validate_model(metadata, histone_data_object, y_test, batch_size, epochs, model_type, model_params, latent_size, gaussian_noise, df, k = 4, data_transform = None, age_transform = None):
    metadata = metadata.drop(y_test.index)

    X = histone_data_object.df
    samples = np.intersect1d(metadata.index, X.index)
    metadata_temp = metadata.loc[samples, :]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    kfold_data = kf.split(metadata_temp.groupby(['Experiment accession']).count().index)
    val_metrics_array = []
    min_auto_encoder_train_mse_array = []
    min_auto_encoder_train_mae_array = []
    min_auto_encoder_val_mse_array = []
    min_auto_encoder_val_mae_array = []
    min_train_loss_array = []
    min_train_mse_array = []
    min_train_mae_array = []
    min_val_loss_array = []
    min_val_mse_array = []
    min_val_mae_array = []

    for train_index, val_index in kfold_data:
        
        experiment_training = metadata_temp.groupby(['Experiment accession']).count().index[train_index]
        experiment_val = metadata_temp.groupby(['Experiment accession']).count().index[val_index]
        
        training_list = [i in experiment_training for i in np.array(metadata_temp['Experiment accession'])]
        train_metadata = metadata_temp.loc[training_list, :]
        
        val_list = [i in experiment_val for i in np.array(metadata_temp['Experiment accession'])]
        val_metadata = metadata_temp.loc[val_list, :]

        training_x = X.loc[train_metadata.index]
        training_y = np.expand_dims(np.array(train_metadata.loc[training_x.index].age),1)
        training_x = np.array(training_x)

        validation_x = X.loc[val_metadata.index]
        validation_y = val_metadata.loc[validation_x.index].age

        validation_y_index = validation_y.index

        validation_x = np.array(validation_x)
        validation_y = np.expand_dims(np.array(validation_y),1)

        # Data + Age transform

        if data_transform == "scaler":
            transformer = StandardScaler()
            training_x = transformer.fit_transform(training_x)
            validation_x = transformer.fit_transform(validation_x)
        
        if data_transform == "robust":
            transformer = RobustScaler()
            training_x = transformer.fit_transform(training_x)
            validation_x = transformer.fit_transform(validation_x)
        
        if data_transform == "quantile":
            transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            training_x = transformer.fit_transform(training_x)
            validation_x = transformer.fit_transform(validation_x)
        
        if age_transform == "loglinear":
            train_age_transformer = LogLinearTransformer()
            train_age_transformer.fit(training_y)
            training_y = train_age_transformer.transform(training_y)
            val_age_transformer = LogLinearTransformer()
            val_age_transformer.fit(validation_y)
            validation_y = val_age_transformer.transform(validation_y) 

        auto_encoder = AutoEncoder(batch_size, latent_size, model_params[2], model_params[3], gaussian_noise)
        auto_encoder.compile(
        loss='mse',
        metrics=['mae'],
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_params[1]))
        scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=100, min_lr=0.00001)

        auto_history = auto_encoder.fit(
            training_x, 
            training_y, 
            epochs=300, #600 for H3K4me1
            batch_size=batch_size, 
            validation_data=(validation_x, validation_y),
            # verbose = 0,
            callbacks = [scheduler]
        )

        min_auto_encoder_train_mse_array.append(np.min(auto_history.history['loss']))
        min_auto_encoder_train_mae_array.append(np.min(auto_history.history['mae']))
        min_auto_encoder_val_mse_array.append(np.min(auto_history.history['val_loss']))
        min_auto_encoder_val_mae_array.append(np.min(auto_history.history['val_mae']))

        model = create_nn(latent_size, model_params[0], model_params[1], model_params[2], model_params[3])
        history = model.fit(auto_encoder.encoder(training_x),
            training_y,
            batch_size, 
            epochs,
            validation_data=(auto_encoder.encoder(validation_x), validation_y),
            # verbose = 0,
            callbacks = [scheduler]
        )

        min_train_loss_array.append(np.min(history.history['loss']))
        min_train_mse_array.append(np.min(history.history['mse']))
        min_train_mae_array.append(np.min(history.history['mae']))
        min_val_loss_array.append(np.min(history.history['val_loss']))

        results = model.evaluate(auto_encoder.encoder(validation_x), validation_y, batch_size) 
        val_metrics_array.append(results)

        prediction_distribution = model(auto_encoder.encoder(validation_x))
        predicted_age = prediction_distribution.mean().numpy().flatten()
        validation_y = np.squeeze(validation_y)

        if age_transform == "loglinear":
            validation_y = val_age_transformer.inverse_transform(validation_y)
            predicted_age = val_age_transformer.inverse_transform(predicted_age)
            mse = mean_squared_error(validation_y, predicted_age)
            mae = median_absolute_error(validation_y, predicted_age)
            min_val_mse_array.append(np.mean(mse))
            min_val_mae_array.append(np.mean(mae))
        else:
            min_val_mse_array.append(np.min(history.history['val_mse']))
            min_val_mae_array.append(np.min(history.history['val_mae']))
        
        type_arr = np.full(validation_y.shape, model_type)

        if df is None:
            df_dict = {"Actual Age": validation_y, "Predicted Mean Age": predicted_age, "Predicted Stddev": prediction_distribution.stddev().numpy().flatten(), "Model Type" : type_arr}
            df = pd.DataFrame(df_dict, index = validation_y_index)
        else:
            df_dict = {"Actual Age": validation_y, "Predicted Mean Age": predicted_age, "Predicted Stddev": prediction_distribution.stddev().numpy().flatten(), "Model Type" : type_arr}
            df2 = pd.DataFrame(df_dict, index = validation_y_index)
            df = df.append(df2)
        # print(df)
    return df, val_metrics_array, min_auto_encoder_train_mse_array, min_auto_encoder_train_mae_array, min_auto_encoder_val_mse_array, min_auto_encoder_val_mae_array, min_train_loss_array, min_train_mse_array, min_train_mae_array, min_val_loss_array, min_val_mse_array, min_val_mae_array

def loss_function(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

#create neural network with adjustable parameters
def create_nn(input_size, hidden_layers = 3, lr = 0.001, dropout = 0.1, coeff = 0.01):
    hidden_layer_sizes = []

    # hidden layer size
    for i in range(hidden_layers):
        hidden_layer_sizes.append(32)
    
    model = Sequential()

    model.add(Input(shape = (input_size,)))
    model.add(BatchNormalization())
    # model.add(ActivityRegularization(coeff, coeff))
    
    for i in range(hidden_layers):
        model.add(Dense(hidden_layer_sizes[i],
                  kernel_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff),
                  activity_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff)))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(hidden_layer_sizes[-1], activation='selu'))

    model.add(Dense(2))
    model.add(tfp.layers.DistributionLambda(
      lambda t: tfp.distributions.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse', 'mae'])    
    return model

#create neural network with adjustable parameters
def create_nn_shap(input_size, hidden_layers = 3, lr = 0.001, dropout = 0.1, coeff = 0.01):
    hidden_layer_sizes = []

    # hidden layer size
    for i in range(hidden_layers):
        hidden_layer_sizes.append(32)
    
    model = Sequential()

    model.add(Input(shape = (input_size,)))
    model.add(BatchNormalization())
    # model.add(ActivityRegularization(coeff, coeff))
    
    for i in range(hidden_layers):
        model.add(Dense(hidden_layer_sizes[i],
                  kernel_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff),
                  activity_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff)))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(hidden_layer_sizes[-1], activation='selu'))

    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse', 'mae'])    
    return model

class AutoEncoder(tf.keras.Model):
    def __init__(self, batch_size, latent_size, dropout_rate, coeff, gaussian_noise):
        super(AutoEncoder, self,).__init__()
        self.batch_size = batch_size ## 32
        # self.loss = tf.keras.losses.MeanSquaredError()
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.latent_size = latent_size # 1500
        self.hidden_dim = latent_size * 4 #6000
        self.dropout_rate = dropout_rate #0.2
        self.coeff = coeff #0.01
        self.encoder = Sequential([
            GaussianNoise(gaussian_noise),
            Dense(self.hidden_dim, activation='selu', activity_regularizer=tf.keras.regularizers.l1_l2(self.coeff, self.coeff)),
            Dropout(self.dropout_rate),
            Dense(self.hidden_dim/2, activation='selu', activity_regularizer=tf.keras.regularizers.l1_l2(self.coeff, self.coeff)),
            Dropout(self.dropout_rate),
            Dense(self.latent_size, activation='selu', activity_regularizer=tf.keras.regularizers.l1_l2(self.coeff, self.coeff)),
        ])
        self.decoder = Sequential([
            Dense(int(self.hidden_dim/2), activation='selu', activity_regularizer=tf.keras.regularizers.l1_l2(self.coeff, self.coeff)),
            Dropout(self.dropout_rate),
            Dense(self.hidden_dim, activation='selu', activity_regularizer=tf.keras.regularizers.l1_l2(self.coeff, self.coeff)),
            Dropout(self.dropout_rate),
            Dense(30321, activation=None)
        ])
    
    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        return tf.squeeze(self.decoder(encoder_output))

def analyze_metrics(file_path, histone_mark_str):
    metric_dict = defaultdict(list)
    with open(file_path, "r") as file:
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        for model_types in dictionary:
            for metrics in dictionary[model_types]:
                if (metrics != 'val_metrics'):
                    mean_metric = np.mean(dictionary[model_types][metrics])
                    metric_dict["mean"+metrics[3:]].append(mean_metric)
        df = pd.DataFrame(metric_dict, index = list(dictionary.keys()))
        df.to_csv(histone_mark_str + "/simple_nn_grid_metrics_" +  histone_mark_str + ".csv")

        # best_auto_train_mse_models = df.nsmallest(10, 'mean_train_auto_mse')
        # best_auto_train_mse_models.to_csv(histone_mark_str + "/best_auto_encoder_train_mse_models_" +  histone_mark_str + ".csv")
        # best_auto_val_mse_models = df.nsmallest(20, 'mean_val_auto_mse')
        # best_auto_val_mse_models.to_csv(histone_mark_str + "/best_auto_encoder_val_mse_models_" +  histone_mark_str + ".csv")

        # best_auto_train_mae_models = df.nsmallest(10, 'mean_train_auto_mae')
        # best_auto_train_mae_models.to_csv(histone_mark_str + "/best_auto_encoder_train_mae_models_" +  histone_mark_str + ".csv")
        # best_auto_val_mae_models = df.nsmallest(20, 'mean_val_auto_mae')
        # best_auto_val_mae_models.to_csv(histone_mark_str + "/best_auto_encoder_val_mae_models_" +  histone_mark_str + ".csv")

        best_train_loss_models = df.nsmallest(10, 'mean_train_loss')
        best_train_loss_models.to_csv(histone_mark_str + "/best_train_loss_models_" +  histone_mark_str + ".csv")
        best_val_loss_models = df.nsmallest(20, 'mean_val_loss')
        best_val_loss_models.to_csv(histone_mark_str + "/best_val_loss_models_" +  histone_mark_str + ".csv")

        best_train_mse_models = df.nsmallest(10, 'mean_train_mse')
        best_train_mse_models.to_csv(histone_mark_str + "/best_train_mse_models_" +  histone_mark_str + ".csv")
        best_val_mse_models = df.nsmallest(20, 'mean_val_mse')
        best_val_mse_models.to_csv(histone_mark_str + "/best_val_mse_models_" +  histone_mark_str + ".csv")

        best_train_mae_models = df.nsmallest(10, 'mean_train_mae')
        best_train_mae_models.to_csv(histone_mark_str + "/best_train_mae_models_" +  histone_mark_str + ".csv")
        best_val_mae_models = df.nsmallest(20, 'mean_val_mae')
        best_val_mae_models.to_csv(histone_mark_str + "/best_val_mae_models_" +  histone_mark_str + ".csv")
        
        best_auto_train_models = set()
        best_auto_val_models = set()
        best_val_models = set()
        best_train_models = set()
        
        # for model in best_auto_val_mse_models.index:
        #     if model in best_auto_val_mae_models.index:
        #         best_auto_val_models.add(model)
        # for model in best_auto_val_mae_models.index:
        #     if model in best_auto_val_mse_models.index:
        #         best_auto_val_models.add(model)
        
        # for model in best_auto_train_mse_models.index:
        #     if model in best_auto_train_mae_models.index:
        #         best_auto_train_models.add(model)
        # for model in best_auto_train_mae_models.index:
        #     if model in best_auto_train_mse_models.index:
        #         best_auto_train_models.add(model)

        for model in best_val_loss_models.index:
            if model in best_val_mse_models.index or model in best_val_mae_models.index:
                best_val_models.add(model)
        for model in best_val_mse_models.index:
            if model in best_val_loss_models.index or model in best_val_mae_models.index:
                best_val_models.add(model)
        for model in best_val_mae_models.index:
            if model in best_val_loss_models.index or model in best_val_mse_models.index:
                best_val_models.add(model)
        
        for model in best_train_loss_models.index:
            if model in best_train_mse_models.index and model in best_train_mae_models.index:
                best_train_models.add(model)
        for model in best_train_mse_models.index:
            if model in best_train_loss_models.index and model in best_train_mae_models.index:
                best_train_models.add(model)
        for model in best_train_mae_models.index:
            if model in best_train_mse_models.index and model in best_train_mae_models.index:
                best_train_models.add(model)
    
    return list(best_auto_val_models), list(best_auto_train_models), list(best_val_models), list(best_train_models)

def run_grid_search(metadata, histone_data_object, param_grid):
    X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object)
    df = None
    metrics_dict = dict()
    for epoch in param_grid['epochs']:
        for batch in param_grid['batch_size']:
            for hidden_layers in param_grid['hidden_layers']:
                for lr in param_grid['lr']:
                    for dropout in param_grid['dropout']:
                        for coeff in param_grid['coeff']:
                            for latent_size in param_grid['latent_size']:
                                for gn in param_grid['gaussian_noise']:
                                    model_params = [hidden_layers, lr, dropout, coeff]
                                    str_model_params = [str(param) for param in model_params]
                                    model_name = "simple_nn " + str(batch) +" "+" ".join(str_model_params) + " " + str(latent_size) + " " + str(gn)
                                    df, val_metrics_array, min_auto_encoder_train_mse_array, min_auto_encoder_train_mae_array, min_auto_encoder_val_mse_array, min_auto_encoder_val_mae_array,  min_train_loss_array, min_train_mse_array, min_train_mae_array, min_val_loss_array, min_val_mse_array, min_val_mae_array = k_cross_validate_model(metadata, histone_data_object, y_test, batch, epoch, model_name, model_params, latent_size, gn, df)
                                    metrics_dict[model_name] = dict({"val_metrics" : val_metrics_array, "min_train_auto_mse": min_auto_encoder_train_mse_array, "min_val_auto_mse": min_auto_encoder_val_mse_array, "min_train_auto_mae": min_auto_encoder_train_mae_array, "min_val_auto_mae": min_auto_encoder_val_mae_array, "min_train_loss" : min_train_loss_array, "min_val_loss" : min_val_loss_array, "min_train_mse" : min_train_mse_array, "min_val_mse" : min_val_mse_array, "min_train_mae" : min_train_mae_array, "min_val_mae" : min_val_mae_array})
                                    print("run for model " + model_name)
                                    print(metrics_dict)
    return df, metrics_dict

def post_process(metadata, histone_data_object, histone_mark_str, X_train, X_test, y_train, y_test):
    
    # best_auto_val_models, best_auto_train_models, best_val_models, best_train_models = analyze_metrics(os.getcwd() + "/" + histone_mark_str + "/metrics-output-" + histone_mark_str + ".txt", histone_mark_str)

    # print("Best auto val models:", *list(best_auto_val_models), sep='\n')
    # print("Best auto train models:", *list(best_auto_train_models), sep='\n')
    # print("Best val models:", *list(best_val_models), sep='\n')
    # print("Best train models:", *list(best_train_models), sep='\n')

    # scaler_list = ["standard", "robust", "quantile"]
    # age_transform_list = ["loglinear"]

    # df, val_metrics_array, min_auto_encoder_train_mse_array, min_auto_encoder_train_mae_array, min_auto_encoder_val_mse_array, min_auto_encoder_val_mae_array,  min_train_loss_array, min_train_mse_array, min_train_mae_array, min_val_loss_array, min_val_mse_array, min_val_mae_array = k_cross_validate_model(metadata, histone_data_object, y_test, 16, 1000, "simple_nn 16 3 0.0002 0.05 0.1 150 0.2", [3, 0.0002, 0.05, 0.1], 50, 0.2, None, data_transform=None, age_transform=None)

    # print("Dataframe: ", df, "\n Val-metrics array:", val_metrics_array, "\n Mean-min-autoencoder-train-MSE:", np.mean(min_auto_encoder_train_mse_array), "\n Mean-Min-autoencoder-train-MAE:", np.mean(min_auto_encoder_train_mae_array), "\n Mean-Min-autoencoder-val-MSE:", np.mean(min_auto_encoder_val_mse_array), "\n Mean-Min-autoencoder-val-MAE:", np.mean(min_auto_encoder_val_mae_array),  "\n Mean-Min-train-loss:", np.mean(min_train_loss_array), "\n Mean-Min-train-mse:", np.mean(min_train_mse_array), "\n Mean-Min-train-mae:", np.mean(min_train_mae_array), "\n Mean-Min-val-loss:", np.mean(min_val_loss_array), "\n Mean-val-mse:", np.mean(min_val_mse_array), "\n Mean-val-mae:", np.mean(min_val_mae_array))
    # df.to_csv("Model_Results_" +  histone_mark_str + ".csv")
    test_model(X_train, X_test, y_train, y_test, histone_mark_str)

    # if histone_mark_str == "H3K4me3":
    #     model = create_nn_shap(30321, 5, 0.0003, 0.0, 0.01) # Best Model: simple_nn 16 5 0.0003 0.0 0.01 50 0.1
    # elif histone_mark_str == "H3K27ac":
    #     model = create_nn_shap(30321, 5, 0.0003, 0.0, 0.01) # Best Model: simple_nn 16 3 0.0002 0.05 0.1 150 0.2 / simple_nn 16 3 0.0003 0.0 0.1 50 0.2 / simple_nn 16 5 0.0003 0.0 0.01 50 0.1
    # elif histone_mark_str == "H3K27me3":
    #     model = create_nn_shap(30321, 3, 0.0003, 0.0, 0.1) # Best Model: simple_nn 16 3 0.0003 0.0 0.1 300 0.1
    # elif histone_mark_str == "H3K36me3":
    #     model = create_nn_shap(30321, 3, 0.0003, 0.0, 0.1) # Best Model: simple_nn 16 3 0.0003 0.0 0.1 50 0.1
    # elif histone_mark_str == "H3K4me1":
    #     model = create_nn_shap(30321, 3, 0.0003, 0.0, 0.01) # Best Model: simple_nn 16 3 0.0003 0.0 0.01 50 0.2 / simple_nn 16 5 0.0002 0.1 0.05 50 0.1
    # elif histone_mark_str == "H3K9me3":
    #     model = create_nn_shap(30321, 3, 0.0001, 0.0, 0.05) # Best Model: simple_nn 16 3 0.0001 0.0 0.05 50 0.1

    # get_shap_values(model, X_train, X_test, histone_mark_str)


    # model = ElasticNet(max_iter=1000, random_state = 42)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # mse = mean_squared_error(y_test, predictions)
    # mae = median_absolute_error(y_test, predictions)
    # corr, _ = pearsonr(y_test, predictions)
    # print("ELASTIC NET")
    # print('Pearsons correlation: %.3f' % corr)
    # print("Mean Median AE: ", mae, "\n Mean MSE:", mse)
    # df_dict = {"Actual Age": y_test, "Predicted Mean Age": predictions}
    # df = pd.DataFrame(df_dict)
    # df.to_csv('/gpfs/data/rsingh47/masif/ChromAge/ElasticNet-' + histone_mark_str + '_results.csv')
    
    return

def test_model(X_train, X_test, y_train, y_test, histone_mark_str, data_transform = None, age_transform = None):
    # Testing
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.expand_dims(np.array(y_train),1)
    y_test = np.expand_dims(np.array(y_test),1)

    if data_transform == "scaler":
        transformer = StandardScaler()
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.fit_transform(X_test)
        
    if data_transform == "robust":
        transformer = RobustScaler()
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.fit_transform(X_test)
        
    if data_transform == "quantile":
        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.fit_transform(X_test)
        
    if age_transform == "loglinear":
        train_age_transformer = LogLinearTransformer()
        train_age_transformer.fit(y_train)
        y_train = train_age_transformer.transform(y_train)
        test_age_transformer = LogLinearTransformer()
        test_age_transformer.fit(y_test)
        y_test = test_age_transformer.transform(y_test) 

    # pca = PCA(n_components=len(X_train))
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)

    # model = create_nn(300, 3, 0.0003, 0.0, 0.1)
    # auto_encoder_args = [16, 300, 0.0, 0.1, 0.1, 0.0003]

    if histone_mark_str == "H3K4me3":
        model = create_nn(50, 5, 0.0003, 0.0, 0.01) # Best Model: simple_nn 16 5 0.0003 0.0 0.01 50 0.1
        auto_encoder_args = [16, 50, 0.0, 0.01, 0.1, 0.0003]
    elif histone_mark_str == "H3K27ac":
        model = create_nn(50, 5, 0.0003, 0.0, 0.01) # Best Model: simple_nn 16 3 0.0002 0.05 0.1 150 0.2 / simple_nn 16 3 0.0003 0.0 0.1 50 0.2 / simple_nn 16 5 0.0003 0.0 0.01 50 0.1
        auto_encoder_args = [16, 50, 0.0, 0.01, 0.1, 0.0003]
    elif histone_mark_str == "H3K27me3":
        model = create_nn(300, 3, 0.0003, 0.0, 0.1) # Best Model: simple_nn 16 3 0.0003 0.0 0.1 300 0.1
        auto_encoder_args = [16, 300, 0.0, 0.1, 0.1, 0.0003]
    elif histone_mark_str == "H3K36me3":
        model = create_nn(50, 3, 0.0003, 0.0, 0.1) # Best Model: simple_nn 16 3 0.0003 0.0 0.1 50 0.1
        auto_encoder_args = [16, 50, 0.0, 0.1, 0.1, 0.0003]
    elif histone_mark_str == "H3K4me1":
        model = create_nn(50, 3, 0.0003, 0.0, 0.01) # Best Model: simple_nn 16 3 0.0003 0.0 0.01 50 0.2 / simple_nn 16 5 0.0002 0.1 0.05 50 0.1
        auto_encoder_args = [16, 50, 0.0, 0.01, 0.2, 0.0003]
    elif histone_mark_str == "H3K9me3":
        model = create_nn(50, 3, 0.0001, 0.0, 0.05) # Best Model: simple_nn 16 3 0.0001 0.0 0.05 50 0.1
        auto_encoder_args = [16, 50, 0.0, 0.05, 0.1, 0.0003]
    
    auto_encoder = AutoEncoder(auto_encoder_args[0], auto_encoder_args[1], auto_encoder_args[2], auto_encoder_args[3], auto_encoder_args[4])
    auto_encoder.compile(
    loss='mse',
    metrics=['mae'],
    optimizer = tf.keras.optimizers.Adam(learning_rate=auto_encoder_args[5]))
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=100, min_lr=0.00001)
    history = auto_encoder.fit(
        X_train, 
        y_train, 
        epochs=400, 
        batch_size=auto_encoder_args[0], 
        callbacks = [scheduler],
        verbose = 0
    )

    history = model.fit(auto_encoder.encoder(X_train),y_train, epochs = 1000, batch_size=16, callbacks=[scheduler], verbose = 0)
    # visualizer(model, format='png', view=True)
    # plot_model(model, to_file="feed-forward.png", show_shapes=True, show_layer_names=True)
    # plot_model(auto_encoder, to_file="auto-encoder.png", show_shapes=True, show_layer_names=True)
    y_test = np.squeeze(y_test)
    prediction_distribution = model(auto_encoder.encoder(X_test))
    predictions = model.predict(auto_encoder.encoder(X_test)).flatten()

    if age_transform == "loglinear":
        predictions = test_age_transformer.inverse_transform(predictions)
        y_test = test_age_transformer.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = median_absolute_error(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    print(histone_mark_str)
    print("--------------------")
    print('Pearsons correlation: %.3f' % corr)
    print("MSE:", mse)
    print("MSE:", mae)

    # df_dict = {"Actual Age": y_test, "Predicted Mean Age": predictions, "Predicted Stddev": prediction_distribution.stddev().numpy().flatten()}
    # df = pd.DataFrame(df_dict)
    # df.to_csv('/gpfs/data/rsingh47/masif/ChromAge/Model-' + histone_mark_str + '_results.csv')

def main(metadata, histone_data_object, histone_mark_str, process = False, GEO = False):
    metadata = filter_metadata(metadata, biological_replicates = True)
    X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object)

    if process:
        post_process(metadata, histone_data_object, histone_mark_str, X_train, X_test, y_train, y_test)
    elif GEO:
        training_x = np.concatenate((np.array(X_train), np.array(X_test)), axis=0)
        training_y = np.concatenate((np.array(y_train), np.array(y_test)), axis=0)
        
        #GEO
        # GEO_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/GEO_histone_data/H3K4me3/processed_data/H3K4me3_mean_bins.pkl', 'rb'))
        GEO_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/GEO_histone_data/H3K27ac/processed_data/H3K27ac_mean_bins.pkl', 'rb'))
        metadata = pd.read_csv('/users/masif/data/masif/ChromAge/GEO_metadata.csv')

        X_train, X_test, y_train, y_test = split_data(metadata, GEO_data_object, histone_str = histone_mark_str + " SRR list", GEO = GEO)
        testing_x = np.concatenate((np.array(X_train), np.array(X_test)), axis=0)
        testing_y = np.concatenate((np.array(y_train), np.array(y_test)), axis=0)

        test_model(training_x, testing_x, training_y, testing_y, histone_mark_str, data_transform = "scaler", age_transform = "loglinear")
        # test_model(training_x, testing_x, training_y, testing_y, histone_mark_str, data_transform = "robust")

        model = ElasticNet(max_iter=1000, random_state = 42)
        model.fit(training_x, training_y)
        predictions = model.predict(testing_x)
        mse = mean_squared_error(testing_y, predictions)
        mae = median_absolute_error(testing_y, predictions)
        corr, _ = pearsonr(testing_y, predictions)
        print('Pearsons correlation: %.3f' % corr)
        print("Mean Median AE: ", mae, "\n Mean MSE:", mse)

    else:
        param_grid = {
            'epochs':[1000],
            'batch_size': [16, 48],
            'hidden_layers':[3,5],
            'lr':[0.0001, 0.0002, 0.0003],
            'dropout':[0.0, 0.05, 0.1, 0.2],
            'coeff':[0.01, 0.05, 0.1],
            'latent_size':[50,150,300,450],
            'gaussian_noise':[0.1,0.2,0.3]
        }

        experiment_DataFrame, metrics_dict = run_grid_search(metadata, histone_data_object, param_grid)
        experiment_DataFrame.to_csv('/gpfs/data/rsingh47/masif/ChromAge/NN-' + histone_mark_str + '_results.csv')
        with open('metrics-output-' + histone_mark_str + '.txt', 'w') as convert_file:
            convert_file.write(json.dumps(metrics_dict))

def get_shap_values(model, X_train, X_test, histone_mark_str):
    explainer = shap.GradientExplainer(model, np.array(X_train))

    # shap_values_train = explainer.shap_values(np.array(X_train))
    # shap_values_test = explainer.shap_values(np.array(X_test))
    # pd.Series(shap_values_test).to_pickle('annotation/' + histone_mark_str +'_shap_values_test.pkl')

    shap_values = pd.read_pickle('annotation/' + histone_mark_str +'_shap_values_test.pkl')
    shap_vals = pd.DataFrame(shap_values[0], columns = X_test.columns.values.tolist())
    vals = np.abs(shap_vals).mean()
    shap_importance = pd.DataFrame(list(zip(X_test.columns.values.tolist(), vals)), columns=['col_name','shap_importance'])
    shap_importance.sort_values(by=['shap_importance'], ascending=False,inplace=True)
    shap_importance.shap_importance = 100*shap_importance.shap_importance/np.sum(shap_importance.shap_importance)
    feature_importance = shap_importance.sort_values('shap_importance', ascending = False).reset_index()
    shap_fig0, ax = plt.subplots(figsize=(10,10), dpi = 1000)
    # ax.set_xlim(-1.5, 2.5)
    #ax.set_ylim(-2.5,1.4)
    print(feature_importance.col_name[0])
    print(shap_values)
    shap.dependence_plot(feature_importance.col_name[0], shap_values[0], np.array(X_test), feature_names=X_test.columns.values.tolist(),
                        alpha = 1, ax = ax, dot_size=300)
    ax.set_ylabel('SHAP value')
    ax.set_xlabel(feature_importance.col_name[0])
    shap_fig0.savefig('annotation/' + histone_mark_str +'_shap_fig0_revision.pdf', bbox_inches='tight')
    shap_fig0

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

if __name__ == '__main__':
    #Encode
    metadata = pd.read_pickle('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/metadata_summary.pkl') 

    H3K4me3_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K4me3/processed_data/H3K4me3_mean_bins.pkl', 'rb'))
    H3K27ac_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K27ac/processed_data/H3K27ac_mean_bins.pkl', 'rb'))
    H3K27me3_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K27me3/processed_data/H3K27me3_mean_bins.pkl', 'rb'))
    H3K36me3_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K36me3/processed_data/H3K36me3_mean_bins.pkl', 'rb'))
    H3K4me1_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K4me1/processed_data/H3K4me1_mean_bins.pkl', 'rb'))
    H3K9me3_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K9me3/processed_data/H3K9me3_mean_bins.pkl', 'rb'))
    
    # For Grid Searches
    # main(metadata, H3K4me3_data_object, "H3K4me3")
    # main(metadata, H3K27ac_data_object, "H3K27ac")
    # main(metadata, H3K27me3_data_object, "H3K27me3")
    # main(metadata, H3K36me3_data_object, "H3K36me3")
    # main(metadata, H3K4me1_data_object, "H3K4me1")
    # main(metadata, H3K9me3_data_object, "H3K9me3")

    # For post-processing
    main(metadata, H3K4me3_data_object, "H3K4me3", process = True) # Best Model: simple_nn 16 5 0.0003 0.0 0.01 50 0.1
    # main(metadata, H3K27ac_data_object, "H3K27ac", process = True) # Best Model: simple_nn 16 3 0.0002 0.05 0.1 150 0.2 / simple_nn 16 3 0.0003 0.0 0.1 50 0.2
    # main(metadata, H3K27me3_data_object, "H3K27me3", process = True) # Best Model: simple_nn 16 3 0.0003 0.0 0.1 300 0.1
    # main(metadata, H3K36me3_data_object, "H3K36me3", process = True) # Best Model: simple_nn 16 3 0.0003 0.0 0.1 50 0.1
    # main(metadata, H3K4me1_data_object, "H3K4me1", process = True) # Best Model: simple_nn 16 3 0.0003 0.0 0.01 50 0.2 / simple_nn 16 5 0.0002 0.1 0.05 50 0.1
    # main(metadata, H3K9me3_data_object, "H3K9me3", process = True) # Best Model: simple_nn 16 3 0.0001 0.0 0.05 50 0.1

    # GEO post_processing
    # main(metadata, H3K4me3_data_object, "H3K4me3", GEO = True)
    # main(metadata, H3K27ac_data_object, "H3K27ac", GEO = True)