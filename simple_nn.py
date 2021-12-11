from collections import defaultdict
import numpy as np
import pandas as pd
import random
import pickle
import gc
import ast
import json
import os

from tensorflow.python.ops.gen_nn_ops import Selu

from gtfparse import read_gtf
import pyBigWig

from os import listdir
from os.path import isfile, join

from progressbar import ProgressBar, Bar, Percentage, AnimatedMarker, AdaptiveETA
from IPython.display import clear_output

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import regularizers, datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional,Conv1DTranspose, ActivityRegularization, Input, LSTM, ReLU, GRU, multiply, Lambda, PReLU, SimpleRNN, Dense, Activation, BatchNormalization, Conv2D, Conv1D, Flatten, LeakyReLU, Dropout, GaussianNoise, MaxPooling1D, Reshape, UpSampling1D
import tensorflow_probability as tfp
import keras.backend as K
from matplotlib import pyplot as plt

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

def split_data(metadata, histone_data_object, split = 0.2):
    X = histone_data_object.df
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

def k_cross_validate_model(metadata, histone_data_object, y_test, batch_size, epochs, model_type, model_params, df, k = 4):
    metadata = metadata.drop(y_test.index)

    X = histone_data_object.df
    samples = np.intersect1d(metadata.index, X.index)
    metadata_temp = metadata.loc[samples, :]

    kf = KFold(n_splits=k, shuffle=True)

    kfold_data = kf.split(metadata_temp.groupby(['Experiment accession']).count().index)
    val_metrics_array = []
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
        # print(training_x)
        training_y = train_metadata.loc[training_x.index].age

        validation_x = X.loc[val_metadata.index]
        validation_y = val_metadata.loc[validation_x.index].age

        validation_y_index = validation_y.index

        # train_auto_encoder = auto_encoder.predict(np.array(training_x))
        # val_auto_encoder = auto_encoder.predict(np.array(validation_x))

        # mse = tf.keras.losses.MeanSquaredError()
        # print("Average validation mean squared error for auto-encoder:",np.mean(mse(val_auto_encoder,np.array(validation_x)).numpy()))

        model = create_nn(model_params[0], model_params[1], model_params[2], model_params[3])
        history = model.fit(np.array(training_x), np.array(training_y), batch_size, epochs, verbose=0, validation_data=(np.array(validation_x), np.array(validation_y)))
        min_train_loss_array.append(np.min(history.history['loss']))
        min_train_mse_array.append(np.min(history.history['mse']))
        min_train_mae_array.append(np.min(history.history['mae']))
        min_val_loss_array.append(np.min(history.history['val_loss']))
        min_val_mse_array.append(np.min(history.history['val_mse']))
        min_val_mae_array.append(np.min(history.history['val_mae']))

        results = model.evaluate(np.array(validation_x), np.array(validation_y), int(batch_size/2))
        # print("Validation metrics:", results)     
        val_metrics_array.append(results)

        prediction_distribution = model(np.array(validation_x))
        type_arr = np.full(np.array(validation_y).shape, model_type)

        if df is None:
            df_dict = {"Actual Age": np.array(validation_y), "Predicted Mean Age": prediction_distribution.mean().numpy().flatten(), "Predicted Stddev": prediction_distribution.stddev().numpy().flatten(), "Model Type" : type_arr}
            df = pd.DataFrame(df_dict, index = validation_y_index)
        else:
            df_dict = {"Actual Age": np.array(validation_y), "Predicted Mean Age": prediction_distribution.mean().numpy().flatten(), "Predicted Stddev": prediction_distribution.stddev().numpy().flatten(), "Model Type" : type_arr}
            df2 = pd.DataFrame(df_dict, index = validation_y_index)
            df = df.append(df2)
        print(df)
    return df, val_metrics_array, min_train_loss_array, min_train_mse_array, min_train_mae_array, min_val_loss_array, min_val_mse_array, min_val_mae_array

def loss_function(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

#create neural network with adjustable parameters
def create_nn(hidden_layers = 3, lr = 0.001, dropout = 0.1, coeff = 0.01):
    hidden_layer_sizes = []

    # hidden layer size
    for i in range(hidden_layers):
        hidden_layer_sizes.append(32)
    
    model = Sequential()

    model.add(Input(shape = (30321,)))
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

def create_LSTM(hidden_layers = 3, lr = 0.001, dropout = 0.1, coeff = 0.01):
    hidden_layer_sizes = []

    if hidden_layers == 1:
        hidden_layer_sizes.append(64)
    else:
        for i in range(hidden_layers):
            hidden_layer_sizes.append(16 * (i+1))

    inputs = Input(shape = (30321,))
    x = BatchNormalization()(inputs)
    # x = ActivityRegularization(coeff, coeff)(inputs)
    
    x = tf.expand_dims(tf.convert_to_tensor(x), axis = 0)

    x, _, _ = LSTM(hidden_layer_sizes[0], return_sequences=True, return_state=True)(x)

    x = tf.squeeze(x, axis=0)

    for i in range(1, hidden_layers):
        x = Dense(hidden_layer_sizes[i],activation = 'selu',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff),
                  activity_regularizer= tf.keras.regularizers.l1_l2(coeff, coeff))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    distribution_params = Dense(2, activation='relu')(x)
    outputs = tfp.layers.DistributionLambda(
      lambda t: tfp.distributions.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(distribution_params)
    model = Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse', 'mae'])    

    return model

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.batch_size = 32
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.latent_size = 250
        self.hidden_dim = 1000
        self.encoder = Sequential([
            Dense(self.hidden_dim, activation='selu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
            # Dropout(0.1),
            Dense(int(self.hidden_dim/2), activation='selu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
            # Dropout(0.1),
            Dense(self.latent_size, activation='selu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
            Dropout(0.1),
        ])
        self.decoder = Sequential([
            Dense(int(self.hidden_dim/2), activation='selu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
            # Dropout(0.1),
            Dense(self.hidden_dim, activation='selu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
            # Dropout(0.1),
            Dense(30321, activation=None)
        ])

        # self.encoder = Sequential(layers=[
        #     GaussianNoise(0.2),
        #     Conv1D(32, 3, activation='relu', padding='same'),
        #     MaxPooling1D(2, padding='same'),
        #     Conv1D(16, 3, activation='relu', padding='same'),
        #     MaxPooling1D(2, padding='same'),
        #     Conv1D(8, 3, activation='relu', padding='same'),
        #     MaxPooling1D(2, padding='same')])

        # self.decoder = Sequential(layers=[
        #     Conv1D(8, 3, activation='relu', padding='same'),
        #     UpSampling1D(2),
        #     Conv1D(16, 3, activation='relu', padding='same'),
        #     UpSampling1D(2),
        #     Conv1D(32, 3, activation='relu', padding='same'),
        #     UpSampling1D(2),
        #     Conv1D(1, 3, activation='sigmoid', padding='same')])
    
    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        return tf.squeeze(self.decoder(encoder_output))
    
    def train(self, train_inputs, num_epochs):
        for i in range(num_epochs):
            loss_list = []
            indices = tf.random.shuffle([x for x in range(len(train_inputs))])
            train_inputs = tf.gather(train_inputs,indices)
            for i in range(0,len(train_inputs),self.batch_size):
                batched_inputs = train_inputs[i:i+self.batch_size]
                with tf.GradientTape() as tape:
                    predictions = self.call(batched_inputs)
                    loss = self.loss(predictions,batched_inputs)
                    loss_list.append(loss.numpy())
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            print(sum(loss_list)/len(loss_list))

def analyze_metrics(file_path):
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
        df.to_csv("simple_nn_grid_metrics.csv")
        best_train_loss_models = df.nsmallest(10, 'mean_train_loss')
        best_train_loss_models.to_csv("best_train_loss_models.csv")
        best_val_loss_models = df.nsmallest(20, 'mean_val_loss')
        best_val_loss_models.to_csv("best_val_loss_models.csv")

        best_train_mse_models = df.nsmallest(10, 'mean_train_mse')
        best_train_mse_models.to_csv("best_train_mse_models.csv")
        best_val_mse_models = df.nsmallest(20, 'mean_val_mse')
        best_val_mse_models.to_csv("best_val_mse_models.csv")

        best_train_mae_models = df.nsmallest(10, 'mean_train_mae')
        best_train_mae_models.to_csv("best_train_mae_models.csv")
        best_val_mae_models = df.nsmallest(20, 'mean_val_mae')
        best_val_mae_models.to_csv("best_val_mae_models.csv")
        
        best_val_models = set()
        best_train_models = set()

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

        print("Best val models:", *list(best_val_models), sep='\n')
        print("Best train models:", *list(best_train_models), sep='\n')
    
    return list(best_val_models), list(best_train_models)

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
                            model_params = [hidden_layers, lr, dropout, coeff]
                            str_model_params = [str(param) for param in model_params]
                            model_name = "simple_nn " + str(batch) +" "+" ".join(str_model_params)
                            df, val_metrics_array, min_train_loss_array, min_train_mse_array, min_train_mae_array, min_val_loss_array, min_val_mse_array, min_val_mae_array = k_cross_validate_model(metadata, histone_data_object, y_test, batch, epoch, model_name, model_params, df)
                            metrics_dict[model_name] = dict({"val_metrics" : val_metrics_array, "min_train_loss" : min_train_loss_array, "min_val_loss" : min_val_loss_array, "min_train_mse" : min_train_mse_array, "min_val_mse" : min_val_mse_array, "min_train_mae" : min_train_mae_array, "min_val_mae" : min_val_mae_array})
                            print("run for model " + model_name)
                            print(metrics_dict)
    return df, metrics_dict

def run_model():
    histone_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K4me3/processed_data/H3K4me3_mean_bins.pkl', 'rb'))
    metadata = pd.read_pickle('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/metadata_summary.pkl') 
    metadata = filter_metadata(metadata, biological_replicates = True)

    X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object)

    # imputer = KNNImputer()
    # scaler = StandardScaler()

    # X = histone_data_object.df
    # samples = np.intersect1d(metadata.index, X.index)
    # metadata_temp = metadata.loc[samples, :]

    # all_data_x = X.loc[metadata_temp.index]
    # auto_encoder = AutoEncoder()
    # auto_encoder.train(np.array(all_data_x), 10)

    # df = k_cross_validate_model(auto_encoder, metadata, histone_data_object, y_test, 32, 1000, "", [3, 0.0001, 0.1, 0.01], None)

    # df.to_csv('/gpfs/data/rsingh47/masif/ChromAge/simple_nn_results.csv')

    # param_grid = {
    #     'epochs':[1000],
    #     'batch_size': [16,32,48],
    #     'hidden_layers':[1,3,5],
    #     'lr':[0.0001, 0.0002, 0.0003],
    #     'dropout':[0.0,0.05, 0.1, 0.125],
    #     'coeff':[0.01, 0.02, 0.05, 0.1]
    # }

    # experiment_DataFrame, metrics_dict = run_grid_search(metadata, histone_data_object, param_grid)
    # experiment_DataFrame.to_csv('/gpfs/data/rsingh47/masif/ChromAge/simple_nn_results.csv')
    # with open('metrics-output.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(metrics_dict))

    best_val_models, best_train_models = analyze_metrics(os.getcwd() + "/metrics-output.txt")
    test_df = None

    # for model_name in best_val_models:
    #     model_params = model_name.split(" ")
    #     batch_size = int(model_params[1])
    #     num_layers = int(model_params[2])
    #     learning_rate = float(model_params[3])
    #     dropout = float(model_params[4])
    #     coeff = float(model_params[5])

    #     model = create_nn(num_layers, learning_rate, dropout, coeff)
    #     history = model.fit(np.array(X_train),np.array(y_train), epochs = 1000, batch_size=batch_size, verbose = 0)
    #     print("Model: ", model_name, "with min loss, mse, mae: ", [np.min(history.history['loss']), np.min(history.history['mse']), np.min(history.history['mae'])])

    #     prediction_distribution = model(np.array(X_test))
    #     results = model.evaluate(np.array(X_test), np.array(y_test), batch_size)
    #     print("Testing metrics (loss, mse, mae) for model:", model_name, results) 
    #     predictions = model.predict(np.array(X_test))

    #     type_arr = np.full(np.array(y_test).shape, model_name)
    #     df_dict = {"Model": type_arr, "Actual Age": np.array(y_test), "Predicted Mean Age": np.array(predictions).flatten(), "Predicted Stddev": prediction_distribution.stddev().numpy().flatten()}
    #     if test_df is None:
    #         test_df = pd.DataFrame(df_dict, index = y_test.index)
    #         print(pd.DataFrame(df_dict, index = y_test.index))
    #     else:
    #         test_df = test_df.append(pd.DataFrame(df_dict, index = y_test.index))
    #         print(pd.DataFrame(df_dict, index = y_test.index))
    
    # print(test_df)
    # test_df.to_csv("Best_Models_testing.csv")

    model = create_nn(3, 0.0003, 0.03, 0.01)
    history = model.fit(np.array(X_train),np.array(y_train), epochs = 1000, batch_size=48, verbose=0)
    print("Min loss, mse, mae: ", [np.min(history.history['loss']), np.min(history.history['mse']), np.min(history.history['mae'])])
    prediction_distribution = model(np.array(X_test))
    results = model.evaluate(np.array(X_test), np.array(y_test), 48, verbose = 0)
    print("Testing metrics:", results) 
    predictions = model.predict(np.array(X_test), verbose = 0)
    df_dict = {"Actual Age": np.array(y_test), "Predicted Mean Age": np.array(predictions).flatten(), "Predicted Stddev": prediction_distribution.stddev().numpy().flatten()}
    print(pd.DataFrame(df_dict, index = y_test.index))

if __name__ == '__main__':
    run_model()