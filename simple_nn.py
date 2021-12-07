import re
import numpy as np
import pandas as pd
import random
import pickle
import gc

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
from tensorflow.keras.layers import Embedding, Bidirectional,Conv1DTranspose, ActivityRegularization, Input, LSTM, ReLU, GRU, multiply, Lambda, PReLU, SimpleRNN, Dense, Activation, BatchNormalization, Conv2D, Conv1D, Flatten, LeakyReLU, Dropout, MaxPooling2D, MaxPooling1D, Reshape
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

    experiment_training, experiment_testing = train_test_split(metadata_temp.groupby(['Experiment accession']).count().index, test_size = split)

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

    kf = KFold(n_splits=k)

    kfold_data = kf.split(metadata_temp.groupby(['Experiment accession']).count().index)

    for train_index, val_index in kfold_data:
        
        experiment_training = metadata_temp.groupby(['Experiment accession']).count().index[train_index]
        experiment_val = metadata_temp.groupby(['Experiment accession']).count().index[val_index]
        
        training_list = [i in experiment_training for i in np.array(metadata_temp['Experiment accession'])]
        train_metadata = metadata_temp.loc[training_list, :]
        
        val_list = [i in experiment_val for i in np.array(metadata_temp['Experiment accession'])]
        val_metadata = metadata_temp.loc[val_list, :]

        training_x = X.loc[train_metadata.index]
        print(training_x)
        training_y = train_metadata.loc[training_x.index].age

        validation_x = X.loc[val_metadata.index]
        validation_y = val_metadata.loc[validation_x.index].age

        validation_y_index = validation_y.index
        
        model = create_nn(model_params[0], model_params[1], model_params[2], model_params[3])
        model.fit(np.array(training_x), np.array(training_y), batch_size, epochs, verbose=1, validation_data=(np.array(validation_x), np.array(validation_y)))
        
        results = model.evaluate(np.asarray(validation_x), np.asarray(validation_y), batch_size)
        print("Validation metrics:", results)     
        prediction_distribution = model(np.asarray(validation_x))
        type_arr = np.full(np.asarray(validation_y).shape, model_type)

        if df is None:
            df_dict = {"Actual Age": np.asarray(validation_y), "Predicted Mean Age": prediction_distribution.mean().numpy().flatten(), "Predicted Stddev": prediction_distribution.stddev().numpy().flatten(), "Model Type" : type_arr}
            df = pd.DataFrame(df_dict, index = validation_y_index)
        else:
            df_dict = {"Actual Age": np.asarray(validation_y), "Predicted Mean Age": prediction_distribution.mean().numpy().flatten(), "Predicted Stddev": prediction_distribution.stddev().numpy().flatten(), "Model Type" : type_arr}
            df2 = pd.DataFrame(df_dict, index = validation_y_index)
            df = df.append(df2)
        
        print(df)
    return df

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
                  kernel_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff)))
                #   activity_regularizer= tf.keras.regularizers.l1_l2(coeff, coeff)))
        model.add(Activation('selu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(32, activation='selu'))

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

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

def run_grid_search(metadata, histone_data_object, param_grid):
    X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object)
    df = None
    # imputer = KNNImputer()
    # scaler = StandardScaler()
    # X_train, y_train = imputer.fit_transform(X= X_train, y = y_train)
    # X_train, y_train = scaler.fit_transform(X= X_train, y = y_train)

    for epoch in param_grid['epochs']:
        for batch in param_grid['batch_size']:
            for hidden_layers in param_grid['hidden_layers']:
                for lr in param_grid['lr']:
                    for dropout in param_grid['dropout']:
                        for coeff in param_grid['coeff']:
                            model_params = [hidden_layers, lr, dropout, coeff]
                            str_model_params = [str(param) for param in model_params]
                            df = k_cross_validate_model(metadata, histone_data_object, y_test, batch, epoch, "simple_nn_new " + str(batch) +" "+" ".join(str_model_params), model_params, df)
                            model = create_nn(model_params[0], model_params[1], model_params[2], model_params[3])
                            history = model.fit(X_train,y_train, epochs = epoch, verbose=0)
                            # predictions = model.predict(X_test)
                            print(history.history)
    return df

param_grid = {
    'epochs':[1000],
    'batch_size': [16, 32],
    'hidden_layers':[1,3,5],
    'lr':[0.0001, 0.001, 0.01],
    'dropout':[0.0,0.1,0.3],
    'coeff':[0.005, 0.01, 0.05],
}

histone_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K4me3/processed_data/H3K4me3_mean_bins.pkl', 'rb'))
metadata = pd.read_pickle('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/metadata_summary.pkl') 
metadata = filter_metadata(metadata, biological_replicates = True)

X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object)

df = k_cross_validate_model(metadata, histone_data_object, y_test, 32, 1000, "", [3, 0.0001, 0.1, 0.05], None)

# model = create_nn(3, 0.001, 0.1,0)
# history = model.fit(X_train,y_train, epochs = 1000, verbose=0)
# prediction_distribution = model(np.asarray(X_test))
# results = model.evaluate(np.asarray(X_test), np.asarray(y_test), 32, verbose = 1)
# print("Validation metrics:", results) 
# predictions = model.predict(np.asarray(X_test), verbose = 1)
# df_dict = {"Actual Age": np.asarray(y_test), "Predicted Mean Age": predictions, "Predicted Stddev": prediction_distribution.stddev().numpy().flatten()}
# print(df_dict)

# experiment_DataFrame = run_grid_search(metadata, histone_data_object, param_grid)

# experiment_DataFrame.to_csv('/gpfs/data/rsingh47/masif/ChromAge/simple_nn_new_results.csv')

# history_cache = model.fit(X,y, epochs=100)

# print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))

# neural_network = KerasRegressor(build_fn = create_nn, verbose = 0)

# scaler = StandardScaler()

# # create parameter grid, as usual, but note that you can
# # vary other model parameters such as 'epochs' (and others 
# # such as 'batch_size' too)
# param_grid = {
#     'neural_network__epochs':[100,500],
#     'neural_network__hidden_layers':[1,3,5],
#     'neural_network__hidden_layer_sizes':[[64],[16,32,64],[16,32,64,64,64]],
#     'neural_network__lr':[0.00001,0.00005, 0.001, 0.01],
#     'neural_network__dropout':[0.0,0.1,0.3,0.5],
#     'neural_network__coeff':[0.005, 0.05, 0.01],
# }

# pipeline = Pipeline(steps = [('imputer', KNNImputer()), ('scaler', StandardScaler()), ('neural_network', neural_network)])

# # if you're not using a GPU, you can set n_jobs to something other than 1
# grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
# grid.fit(X, y)

# # summarize results
# print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# params = grid.cv_results_['params']