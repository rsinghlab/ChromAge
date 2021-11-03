import re
import numpy as np
import pandas as pd
import random
import pickle
import gc

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

def split_data(metadata, histone_data_object, biological_replicates = False, split = 0.2):
    #keep or remove biological replicates
    biological_replicate_experiments = metadata.groupby(['Experiment accession']).count()[metadata.groupby(['Experiment accession']).count()['Biological replicate(s)']>2].index
    
    if biological_replicates == True:
        #add the replicates here
        replicates_arr = []
        for replicate in biological_replicate_experiments:
            replicates_arr.append(metadata.loc[metadata['Experiment accession'].isin([replicate])])
    
    metadata = metadata[~metadata['Experiment accession'].isin(biological_replicate_experiments)]
    
    #ensures both X and y have same samples
    X = histone_data_object.df
    print(metadata.index)
    samples = np.intersect1d(metadata.index, X.index)
    X = X.loc[samples]
    y = metadata.loc[X.index].age

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = 42)  

    if biological_replicates == True:
        data_array = []

        for i in range(len(replicates_arr)):
            X = histone_data_object.df
            print(replicates_arr[i].index)
            samples = np.intersect1d(replicates_arr[i].index, X.index)
            print(samples)
            X = X.loc[samples]
            y = metadata.loc[X.index].age
            data_array.append((X,y))

        random.shuffle(data_array)
        print(len(data_array))
        train_data = data_array[0 : int((1-split) * len(data_array))]
        test_data = data_array[int((1-split) * len(data_array)) : len(data_array)]

        for x_replicate, y_replicate in train_data:
            print("HIT")
            X_train.append(x_replicate)
            y_train.append(y_replicate)

        for x_replicate, y_replicate in test_data:
            X_test.append(x_replicate)
            y_test.append(y_replicate)

    return X_train, X_test, y_train, y_test

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

def k_cross_validate_model(train_x, train_y, k):
    for i in range(k):
        validation_x = train_x[int(i*(1/k)*train_x.shape[0]):int((i+1)*(1/k)*train_x.shape[0])]
        validation_y = train_y[int(i*(1/k)*train_y.shape[0]):int((i+1)*(1/k)*train_y.shape[0])]
        training_x = np.concatenate((train_x[0:int(i*(1/k)*train_x.shape[0])],train_x[int((i+1)*(1/k)*train_x.shape[0]):train_x.shape[0]]), axis=0)
        training_y = np.concatenate((train_y[0:int(i*(1/k)*train_y.shape[0])],train_y[int((i+1)*(1/k)*train_y.shape[0]):train_y.shape[0]]), axis=0)
        model = tf.keras.Sequential()
        layer_1 = tf.keras.layers.Conv2D(8,9,activation=tf.keras.layers.ReLU(), padding="valid")
        layer_2 = tf.keras.layers.Conv2D(8,1,activation=tf.keras.layers.ReLU(), padding="valid")
        layer_3 = tf.keras.layers.Conv2D(1,5,activation=tf.keras.layers.ReLU(), padding="valid")
        model.add(layer_1)
        model.add(layer_2)
        model.add(layer_3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008), loss=tf.keras.losses.MeanSquaredError())
        model.fit(x=training_x, y=training_y, batch_size=20, epochs=20, validation_data=(validation_x,validation_y), shuffle=True)

def create_google_mini_net():
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input and first CONV module
    inputs = Input(shape=inputShape)
    x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
    # two Inception modules followed by a downsample module
    x = inception_module(x, 32, 32, chanDim)
    x = inception_module(x, 32, 48, chanDim)
    x = downsample_module(x, 80, chanDim)
    # four Inception modules followed by a downsample module
    x = inception_module(x, 112, 48, chanDim)
    x = inception_module(x, 96, 64, chanDim)
    x = inception_module(x, 80, 80, chanDim)
    x = inception_module(x, 48, 96, chanDim)
    x = downsample_module(x, 96, chanDim)
    # two Inception modules followed by global POOL and dropout
    x = inception_module(x, 176, 160, chanDim)
    x = inception_module(x, 176, 160, chanDim)
    x = AveragePooling2D((7, 7))(x)
    x = Dropout(0.5)(x)
    # softmax classifier
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    # create the model
    model = Model(inputs, x, name="minigooglenet")

def loss_function(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

#create neural network with adjustable parameters
def create_nn(hidden_layers = 5, hidden_layer_sizes = [16,32, 64, 64, 64], lr = 0.001, coeff = 0.01, dropout = 0.1):
    
    inputs = Input(shape = (30321,))
    x = BatchNormalization()(inputs)
    x = ActivityRegularization(coeff, coeff)(inputs)
    
    for i in range(hidden_layers):
        x = Dense(hidden_layer_sizes[i],activation = 'selu',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff),
                  activity_regularizer= tf.keras.regularizers.l1_l2(coeff, coeff))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
    x = Dense(32,activation = 'selu',
              kernel_regularizer = tf.keras.regularizers.l1_l2(coeff, coeff),
              activity_regularizer= tf.keras.regularizers.l1_l2(coeff, coeff))(x)

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

histone_data_object = pickle.load(open('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/H3K4me3/processed_data/H3K4me3_mean_bins.pkl', 'rb'))

metadata = pd.read_pickle('/users/masif/data/masif/ChromAge/encode_histone_data/human/tissue/metadata_summary.pkl') 
# metadata = filter_metadata(metadata)


X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object, True)

print(len(X_train), len(X_test), len(y_train), len(y_test))

X_train, X_test, y_train, y_test = split_data(metadata, histone_data_object, False)

print(len(X_train), len(X_test), len(y_train), len(y_test))

model = create_nn()

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

# # accession code, real age and the actual age, predicted mean, predicted stddev for each model, model type 

# prediction_distribution = model(x_test)

# prediction_distribution.mean().numpy().flatten()

# prediction_distribution.sttdev().numpy().flatten()

# # summarize results
# print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# params = grid.cv_results_['params']