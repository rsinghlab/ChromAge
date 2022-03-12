import numpy as np
import pandas as pd
import urllib.request

organism = 'human'

def setup_tissue(path, data_type = 'tissue'):
    metadata = pd.read_csv(path + '/' + data_type + '/metadata.tsv', sep = '\t', index_col = 0, low_memory=False)

    metadata = metadata[metadata['File type'] == 'bigWig']
    metadata = metadata[metadata['File assembly'] == 'GRCh38']
    metadata = metadata[metadata['File analysis status'] == 'released']
    metadata = metadata[metadata['Biosample treatments'].isnull()]
    metadata['Technical replicate(s)'] = metadata['Technical replicate(s)'].apply(lambda x: [int(t[-1]) for t in x.split(',', 1)])
    metadata['Biological replicate(s)'] = metadata['Biological replicate(s)'].apply(lambda x: [int(b[-1]) for b in x.split(',', 1)])

    metadata_summary = metadata[['Biosample term name', 'Experiment accession', 'Experiment target','Biological replicate(s)', 'Technical replicate(s)', 'Audit WARNING', 'Audit NOT_COMPLIANT','Audit ERROR']]

    report = pd.read_csv(path + '/' + data_type + '/report.tsv', sep = '\t', header = 1, index_col = 1, low_memory=False)

    normal_gestational_week = 40
    ages = pd.DataFrame(np.empty([report.shape[0],1]), columns = ['age',], index = report.index)
    for sample in report.index:
        age_string = report.loc[sample,'Biosample age']
        if type(age_string) == float or type(age_string) == int:
            age = age_string
            if age > 0 and age.is_integer():
                ages.loc[sample] = age + 0.5
            else: 
                ages.loc[sample] = age
        elif 'years' in age_string and 'above' not in age_string:
            age = float(age_string.split(' ', 1)[0])
            if age > 0 and age.is_integer():
                ages.loc[sample] = age + 0.5
            else: 
                ages.loc[sample] = age
        elif 'weeks' in age_string:
            age = (float(age_string.split(' ', 1)[0]) - normal_gestational_week)*7/365
            ages.loc[sample] = age 
        elif 'days' in age_string:
            age = (float(age_string.split(' ', 1)[0]) - normal_gestational_week*7)/365
            ages.loc[sample] = age 
        else:
            ages.loc[sample] = np.nan

    gender = pd.DataFrame(np.empty([report.shape[0],1]), columns = ['gender',], index = report.index)
    for sample in report.index:
        gender_string = report.loc[sample,'Biosample summary']
        if 'female' in gender_string:
            gender.loc[sample] = 'F' 
        elif 'male' in gender_string:
            gender.loc[sample] = 'M'    
        else:
            gender.loc[sample] = np.nan  

    ages = ages.loc[metadata_summary['Experiment accession']]
    gender = gender.loc[metadata_summary['Experiment accession']]
    description = report['Description'].loc[metadata_summary['Experiment accession']]
    biosample = report['Biosample accession'].loc[metadata_summary['Experiment accession']]
    ages.index = metadata_summary.index
    gender.index = metadata_summary.index
    description.index = metadata_summary.index
    biosample.index = metadata_summary.index

    metadata_summary = pd.concat([ages, gender, metadata_summary, biosample, description], axis = 1)

    metadata_summary = metadata_summary.dropna(subset=['age'])

    metadata_summary.to_pickle(path + '/' + data_type + '/metadata_summary.pkl')

    return metadata, metadata_summary

def download_tissue(path, metadata, metadata_summary, data_type = 'tissue'):
    metadata = metadata.loc[metadata_summary.index]

    for histone_mark in np.unique(metadata['Experiment target']):
        
        metadata_mark = metadata[metadata['Experiment target'] == histone_mark]
        
        #to download from a txt file using the terminal
        file = open(path + '/' + data_type + '/' + histone_mark[:-6] + "/raw_data/files_" + histone_mark[:-6] + ".txt", "w+")
        for url in metadata_mark['File download URL']:
            file.write(url + '\n')
        file.close()

def setup_primary_cell(path, data_type='primary_cell'):
    metadata = pd.read_csv(path + '/' + data_type + '/metadata.tsv', sep = '\t', index_col = 0, low_memory=False)
    metadata = metadata[metadata['File type'] == 'bigWig']
    metadata = metadata[metadata['File assembly'] == 'GRCh38']
    metadata = metadata[metadata['File analysis status'] == 'released']
    metadata = metadata[metadata['Biosample treatments'].isnull()]
    metadata['Technical replicate(s)'] = metadata['Technical replicate(s)'].apply(lambda x: [int(t[-1]) for t in x.split(',', 1)])
    metadata['Biological replicate(s)'] = metadata['Biological replicate(s)'].apply(lambda x: [int(b[-1]) for b in x.split(',', 1)])
    
    metadata_summary = metadata[['Biosample term name', 'Experiment accession', 'Experiment target','Biological replicate(s)', 'Technical replicate(s)', 'Audit WARNING', 'Audit NOT_COMPLIANT','Audit ERROR']]
    report = pd.read_csv(path + '/' + data_type + '/report.tsv', sep = '\t', header = 1, index_col = 1, low_memory=False)

    normal_gestational_week = 40
    ages = pd.DataFrame(np.empty([report.shape[0],1]), columns = ['age',], index = report.index)
    for sample in report.index:
        age_string = report.loc[sample,'Biosample age']
        if type(age_string) == float or type(age_string) == int:
            age = age_string
            if age > 0 and age.is_integer():
                ages.loc[sample] = age + 0.5
            else: 
                ages.loc[sample] = age
        elif 'years' in age_string and 'above' not in age_string:
            age = float(age_string.split(' ', 1)[0])
            if age > 0 and age.is_integer():
                ages.loc[sample] = age + 0.5
            else: 
                ages.loc[sample] = age  
        elif 'weeks' in age_string:
            age = (float(age_string.split(' ', 1)[0]) - normal_gestational_week)*7/365
            ages.loc[sample] = age 
        elif 'days' in age_string:
            if age_string == '2-4 days': #this just codes these cells as 3 days old
                age = (3 - normal_gestational_week*7)/365 
                ages.loc[sample] = age 
                continue
            age = (float(age_string.split(' ', 1)[0]) - normal_gestational_week*7)/365
            ages.loc[sample] = age 
        else:
            ages.loc[sample] = np.nan
    
    gender = pd.DataFrame(np.empty([report.shape[0],1]), columns = ['gender',], index = report.index)
    for sample in report.index:
        gender_string = report.loc[sample,'Biosample summary']
        if 'female' in gender_string:
            gender.loc[sample] = 'F' 
        elif 'male' in gender_string:
            gender.loc[sample] = 'M'    
        else:
            gender.loc[sample] = np.nan 

    ages = ages.loc[metadata_summary['Experiment accession']]
    gender = gender.loc[metadata_summary['Experiment accession']]
    description = report['Description'].loc[metadata_summary['Experiment accession']]
    biosample = report['Biosample accession'].loc[metadata_summary['Experiment accession']]
    ages.index = metadata_summary.index
    gender.index = metadata_summary.index
    description.index = metadata_summary.index
    biosample.index = metadata_summary.index

    metadata_summary = pd.concat([ages, gender, metadata_summary, biosample, description], axis = 1)
    metadata_summary = metadata_summary.dropna(subset=['age'])
    metadata_summary.to_pickle(path + '/' + data_type + '/metadata_summary.pkl')

    return metadata, metadata_summary

def download_primary(path, metadata, metadata_summary, data_type='primary_cell'):
    metadata = metadata.loc[metadata_summary.index]

    for histone_mark in np.unique(metadata['Experiment target']): 
        metadata_mark = metadata[metadata['Experiment target'] == histone_mark]
        
        #to download from a txt file using the terminal
        file = open(path + '/' + data_type + '/' + histone_mark[:-6] + "/raw_data/files_" + histone_mark[:-6] + ".txt", "w+")
        for url in metadata_mark['File download URL']:
            file.write(url + '\n')
        file.close()

path = "/Users/haider/Documents/Fall-2021/ChromAge/encode_histone_data/human"

metadata, metadata_summary = setup_tissue(path)

download_tissue(path, metadata, metadata_summary)

metadata, metadata_summary = setup_primary_cell(path)

download_primary(path, metadata, metadata_summary)

