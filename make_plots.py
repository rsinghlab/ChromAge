import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def scatter_plot(ModelArr, ENArr):
    actual_age = []
    predicted_age = []
    elasticNet_Age = []
    for csv in ModelArr:
        df = pd.read_csv(csv)
        predicted_age.extend(df["Predicted Mean Age"].to_numpy().flatten())
        actual_age.extend(df["Actual Age"].to_numpy().flatten())
    for csv in ENArr:
        df = pd.read_csv(csv)
        elasticNet_Age.extend(df["Predicted Mean Age"].to_numpy().flatten())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(actual_age, elasticNet_Age, color='purple', alpha=0.5, s=50, label="ElasticNet")
    ax.scatter(actual_age, predicted_age, color='darkblue', alpha=0.5, s=50, label="Neural Network")
    ax.plot(actual_age, actual_age, color="black")
    ax.set_ylabel("Predicted Age (years)")
    ax.set_xlabel("Actual Age (years)")
    ax.set_title("NN vs. ElasticNet Age Predictions")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.7)
    
    plt.show()
    return

def histogram():
    barWidth = 0.2
    nn_mae = [4.28, 7.76, 3.64, 3.46, 3.83, 4.36]
    en_mae = [7.04, 9.64, 7.37, 7.47, 7.00, 7.42]

    r1 = np.arange(len(nn_mae))
    r2 = [x + barWidth for x in r1] 

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(r1, nn_mae, width = barWidth, color = 'yellow', alpha=0.6, edgecolor = 'black', capsize=7, label='Neural Network')
    ax.bar(r2, en_mae, width = barWidth, color = 'lightgreen', alpha=1, edgecolor = 'black', capsize=7, label='ElasticNet') 

    plt.xticks([r + barWidth for r in range(len(nn_mae))], ['H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K9me3'])
    ax.set_ylabel('Test Median Absolute Error (MAE)')
    ax.set_xlabel("Histone Marks")
    ax.set_title("NN vs. ElasticNet Test MAE")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.7)
    plt.show()

    return

ModelCsvArr = ["H3K4me1_results.csv", "H3K36me3_results.csv",
     "H3K27me3_results.csv", "H3K27ac_results.csv",
      "H3K9me3_results.csv", "H3K4me3_results.csv"]

ElasticNetCsvArr = ["ElasticNet-H3K4me1_results.csv", "ElasticNet-H3K36me3_results.csv",
     "ElasticNet-H3K27me3_results.csv", "ElasticNet-H3K27ac_results.csv",
      "ElasticNet-H3K9me3_results.csv", "ElasticNet-H3K4me3_results.csv"]

# scatter_plot(ModelCsvArr,ElasticNetCsvArr)
histogram()
    