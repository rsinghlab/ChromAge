import pandas as pd
from matplotlib import pyplot as plt
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

scatter_plot(
    ["H3K4me1_results.csv", "H3K36me3_results.csv",
     "H3K27me3_results.csv", "H3K27ac_results.csv",
      "H3K9me3_results.csv", "H3K4me3_results.csv"],
      ["ElasticNet-H3K4me1_results.csv", "ElasticNet-H3K36me3_results.csv",
     "ElasticNet-H3K27me3_results.csv", "ElasticNet-H3K27ac_results.csv",
      "ElasticNet-H3K9me3_results.csv", "ElasticNet-H3K4me3_results.csv"])
    