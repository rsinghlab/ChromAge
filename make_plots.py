from turtle import color
import pandas as pd
from matplotlib import pyplot as plt
def scatter_plot(csvArr):
    actual_age = []
    predicted_age = []
    for csv in csvArr:
        df = pd.read_csv(csv)
        predicted_age.extend(df["Predicted Mean Age"].to_numpy().flatten())
        actual_age.extend(df["Actual Age"].to_numpy().flatten())
    plt.scatter(actual_age, predicted_age, color='green', s=20)
    plt.plot(actual_age, actual_age, color="black")
    plt.show()

scatter_plot(["H3K4me1_results.csv", "H3K36me3_results.csv", "H3K27me3_results.csv", "H3K27ac_results.csv", "H3K9me3_results.csv", "H3K4me3_results.csv"])
    