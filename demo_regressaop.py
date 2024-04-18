import pandas as pd
import numpy as np
from plotter import create_polynomial_regression_plot

dataset_dataframe: pd.DataFrame = pd.read_csv(r'content/dataset_third_phase.csv', header=None)
dataset_dataframe_x: pd.Series = dataset_dataframe.iloc[:, 0]
dataset_dataframe_y: pd.Series = dataset_dataframe.iloc[:, 1]

create_polynomial_regression_plot(
    dataset_dataframe_x,
    dataset_dataframe_y
)
