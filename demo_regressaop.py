import pandas as pd
import numpy as np
from plotter import create_2d_scatter_plots_poly_reg
from regression_utils import calculate_polynomial_regression

dataset_dataframe: pd.DataFrame = pd.read_csv(r'content/dataset_third_phase.csv', header=None)