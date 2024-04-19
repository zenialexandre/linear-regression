import pandas as pd
import numpy as np
from plotter import create_2d_scatter_plots_poly_reg
from regression_utils import calculate_polynomial_regression

df = pd.read_csv(r"content/data_preg.csv", header=None)



#print(create_2d_scatter_plots_poly_reg(df, "Gráfico Regressão Polinomial"))

print(calculate_polynomial_regression(df[0], df[1], 1))