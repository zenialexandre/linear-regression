import pandas as pd
import numpy as np
from plotter import create_2d_plots_poly_reg
from regression_utils import calculate_polynomial_regression, calculate_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset_dataframe: pd.DataFrame = pd.read_csv(r'content/dataset_third_phase.csv', header=None)
X = dataset_dataframe[0]
y = dataset_dataframe[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

ploynomial_degree = {'red': 1, 'green': 2, 'black': 3, 'yellow': 8}
regression_without_sep_results = {}
regression_with_sep_results = {}

for degree in ploynomial_degree.items():
    y_hat_without_sep = calculate_polynomial_regression(X, y, degree[1])
    mse_without_sep = calculate_mean_squared_error(y, y_hat_without_sep)

    y_hat_with_sep = calculate_polynomial_regression(X_train, y_train, degree[1])
    mse_with_sep = calculate_mean_squared_error(y_test, y_hat_with_sep)
    r2_train = r2_score(X_train, y_train)
    r2_test = r2_score(X_test, y_test)

    regression_without_sep_results[degree[0]] = {'pred_values': y_hat_without_sep, 'mse': mse_without_sep, 'title': 'Regressão polinomial antes da divisão'}
    regression_with_sep_results[degree[0]] = {'pred_values': y_hat_with_sep, 'mse': mse_with_sep, 'title': 'Regressão polinomial após divisão','r2_train': r2_train, 'r2_test': r2_test}

    create_2d_plots_poly_reg(X, y, [regression_without_sep_results, regression_with_sep_results])
    
'''
result_reg_1 = calculate_polynomial_regression(dataset_dataframe[0], dataset_dataframe[1])
result_reg_2 = calculate_polynomial_regression(dataset_dataframe[0], dataset_dataframe[1], 2)
result_reg_3 = calculate_polynomial_regression(dataset_dataframe[0], dataset_dataframe[1], 3)
result_reg_4 = calculate_polynomial_regression(dataset_dataframe[0], dataset_dataframe[1], 8)



'''