import pandas as pd
import numpy as np
from plotter import create_2d_figure, create_scatter_plot_2d_polyreg, update_plot_2d_polyreg, print_legend_2d_polyreg
from regression_utils import calculate_polynomial_regression, calculate_mean_squared_error, train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataset_dataframe: pd.DataFrame = pd.read_csv(r'content/dataset_third_phase.csv', header=None)
X = dataset_dataframe[0]
y = dataset_dataframe[1]

X_train, y_train, y_test = train_test_split(X, y, test_size=0.10)
ploynomial_degree = {'red': 1, 'green': 2, 'black': 3, 'yellow': 8}
n_rows = 2
n_cols = 1

figure_data = create_2d_figure(n_rows, n_cols)
figure_data = create_scatter_plot_2d_polyreg(X, y, figure_data, n_rows)

for color, degree in ploynomial_degree.items():
    #Calcula regressão polinomial sem dividir os dados em treio e teste
    y_hat = calculate_polynomial_regression(X, y, degree)
    mse = calculate_mean_squared_error(y, y_hat)

    figure_data = update_plot_2d_polyreg(X=X, 
                                         y=y_hat, 
                                         color=color, 
                                         legend=f'MSE for {color}: {round(mse, 4)}', 
                                         figure_data=figure_data, 
                                         row=0)
    
    #Calcula a regressão polinomial dividindo os dados em treino e teste
    y_hat = calculate_polynomial_regression(X_train, y_train, degree)
    mse = calculate_mean_squared_error(y_test, y_hat)
    #r2_train = r2_score(X_train, y_train)
    #r2_test = r2_score(X_test, y_test)

    figure_data = update_plot_2d_polyreg(X=X_train, 
                                         y=y_hat, 
                                         color=color, 
                                         legend=f'MSE for {color}: {round(mse, 4)}', 
                                         figure_data=figure_data, 
                                         row=1)
    

print_legend_2d_polyreg(figure_data, n_rows)
plt.tight_layout()
plt.show()