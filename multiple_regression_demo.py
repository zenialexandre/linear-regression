import pandas as pd
from regression_utils import correlation_coefficient, linear_regression
from plotter import create_plots
import numpy as np

data = pd.read_csv("data.csv", header=None)
pd.DataFrame.describe(data)

## tamanho, quartos, preço
prices = data[2]
##print(round(prices.mean(), 2))
##print(prices.min())
most_expensive_house_value = prices.max()
most_expensive_house_rooms = data[data[2] == most_expensive_house_value][1]
##print(most_expensive_house_rooms)

matrix_x = data.iloc[:, 0:2].values
vector_y = data.iloc[:, 2].values
##print(matrix_x)
#print(vector_y)

price_size_list = [data[0].values.tolist(), data[2].values.tolist()]
correlation_size_price = correlation_coefficient(price_size_list)
price_rooms_list = [data[1].values.tolist(), data[2].values.tolist()]
correlation_room_price = correlation_coefficient(price_rooms_list)
#print(correlation_size_price)
#print(correlation_room_price)

(_, first_dataset_matrix_beta_1, first_dataset_matrix_beta_0) = linear_regression(price_size_list)
(_, second_dataset_matrix_beta_1, second_dataset_matrix_beta_0) = linear_regression(price_rooms_list)

len_a = len(price_rooms_list[0])
list_a = []

for x in range(len_a):
    list_a.append(1)

price_size_list.insert(0, list_a)

data.insert(loc=0, column="initial", value=1)

matrix_x = data.iloc[:, 0:3].values
vector_y = data.iloc[:, 3].values
##print(matrix_x)
#print(vector_y)

matrix_x_transposed = np.transpose(matrix_x)

beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(matrix_x_transposed, matrix_x)), matrix_x_transposed), vector_y)
#print(beta)

ret = np.matmul(matrix_x, beta)
print(len(ret))

#create_plots([np.array(price_size_list), np.array(price_rooms_list)])
