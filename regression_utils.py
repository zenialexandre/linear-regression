import numpy as np
import math

def corelation_coefficient(array):
    len_array = len(array[0])
    x_mean = np.mean(array[0])
    y_mean = np.mean(array[1])
    sum_numerator = 0
    sum_denominator_x = 0
    sum_denominator_y = 0

    for idx in range(len_array):
        sum_numerator += (array[0][idx] - x_mean)*(array[1][idx] - y_mean)
        sum_denominator_x += (array[0][idx] - x_mean)**2
        sum_denominator_y += (array[1][idx] - y_mean)**2

    return sum_numerator / math.sqrt(sum_denominator_x * sum_denominator_y)


def linear_regression(array):

    x_mean = np.mean(array[0])
    y_mean = np.mean(array[1])

    beta_1 = caculate_beta_1(array, x_mean, y_mean)
    beta_0 = calculate_beta_0(beta_1, x_mean, y_mean)

    y_hat = []
    for x in array[0]:
        y_hat.append(beta_0 + beta_1*x)
        
    return y_hat

def caculate_beta_1(array, x_mean, y_mean):

    len_array = len(array[0])

    sum_numerator = 0
    sum_dominator = 0

    for idx in range(len_array):
        sum_numerator += (array[0][idx] - x_mean)*(array[1][idx] - y_mean)
        sum_dominator += (array[0][idx] - x_mean)**2
    
    
    return round(sum_numerator / sum_dominator,4)

def calculate_beta_0(beta_1, x_mean, y_mean):
    return round(y_mean - beta_1*x_mean,4)
