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

'''
data_1 = np.array(
    [[10,8,13,9,11,14,6,4,12,7,5],
     [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]
    ]
)

print(corelation_coefficient(data_1))


data_2 = np.array(
    [[10,8,13,9,11,14,6,4,12,7,5],
     [9.14,8.14,8.47,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74],
    ]
)

print(corelation_coefficient(data_2))
'''

data_3 = np.array(
    [[8,8,8,8,8,8,8,8,8,8,19],
     [6.58,5.76,7.71,8.84,8.47,7.04,5.25,5.56,7.91,6.89,12.50],
    ]
)

print(corelation_coefficient(data_3))