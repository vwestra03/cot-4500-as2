import numpy as np
import scipy.linalg 
from scipy.linalg import solve

# Question 1 Code

def nevilles_method(x_points, y_points, approximated_x):
    size = len(x_points)
    matrix = np.zeros((size, size))


    for index, row in enumerate(matrix):
        row[0] = y_points[index]
    
    num_of_points = len(x_points)
    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            first_multiplication = (approximated_x - x_points[i - j]) * matrix[i][j - 1]
            second_multiplication = (approximated_x - x_points[i]) * matrix[i - 1][j - 1]


            denominator = x_points[i] - x_points[i - j]


            coefficient = (first_multiplication - second_multiplication) / denominator


            matrix[i][j] = coefficient
    
    # print(matrix, "\n") - check, comment out later
    print(matrix[num_of_points - 1][num_of_points - 1], "\n")


# Question 2 Code 

def newton_method_and_approx():
    x1 = 7.2
    x2 = 7.4
    x3 = 7.5
    x4 = 7.6
    f_x1 = 23.5492
    f_x2 = 25.3913
    f_x3 = 26.8224
    f_x4 = 27.4589
    first_dd_1 = (f_x2 - f_x1) / (x2 - x1)
    first_dd_2 = (f_x3 - f_x2) / (x3 - x2)
    first_dd_3 = (f_x4 - f_x3) / (x4 - x3)
    second_dd_1 = (first_dd_2 - first_dd_1) / (x3 - x1)
    second_dd_2 = (first_dd_3 - first_dd_2) / (x4 - x2)
    third_dd = (second_dd_2 - second_dd_1) / (x4 - x1)
    d = [first_dd_1, second_dd_1, third_dd]
    print(d, "\n")
    
    # Question 3 Code
    approx_x = 7.3
    p_x = f_x1 + first_dd_1 * (approx_x - x1) + second_dd_1 * (approx_x - x2) * (approx_x - x1)\
          + third_dd * (approx_x - x3) * (approx_x - x2) * (approx_x - x1)
    print(p_x, "\n")

# Question 4 Code

np.set_printoptions(precision=7, suppress=True, linewidth=100)
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue      
            left: float = matrix[i][j - 1]
            diagonal_left: float = matrix[i - 1][j - 1]
            numerator: float = left - diagonal_left
            denominator = matrix[i][0] - matrix[i - j + 1][0]
            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix

def hermite_interpolation(x_points, y_points, slopes):
    num_of_points = len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))

    for i, x in enumerate(x_points):
        matrix[2 * i][0] = x
        matrix[2 * i + 1][0] = x
    
    for i, y in enumerate(y_points):
        matrix[2 * i][1] = y
        matrix[2 * i + 1][1] = y

    for i, slope in enumerate(slopes):
        matrix[2 * i + 1][2] = slope

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)

# Question 5 Code 

def cubic_spline_matrix(x, y):
    size = len(x)
    matrix: np.array = np.zeros((size, size))
    matrix[0][0] = 1
    matrix[1][0] = x[1] - x[0]
    matrix[1][1] = 2 * ((x[1] - x[0]) + (x[2] - x[1]))
    matrix[1][2] = x[2] - x[1]
    matrix[2][1] = x[2] - x[1]
    matrix[2][2] = 2 * ((x[3] - x[2]) + (x[2] - x[1]))
    matrix[2][3] = x[3] - x[2]
    matrix[3][3] = 1
    print(matrix, "\n")

    c0 = 0
    c1 = ((3 / (x[2] - x[1])) * (y[2] - y[1])) - ((3 / (x[1] - x[0])) * (y[1] - y[0]))
    c2 = ((3 / (x[3] - x[2])) * (y[3] - y[2])) - ((3 / (x[2] - x[1])) * (y[2] - y[1]))
    c3 = 0
    c = np.array([c0, c1, c2, c3])
    print(c, "\n")

    f = [[matrix]]
    g = [[c0], [c1], [c2], [c3]]

    h = solve(f, g)

    print(h.T[0], "\n")

# This allows to the code to print and where all the above functions are called

if __name__ == "__main__":
    np.set_printoptions(precision = 7, suppress = True, linewidth = 100)
    
    # Question 1, prints out to terminal
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximated_x = 3.7 
    nevilles_method(x_points, y_points, approximated_x)


    # Questions 2 and 3, respectively, prints out to terminal
    newton_method_and_approx()


    # Question 4, prints to terminal
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    hermite_interpolation(x_points,y_points,slopes)


    # Question 5, prints to terminal
    x = [2, 5, 8, 10]
    y = [3, 5, 7, 9]
    cubic_spline_matrix(x, y)