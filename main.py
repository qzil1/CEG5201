import pickle

import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool

def generate_random(group_size = 10):
    """
    Generate the input matrix set
    :param group_size: the number of groups
    :return: [[(A,B) for different size] for different group]
    """
    size = [16, 32, 64, 128, 256, 512, 1024, 2048]
    result = []
    for j in range(group_size):
        data_set = []
        for i in size:
            A = np.random.randint(0, 256, (i, i))
            B = np.random.randint(0, 256, (i, i))
            data_set.append((A, B))
        result.append(data_set)
    with open('dataset.pkl', 'wb') as file:
        pickle.dump(result, file)

def matrix_multiply(X, Y):
    """
    Directly multiple two matrices
    :param X: Matrix X in NumPy array type
    :param Y: Matrix Y in NumPy array type
    :return: the multiple of two matrices
    """
    rows_X, cols_X = X.shape
    rows_Y, cols_Y = Y.shape
    if cols_X != rows_Y:
        raise ValueError('The column number of X must be equal to that of Y')
    result = np.zeros((rows_X, cols_Y))
    for i in range(rows_X):
        for j in range(cols_Y):
            for k in range(cols_X):
                result[i, j] += X[i, k] * Y[k, j]
    return result
def multiprocess_strassen(X, Y, pool_size):
    """
    Multiprocessing version for strassen algorithm
    :param X: Matrix X in NumPy array type
    :param Y: Matrix Y in NumPy array type
    :param pool_size: the number of available processes in the pool
    :return: the multiply result of X and Y
    """
    if len(X) <= 2:
        return matrix_multiply(X, Y)

    half_size = len(X) // 2
    X11, X12, X21, X22 = X[:half_size, :half_size], X[:half_size, half_size:], X[half_size:, :half_size], X[half_size:, half_size:]
    Y11, Y12, Y21, Y22 = Y[:half_size, :half_size], Y[:half_size, half_size:], Y[half_size:, :half_size], Y[half_size:, half_size:]
    with Pool(pool_size) as pool:
        async_results = [
            pool.apply_async(strassen, (X11 + X22, Y11 + Y22)),
            pool.apply_async(strassen, (X12 + X22, Y21 + Y22)),
            pool.apply_async(strassen, (X11 - X22, Y11 + Y22)),
            pool.apply_async(strassen, (X11, Y12 - Y22)),
            pool.apply_async(strassen, (X21 + X22, Y11)),
            pool.apply_async(strassen, (X11 + X12, Y22)),
            pool.apply_async(strassen, (X22, Y21 - Y11))
        ]
        M1, M2, M3, M4, M5, M6, M7 = [result.get() for result in async_results]

    Z11 = M2 + M3 - M6 - M7
    Z12 = M4 + M6
    Z21 = M5 + M7
    Z22 = M1 - M3 - M4 - M5

    Z = np.vstack((np.hstack((Z11, Z12)), np.hstack((Z21, Z22))))
    return Z

def strassen(X, Y):
    """
    Single-processing version for strassen algorithm
    :param X: Matrix X in NumPy array type
    :param Y: Matrix Y in NumPy array type
    :return: the multiply result of X and Y
    """
    if len(X) <= 2:
        return matrix_multiply(X, Y)

    half_size = len(X) // 2
    X11, X12, X21, X22 = X[:half_size, :half_size], X[:half_size, half_size:], X[half_size:, :half_size], X[half_size:, half_size:]
    Y11, Y12, Y21, Y22 = Y[:half_size, :half_size], Y[:half_size, half_size:], Y[half_size:, :half_size], Y[half_size:, half_size:]

    M1 = strassen(X11 + X21, Y11 + Y12)
    M2 = strassen(X12 + X22, Y21 + Y22)
    M3 = strassen(X11 - X22, Y11 + Y22)
    M4 = strassen(X11, Y12 - Y22)
    M5 = strassen(X21 + X22, Y11)
    M6 = strassen(X11 + X12, Y22)
    M7 = strassen(X22, Y21 - Y11)

    Z11 = M2 + M3 - M6 - M7
    Z12 = M4 + M6
    Z21 = M5 + M7
    Z22 = M1 - M3 - M4 - M5

    Z = np.vstack((np.hstack((Z11, Z12)), np.hstack((Z21, Z22))))
    return Z


def sequential_processing(data):
    """
    Experiment the sequential matrix multiply
    :param data: the input matrix set
    :return: [[time_consumed for different_size] for different_group]
    """
    result = []
    for i in tqdm(range(10)):
        t_array = []
        for matrixs in tqdm(data[i]):
            A, B = matrixs
            start_time = time.perf_counter()
            _ = strassen(A, B)
            end_time = time.perf_counter()
            t = end_time - start_time
            t_array.append(t)
        result.append(t_array)
    return result


def multi_processing(data, max_pool_size):
    """
    Experiment the multiprocessing matrix multiply
    :param data: the input matrix set
    :param max_pool_size: the maximum size for the process pool
    :return: [[[time_consumed for different_process_number] for different_size] for different_group]
    """
    result = []
    for i in tqdm(range(10)):
        group_time_array = []
        for matrixs in tqdm(data[i]):
            A, B = matrixs
            single_time_array = []
            for k in range(2, max_pool_size + 1):
                start_time = time.perf_counter()
                _ = multiprocess_strassen(A, B, k)
                end_time = time.perf_counter()
                t = end_time - start_time
                single_time_array.append(t)
            group_time_array.append(single_time_array)
        result.append(group_time_array)
    return result


if __name__ == "__main__":
    with open('dataset.pkl', 'rb') as file:
        data = pickle.load(file)

    sequential_result = sequential_processing(data)
    with open('seq_result.pkl', 'wb') as file:
        pickle.dump(sequential_result, file)

    multi_result = multi_processing(data, 8)
    with open('mult_result.pkl', 'wb') as file:
        pickle.dump(multi_result, file)
