import pickle

import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool

def generate_random(num = 10):
    size = [16, 32, 64, 128, 256, 512, 1024, 2048]
    result = []
    for j in range(num):
        data_set = []
        for i in size:
            A = np.random.randint(0, 256, (i, i))
            B = np.random.randint(0, 256, (i, i))
            data_set.append((A, B))
        result.append(data_set)
    with open('dataset.pkl', 'wb') as file:
        pickle.dump(result, file)

def multiprocess_strassen(X, Y, pool_size):
    if len(X) <= 2:
        return X @ Y

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


def strassen(X, Y):
    if len(X) <= 2:
        return X @ Y

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
    result = []
    for i in tqdm(range(10)):
        t_array = []
        for matrixs in tqdm(data[i]):
            A, B = matrixs
            start_time = time.perf_counter()
            _ = strassen(A, B)
            end_time = time.perf_counter()
            t = end_time - start_time
            print(f"{len(t_array):^12}|{t:^30}|{sum(t_array):^40}")
            t_array.append(t)
        result.append(t_array)
    return result


def multi_processing(data, max_pool_size):
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

    multi_result = multi_processing(data, 4)
    with open('mult_result.pkl', 'wb') as file:
        pickle.dump(multi_result, file)