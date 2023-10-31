import pickle
import numpy as np

if __name__ == '__main__':
    with open('mult_result.pkl', 'rb') as file:
        mult_result = pickle.load(file)

    with open('seq_result.pkl', 'rb') as file:
        seq_result = pickle.load(file)

    process_num = 8

    for i in range(10):
        for j in range(8):
            mult_result[i][j] = [seq_result[i][j]] + mult_result[i][j]

    print(f'Processing time of G0 under sequential implementation')
    print(f"|{'Pair index':^12}|{'Measured Sequential Time':^30}|{'Measured Cumulative Sequential time':^40}|")
    print(f"|{'-'*12}|{'-'*30}|{'-'*40}|")
    for i in range(8):
        print(f"|{i:^12}|{seq_result[0][i]:^30}|{sum(seq_result[0][:i+1]):^40}|")
    print()

    print(f'Processing time of all groups under sequential implementation')
    print(f"|{'Group index':^12}|{'Measured Sequential Time':^30}|{'Measured Cumulative Sequential time':^40}|")
    print(f"|{'-' * 12}|{'-' * 30}|{'-' * 40}|")
    for i in range(10):
        print(f"|{i:^12}|{sum(seq_result[i]):^30}|{sum([sum(i) for i in seq_result[:i+1]]):^40}|")
    print()

    print(f'Processing time of G0 under multiprocessing implementation')
    print(f"|{'':^20}|{'Measured MP Time':^{(process_num * 25 - 1)}}|{'Measured Cumulative MP time':^{(process_num * 25 - 1)}}|")
    print(f"|{'-' * 20}|{'-' * (process_num * 25 - 1)}|{'-' * (process_num * 25 - 1)}|")
    row = ''.join([f"{i:^24}|" for i in [j for j in range(1, 1 + process_num)] * 2])
    print(f"|{'pair Index/Process':^20}|{row}")
    print(f"|{'-' * 20}|{('-' * 24 + '|') * process_num}{('-' * 24 + '|') * process_num}")
    for i in range(8):
        measured_MP_time_str = ''.join([f"{j:^24}|" for j in mult_result[0][i]])
        measured_cumulative_time = [sum([mult_result[0][j][k] for j in range(i + 1)]) for k in range(process_num)]
        measured_cumulative_time_str = ''.join([f"{j:^24}|" for j in measured_cumulative_time])
        print(f"|{i:^20}|{measured_MP_time_str}{measured_cumulative_time_str}")
    print()

    print(f'Processing time of all groups under multiprocessing implementation')
    print(f"|{'':^20}|{'Measured MP Time':^{(process_num * 25 - 1)}}|{'Measured Cumulative MP time':^{(process_num * 25 - 1)}}|")
    print(f"|{'-' * 20}|{'-' * (process_num * 25 - 1)}|{'-' * (process_num * 25 - 1)}|")
    row = ''.join([f"{i:^24}|" for i in [j for j in range(1, 1 + process_num)] * 2])
    print(f"|{'Grp Index/Process':^20}|{row}")
    print(f"|{'-' * 20}|{('-' * 24 + '|') * process_num}{('-' * 24 + '|') * process_num}")
    measured_MP_time_list = []
    for i in range(10):
        measured_MP_time = [sum([mult_result[i][j][k]for j in range(8)]) for k in range(process_num)]
        measured_MP_time_str = ''.join([f"{j:^24}|" for j in measured_MP_time])
        measured_MP_time_list.append(measured_MP_time)
        measured_cumulative_time = [sum([measured_MP_time_list[j][k] for j in range(i + 1)]) for k in range(process_num)]
        measured_cumulative_time_str = ''.join([f"{j:^24}|" for j in measured_cumulative_time])
        print(f"|{i:^20}|{measured_MP_time_str}{measured_cumulative_time_str}")