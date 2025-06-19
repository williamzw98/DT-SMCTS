import numpy as np
import pandas as pd


def greedy(p, x, packages, cargos):

    per_tar = 100 / cargos

    num_opt = np.ones(packages)
    cargos_extra = cargos - packages

    mse_delta = np.zeros(packages)

    while cargos_extra > 0:
        for i in range(packages):
            if num_opt[i] == 6:
                mse_delta[i] = 0
            else:
                mse_delta[i] = num_opt[i] * np.power(np.abs(p[i] / num_opt[i] - per_tar), x) - (num_opt[i]+1) * np.power(np.abs(p[i] / (num_opt[i] + 1) - per_tar), x)
                mse_delta[i] = np.max(mse_delta[i], 0)

        p_argmax = np.argsort(-mse_delta)[0]
        num_opt[p_argmax] += 1
        cargos_extra -= 1
    return num_opt


def count_commas_in_strings(strings_array):

    results = []
    for string in strings_array:
        comma_count = string.count(',')
        results.append(comma_count+1)
    return results


def read_csv_column(file_path, selected_column):
    """
    从CSV文件中读取指定列的数据并转换为NumPy数组。

    Parameters:
    - file_path (str): CSV文件路径
    - selected_column (str): 要提取的列名称

    Returns:
    - np_array (numpy.ndarray): 包含选定列数据的NumPy数组
    """


    data_frame = pd.read_csv(file_path)


    if selected_column not in data_frame.columns:
        raise ValueError(f"Error: Column '{selected_column}' not found in the CSV file.")
    else:

        selected_data = data_frame[selected_column].values
        np_array = np.array(selected_data)

        return np_array


def standard(array):
    p = array
    p_sum = sum(p)
    p = float(100/p_sum) * p
    return p


def create_csv_file(file_path, column_names, arrays):
    """
    创建一个CSV文件并将不同的NumPy数组作为不同的列插入其中。

    Parameters:
    - file_path (str): 要创建的CSV文件路径
    - column_names (list): 包含列名称的列表
    - arrays (list): 包含NumPy数组的列表，每个数组对应一个列

    Returns:
    - None
    """

    if len(column_names) != len(arrays):
        raise ValueError("Error: Number of column names and arrays must be the same.")


    data_frame = pd.DataFrame()


    for col_name, array in zip(column_names, arrays):
        data_frame[col_name] = array


    data_frame.to_csv(file_path, index=False)
    print(f"CSV file '{file_path}' created successfully.")


def compare(array1, array2):
    n1 = array1
    n2 = array2
    result = True
    for i in range(np.size(n1)):
        if n1[i] != n2[i]:
            result = False
            break
    return result


def filter_and_save_csv(input_file, output_file_include, output_file_exclude, column_name, values_to_include):

    df = pd.read_csv(input_file)


    included_rows = df[df[column_name].isin(values_to_include)]
    excluded_rows = df[~df[column_name].isin(values_to_include)]


    included_rows.to_csv(output_file_include, index=False)
    excluded_rows.to_csv(output_file_exclude, index=False)
