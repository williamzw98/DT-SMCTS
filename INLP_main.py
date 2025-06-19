import numpy as np
import module as md
import pandas as pd

cons_packages = ['T77  T29兜底', '高价值', '-']


md.filter_and_save_csv('./cuhksz/data.csv', './data_cons.csv', './data_opt.csv', '虚拟包牌名称', cons_packages)

name_ori = md.read_csv_column('./data_opt.csv', '虚拟包牌名称')
cargos_ori = md.read_csv_column('./data_opt.csv', '格口')
p_ori = md.read_csv_column('./data_opt.csv', '百分比')
num_ori = np.array((md.count_commas_in_strings(cargos_ori)))

name_cons = md.read_csv_column('./data_cons.csv', '虚拟包牌名称')
cargos_cons = md.read_csv_column('./data_cons.csv', '格口')
p_cons = md.read_csv_column('./data_cons.csv', '百分比')
num_cons = np.array((md.count_commas_in_strings(cargos_cons)))

name_all = np.append(name_ori, name_cons)
p_all = np.append(p_ori, p_cons)
num_all = np.append(num_ori, num_cons)

column_names = []
column_names.append('虚拟包牌名称')
column_names.append('百分比')
column_names.append('原格口数量')

arrays = []
arrays.append(name_all)
arrays.append(p_all)
arrays.append(num_all)

p = md.standard(p_ori)

packages = np.size(name_ori)
cargos = np.sum(num_ori)
last_result = num_all
for k in range(2, 15):
    opt_result_i = np.append(md.greedy(p, k,  packages, cargos), num_cons)
    if not md.compare(opt_result_i, last_result):
        column_names.append('k={}'.format(k))
        arrays.append(opt_result_i)
        last_result = opt_result_i

md.create_csv_file('./opt_result.csv', column_names, arrays)
