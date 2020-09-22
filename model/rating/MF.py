# coding: utf-8

" FunkSVD: Simple Matrix Factorization "

__author__ = "Xiaodong Wang"
__email__ = "jackwangsysu@gmail.com"

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os, time

DATA_DIR = '../../dataset/ml-1m/'

# Load source data
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', names=['u_id', 'i_id', 'rating'], usecols=[0, 1, 2], \
    dtype={0:np.int32, 1:np.int32, 2:np.float32}, engine='python')
print('user_nums: %d, item_nums: %d' % (len(ratings.u_id.unique()), len(ratings.i_id.unique())))

# Train/test split
train_data, test_data = train_test_split(ratings, test_size=1/8)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Generate ratings dict
R, Rt, R_test = defaultdict(dict), defaultdict(dict), defaultdict(dict)
for id in range(train_data.shape[0]):
    u, i, r = train_data['u_id'][id], train_data['i_id'][id], train_data['rating'][id]
    R[u][i] = r
    Rt[i][u] = r
for id in range(test_data.shape[0]):
    u, i, r = test_data['u_id'][id], test_data['i_id'][id], test_data['rating'][id]
    R_test[u][i] = r
print('Ratings dict done.')

# 每个用户/项目的平均评分
r_u_avg, r_i_avg = dict(), dict()
for u in R:
    r_u_avg[u] = sum(R[u].values()) / len(R[u])
for i in Rt:
    r_i_avg[i] = sum(Rt[i].values()) / len(Rt[i])
print('Mean ratings done.')

# 删除rating列
del train_data['rating']
item_users_train = train_data.groupby('i_id').u_id.apply(list).to_dict() # items倒排表


# 计算用户之间的兴趣相似度 (行为相似度)
# 只计算相似度不为0的部分
def cal_user_similarities(sim_type='cosine'):
    t1 = time.time()
    print("Start calculating %s similarities..." % sim_type)
    C, C_d, S = defaultdict(dict), dict(), defaultdict(dict)
    for i, users in item_users_train.items():
        for u in users:
            if u not in C_d:
                C_d[u] = 0
            tmp = R[u][i] - r_i_avg[i] if sim_type == 'adjust_cosine' else (R[u][i] - r_u_avg[u] if sim_type == 'pcc' else R[u][i])
            C_d[u] += np.square(tmp)
            for v in users:
                if u != v:
                    if v not in C[u]:
                        C[u][v] = 0
                    if sim_type == 'cosine': # 余弦相似度
                        C[u][v] += R[u][i] * R[v][i]
                    elif sim_type == 'adjust_cosine': # 修正的余弦相似度
                        C[u][v] += (R[u][i] - r_i_avg[i]) * (R[v][i] - r_i_avg[i])
                    elif sim_type == 'pcc': # 皮尔逊系数
                        C[u][v] += (R[u][i] - r_u_avg[u]) * (R[v][i] - r_u_avg[v])
    for u, v_cuv in C.items():
        for v, cuv in v_cuv.items():
            S[u][v] = cuv / np.sqrt(C_d[u] * C_d[v])
    print("Calculating similarities done. Cost time: %s" % (time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))))
    return S


def run_model(N=10, sim_type='cosine', K=10):
    # 计算用户相似度
    S = cal_user_similarities(sim_type=sim_type)
    # 预测并评估
    abs_sum, square_sum = 0, 0
    for id in range(test_data.shape[0]):
        u, i = test_data['u_id'][id], test_data['i_id'][id]
        r_ui_pre = 0
        k_sim_sum = 0
        k_count = 0
        for v, suv in sorted(S[u].items(), key=lambda item:item[1], reverse=True):
            if k_count >= K:
                break
            if i in item_users_train and v in item_users_train[i]:
                r_ui_pre += suv * R[v][i]
                k_sim_sum += suv
                k_count += 1
        r_ui_pre = r_u_avg[u] if k_count == 0 else r_ui_pre/k_sim_sum
        res = R_test[u][i] - r_ui_pre
        abs_sum += np.abs(res)
        square_sum += np.square(res)
    print('RMSE: %.4f, MAE: %.4f.' % (np.sqrt(square_sum/test_data.shape[0]), abs_sum/test_data.shape[0]))



if __name__ == '__main__':
    t1 = time.time()
    run_model(sim_type='cosine', K=10) # sim_types = ['cosine', 'jacard', 'iif']
    print('Cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
