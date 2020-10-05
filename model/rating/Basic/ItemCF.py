# coding: utf-8

"Simple Item-based Collaborative Filtering"

__author__ = "Xiaodong Wang"
__email__ = "jackwangsysu@gmail.com"

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os, time

DATA_DIR = '../../../dataset/ml-1m/'

# Load source data
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', names=['u_id', 'i_id', 'rating'], usecols=[0, 1, 2], \
    dtype={0:np.int32, 1:np.int32, 2:np.float32}, engine='python')
print('user_nums: %d, item_nums: %d' % (len(ratings.u_id.unique()), len(ratings.i_id.unique())))

# Train/test split
train_data, test_data = train_test_split(ratings, test_size=1/8)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Generate ratings dict
R, Rt = defaultdict(dict), defaultdict(dict)
for id in range(train_data.shape[0]):
    u, i, r = train_data['u_id'][id], train_data['i_id'][id], train_data['rating'][id]
    R[u][i] = r
    Rt[i][u] = r
print('Ratings dict done.')

# 每个用户/项目的平均评分(训练阶段)
r_u_avg, r_i_avg = dict() , dict()
for u in R:
    r_u_avg[u] = sum(R[u].values()) / len(R[u])
for i in Rt:
    r_i_avg[i] = sum(Rt[i].values()) / len(Rt[i])
# 全局平均评分
r_global = train_data.rating.mean()
print('Mean ratings done.')

# 删除train_data中的rating列
del train_data['rating']
user_items_train = train_data.groupby('u_id').i_id.apply(list).to_dict()


# 计算项目之间的相似度
def cal_item_similarities(sim_type='cosine', is_norm=False):
    t1 = time.time()
    print('Start calculating %s similarities...' % sim_type)
    C, C_d, S = defaultdict(dict), dict(), defaultdict(dict)
    for u, items in user_items_train.items():
        for i in items:
            if i not in C_d:
                C_d[i] = 0
            tmp = R[u][i] - r_u_avg[u] if sim_type == 'adjust_cosine' else (R[u][i] - r_i_avg[i] if sim_type == 'pcc' else R[u][i])
            C_d[i] += np.square(tmp)
            for j in items:
                if i != j:
                    if j not in C[i]:
                        C[i][j] = 0
                    if sim_type == 'cosine': # 余弦相似度
                        C[i][j] += R[u][i] * R[u][j]
                    elif sim_type == 'adjust_cosine': # 修正的余弦相似度
                        C[i][j] += (R[u][i] - r_u_avg[u]) * (R[u][j] - r_u_avg[u])
                    elif sim_type == 'pcc': # 皮尔逊系数
                        C[i][j] += (R[u][i] - r_i_avg[i]) * (R[u][j] - r_i_avg[j])
    max_sim = 0
    for i, j_cij in C.items():
        for j, cij in j_cij.items():
            S[i][j] = C[i][j] / (np.sqrt(C_d[i] * C_d[j]))
            if S[i][j] > max_sim:
                max_sim = S[i][j]
    if is_norm:
        for i, j_sij in S.items():
            for j, sij in j_sij.items():
                S[i][j] /= max_sim
    print('Calculating similarities done. Cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
    return S


def run_model(N=10, sim_type='cosine', K=10, is_norm=False):
    # 计算项目相似度
    S = cal_item_similarities(sim_type=sim_type, is_norm=is_norm)
    # 预测并评估
    t1 = time.time()
    abs_sum, square_sum = 0, 0
    for id in range(test_data.shape[0]):
        u, i, r = test_data['u_id'][id], test_data['i_id'][id], test_data['rating'][id]
        r_ui_pre = 0
        # 查找i最相似的K个项目并计算加权平均评分
        k_sim_sum = 0
        k_count = 0
        for j, sij in sorted(S[i].items(), key=lambda item:item[1], reverse=True):
            if k_count >= K:
                break
            if j in user_items_train[u]:
                r_ui_pre += sij * R[u][j]
                k_sim_sum += sij
                k_count += 1
        if k_count == 0:
            if i in r_i_avg:
                r_ui_pre = r_i_avg[i]
            else:
                r_ui_pre = r_global
        else:
            r_ui_pre = r_ui_pre/k_sim_sum # /k_sim_sum将K个相似项目的权重归一化
        res = r - r_ui_pre
        abs_sum += np.abs(res)
        square_sum += np.square(res)
    print('RMSE: %.4f, MAE: %.4f.' % (np.sqrt(square_sum/test_data.shape[0]), abs_sum/test_data.shape[0]))
    print('Cost time(predict) %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))



if __name__ == '__main__':
    t1 = time.time()
    run_model(sim_type='cosine', K=10, is_norm=False)
    print('Cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))