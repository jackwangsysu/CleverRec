# coding: utf-8

"Simple Item-based Collaborative Filtering."

__author__ = "Xiaodong Wang"
__email__ = "jackwangsysu@gmail.com"

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing as mp
import os, sys, time

DATA_DIR = '../../dataset/movieLens/ml-1m/'
RESULT_FILE = './ItemCF.txt'

class ItemCF(object):
    def __init__(self, file_name='ratings.dat', test_size=1.0/8, N=10, K=10, sim_type='cosine', alpha=0.6, is_norm=False, process_nums=1):
        self.file_name = file_name
        self.test_size = test_size
        self.N, self.K, self.sim_type, self.alpha, self.is_norm, self.process_nums = N, K, sim_type, alpha, is_norm, process_nums

    # 读取数据并划分
    def read_and_split(self):
        R = []
        with open(os.path.join(DATA_DIR, self.file_name), 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split('::')[:2]
                R.append((lst[0], lst[1]))
        user_set, item_set = set(r[0] for r in R), set(r[1] for r in R)
        self.user_nums, self.item_nums = len(user_set), len(item_set)

        # Train/test split
        train, test = train_test_split(R, test_size=self.test_size)
        self.user_items_train, self.item_users_len_tr, self.user_items_test = defaultdict(list), defaultdict(int), defaultdict(list)
        for u, i in train:
            self.user_items_train[u].append(i)
            self.item_users_len_tr[i] += 1
        for u, i in test:
            self.user_items_test[u].append(i)

    # 计算项目之间的相似度
    def cal_item_similarities(self):
        self.read_and_split() # 预处理数据
        t1 = time.time()
        print('Start calculating %s similarities...' % self.sim_type)
        C, S = defaultdict(dict), defaultdict(dict)
        for u, items in self.user_items_train.items():
            for i in items:
                for j in items:
                    if i != j:
                        if j not in C[i]:
                            C[i][j] = 0
                        if self.sim_type == 'iuf': # IUF(Inverse User Frequence)
                            C[i][j] += 1.0 / np.log(1 + len(items))
                        else:
                            C[i][j] += 1 # 同时喜欢项目i和j的用户数
        for i, j_cij in C.items():
            for j, cij in j_cij.items():
                # S[i][j] = cij / (np.sqrt(self.item_users_len_tr[i] * self.item_users_len_tr[j]))
                # 哈利波特问题(在分母上加大对热门项目的惩罚)
                S[i][j] = cij / (np.power(self.item_users_len_tr[i], 1.0-self.alpha) * np.power(self.item_users_len_tr[j], self.alpha))
        # 归一化每行的项目相似度(在ml-1m上能带来性能提升)
        if self.is_norm:
            for i, j_sij in S.items():
                max_sim = max(j_sij.values())
                for j, sij in j_sij.items():
                    S[i][j] /= max_sim
        print('Calculating similarities done. Cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
        print('项目相似度矩阵大小: %.2fK' % (sys.getsizeof(S)/1024))
        return S

    # 测试单个用户
    def test_one_user(self, u):
        real_items = self.user_items_test[u]
        # Recommend N items to u
        scores = defaultdict(float)
        for i in self.user_items_train[u]:
            # 查找与i最相似的K个项目(不能是u已经交互过的项目), 并根据相似度进行加权求和得到u对j的偏好分数
            # 方式1：一直往后定位直到找到K个不在u的训练列表中的项目(性能明显高于方式2)
            k_count = 0
            for j, sij in sorted(S_[i].items(), key=lambda item:item[1], reverse=True):
                if k_count >= self.K:
                    break
                if j not in self.user_items_train[u]:
                    scores[j] += sij
                    k_count += 1
            '''
            # 方式2：只定位前K个，如果其在u的训练列表中则直接跳过
            for j, sij in sorted(S_[i].items(), key=lambda item:item[1], reverse=True)[0:self.K]:
                if j in self.user_items_train[u]:
                    continue
                scores[j] += sij '''
        rec_items = []
        rec_popularity = 0 # 项目流行度
        for j, score in sorted(scores.items(), key=lambda item:item[1], reverse=True)[0:self.N]:
            rec_items.append(j)
            rec_popularity += np.log(1 + self.item_users_len_tr[j])
        # 计算评价指标
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    def child_initialize(self, S):
        global S_
        S_ = S

    def run_model(self, S, K=None):
        # self.read_and_split()
        # self.cal_item_similarities()
        if K:
            self.K = K
        # 预测并评估
        t1 = time.time()
        print('Start testing...')
        hit_nums, real_nums, rec_nums, rec_popularity = 0, 0, 0, 0
        all_rec_items = set()
        if self.process_nums == 1:
            # 单线程
            for u in self.user_items_test:
                res = self.test_one_user(u)
                hit_nums += res[0]
                real_nums += res[1]
                rec_popularity += res[2]
                for i in res[3]:
                    all_rec_items.add(i)
        else:
            # 多线程
            pool = mp.Pool(self.process_nums, initializer=self.child_initialize, initargs=(S, )) # 创建进程池时传递相似度矩阵到每个子进程, 否则后续传递S会导致子进程等待
            res = pool.map(self.test_one_user, self.user_items_test.keys(), len(self.user_items_test)//self.process_nums)
            pool.close()
            pool.join()
            for r1, r2, r3, r4 in res:
                hit_nums += r1
                real_nums += r2
                rec_popularity += r3
                for i in r4:
                    all_rec_items.add(i)
        rec_nums = self.N * len(self.user_items_test)
        str1 = '(file_name=%s, N=%d, K=%d, sim_type=%s, is_norm=%s, alpha=%.2f) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.file_name, self.N, self.K, self.sim_type, \
            self.is_norm, self.alpha, hit_nums/rec_nums*100, hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))



if __name__ == '__main__':
    # # 参数寻优
    # sim_types = ['cosine', 'iuf']
    # K = [5, 10, 20, 40, 80, 160]
    # is_norms = [False, True]
    # alphas = [0.4, 0.5, 0.55, 0.6, 0.7]
    # for sim_type in sim_types:
    #     for alpha in alphas:
    #         for is_norm in is_norms:
    #             t1 = time.time()
    #             itemcf = ItemCF(sim_type=sim_type, alpha=alpha, is_norm=is_norm, process_nums=32)
    #             S = itemcf.cal_item_similarities()
    #             print('Cost time(cal_sim): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
    #             for k in K:
    #                 t2 = time.time()
    #                 itemcf.run_model(S, K=k)
    #                 print('Cost time(run_model): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t2)))

    t1 = time.time()
    itemcf = ItemCF(K=10, sim_type='cosine', alpha=0.5, is_norm=True, process_nums=32)
    S = itemcf.cal_item_similarities()
    itemcf.run_model(S)
    print('Cost time(run_model): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))