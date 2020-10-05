# coding: utf-8

"Simple User-based Collaborative Filtering."

__author__ = "Xiaodong Wang"
__email__ = "jackwangsysu@gmail.com"

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing as mp
import os, sys, time

DATA_DIR = '../../dataset/movieLens/ml-1m/'
RESULT_FILE = './UserCF.txt'

class UserCF(object):
    def __init__(self, file_name='ratings.dat', test_size=1.0/8, N=10, K=10, sim_type='cosine', is_norm=False, process_nums=1):
        self.file_name = file_name
        self.test_size = test_size
        self.N, self.K, self.sim_type, self.is_norm, self.process_nums = N, K, sim_type, is_norm, process_nums
    
    # 读取数据并划分
    def read_and_split(self):
        R = []
        with open(os.path.join(DATA_DIR, self.file_name), 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split('::')[:2]
                R.append((lst[0], lst[1]))
        self.user_nums, self.item_nums = len(set(r[0] for r in R)), len(set(r[1] for r in R))

        # Train/test split
        train, test = train_test_split(R, test_size=self.test_size)
        user_items_train, item_users_train, user_items_test = defaultdict(list), defaultdict(list), defaultdict(list)
        for u, i in train:
            user_items_train[u].append(i)
            item_users_train[i].append(u) # items倒排表
        for u, i in test:
            user_items_test[u].append(i)
        return user_items_train, item_users_train, user_items_test

    # 计算用户之间的兴趣相似度 (行为相似度)
    # 只计算相似度不为0的部分
    def cal_user_similarities(self, user_items_train, item_users_train):
        t1 = time.time()
        print("Start calculating %s similarities..." % self.sim_type)
        C, S = defaultdict(dict), defaultdict(dict)
        for i, users in item_users_train.items():
            for u in users:
                for v in users:
                    if u != v:
                        if v not in C[u]:
                            C[u][v] = 0
                        if self.sim_type == 'iif': # IIF(Inverse Item Frequence)
                            C[u][v] += 1.0 / np.log(1 + len(users))
                        else:
                            C[u][v] += 1 # 同时被用户u和v喜欢的项目数
        for u, v_cuv in C.items():
            for v, cuv in v_cuv.items():
                if self.sim_type == 'jacard': # Jacard相似度
                    S[u][v] = cuv / (len(user_items_train[u]) + len(user_items_train[v])-cuv)
                else: # 余弦相似度 / iif改进相似度
                    S[u][v] = cuv / (np.sqrt(len(user_items_train[u]) * len(user_items_train[v])))
        print("Calculating similarities done. Cost time: %s" % (time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))))
        print('用户相似度矩阵大小: %.2fK' % (sys.getsizeof(S)/1024))
        return S

    # 测试单个用户
    def test_one_user(self, u):
        real_items = user_items_test_[u]
        rec_popularity = 0 # 项目流行度
        # Recommend N items to u
        scores = defaultdict(float)
        for v, suv in sorted(S_[u].items(), key=lambda item:item[1], reverse=True)[0:self.K]:
            for i in user_items_train_[v]:
                if i not in user_items_train_[u]:
                    scores[i] += suv
        rec_items = []
        for i, score in sorted(scores.items(), key=lambda item:item[1], reverse=True)[0:self.N]:
            rec_items.append(i)
            rec_popularity += np.log(1 + len(item_users_train_[i]))
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    def child_initialize(self, S, user_items_train, item_users_train, user_items_test):
        global S_, user_items_train_, item_users_train_, user_items_test_
        S_, user_items_train_, item_users_train_, user_items_test_ = S, user_items_train, item_users_train, user_items_test

    def run_model(self, S, user_items_train, item_users_train, user_items_test, K=None):
        # self.read_and_split()
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
            pool = mp.Pool(self.process_nums, initializer=self.child_initialize, initargs=(S, user_items_train, item_users_train, user_items_test)) # 创建进程池时传递相似度矩阵到每个子进程, 否则后续传递S会导致子进程等待
            res = pool.map(self.test_one_user, user_items_test.keys(), len(user_items_test)//self.process_nums)
            pool.close()
            pool.join()
            for r1, r2, r3, r4 in res:
                hit_nums += r1
                real_nums += r2
                rec_popularity += r3
                for i in r4:
                    all_rec_items.add(i)
        rec_nums = self.N * len(user_items_test)
        str1 = '(file_name=%s, N=%d, K=%d, sim_type=%s) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.file_name, self.N, self.K, self.sim_type, \
            hit_nums/rec_nums*100, hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))


if __name__ == '__main__':
    # # 参数寻优
    # sim_types = ['cosine', 'jacard', 'iif']
    # K = [5, 10, 20, 30, 50, 80, 100, 120, 140]
    # for sim_type in sim_types:
    #     t1 = time.time()
    #     usercf = UserCF(sim_type=sim_type, process_nums=32)
    #     S = usercf.cal_user_similarities()
    #     print('Cost time(cal_sim): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
    #     for k in K:
    #         t2 = time.time()
    #         usercf.run_model(S, K=k)
    #         print('Cost time(run_model): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t2)))

    t1 = time.time()
    usercf = UserCF(K=80, sim_type='cosine', process_nums=32) # 性能最优
    user_items_train, item_users_train, user_items_test = usercf.read_and_split()
    S = usercf.cal_user_similarities(user_items_train, item_users_train)
    usercf.run_model(S, user_items_train, item_users_train, user_items_test)
    print('Cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))