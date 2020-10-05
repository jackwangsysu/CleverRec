# coding: utf-8

"LFM: Simple Latent Factor Model."

__author__ = 'Xiaodong Wang'
__email__ = 'jackwangsysu@gmail.com'

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing as mp
import os, time

DATA_DIR = '../../dataset/movieLens/ml-1m/'
RESULT_FILE = './LFM.txt'


class LFM(object):
    def __init__(self, file_name='ratings.dat', test_size=1.0/8, N=10, F=10, lambda_=0.01, alpha=0.02, neg_ratio=1, iter_nums=10, process_nums=1):
        self.file_name = file_name
        self.test_size = test_size
        self.N, self.F, self.lambda_, self.alpha, self.neg_ratio, self.iter_nums, self.process_nums = N, F, lambda_, alpha, neg_ratio, iter_nums, process_nums

    # 读取数据并划分
    def read_and_split(self):
        R = []
        with open(os.path.join(DATA_DIR, self.file_name), 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split('::')[:2]
                R.append((lst[0], lst[1]))
        self.user_set, self.item_set = set(r[0] for r in R), set(r[1] for r in R)
        self.user_nums, self.item_nums = len(self.user_set), len(self.item_set)
        print('user_nums: %d, item_nums: %d' % (self.user_nums, self.item_nums))

        # Train/test split
        train, test = train_test_split(R, test_size=self.test_size)
        self.test_rating_nums = len(test)
        self.user_items_train, self.item_users_len_train, self.user_items_test = defaultdict(list), defaultdict(int), defaultdict(list)
        for u, i in train:
            self.user_items_train[u].append(i)
            self.item_users_len_train[i] += 1
        for u, i in test:
            self.user_items_test[u].append(i)
        self.all_train_items = list(self.item_users_len_train.keys())
        self.all_p = [self.item_users_len_train[i] for i in self.all_train_items] # 项目流行度(概率)

    # 初始化模型参数
    def init_model_params(self):
        self.P, self.Q = dict(), dict()
        for u in self.user_set:
            self.P[u] = np.random.random(self.F)
        for i in self.item_set:
            self.Q[i] = np.random.random(self.F)

    # 生成训练样本
    def sample_instances(self):
        t1 = time.time()
        all_train_items = list(self.item_users_len_train.keys())
        all_p = [self.item_users_len_train[i] for i in all_train_items] # 项目流行度(概率)
        user_train_items = defaultdict(dict)
        for u, items in self.user_items_train.items():
            # 正样本
            for i in items:
                user_train_items[u][i] = 1.0
            # 负样本(根据流行度进行采样)
            neg_items = np.random.choice(all_train_items, 3*self.neg_ratio*len(items), all_p)
            neg_count = 0
            for j in neg_items:
                if neg_count >= self.neg_ratio*len(items):
                    break
                if j not in items:
                    user_train_items[u][j] = 0.0
                    neg_count += 1
        print('cost time(sample): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
        return user_train_items

    # 生成训练样本(多进程)
    def sample_instances_u(self, u):
        u_train_items = dict()
        items = self.user_items_train[u]
        # 正样本
        for i in items:
            u_train_items[i] = 1.0
        # 负样本
        neg_items = np.random.choice(self.all_train_items, 3*self.neg_ratio*len(items), self.all_p)
        neg_count = 0
        for j in neg_items:
            if neg_count >= self.neg_ratio*len(items):
                break
            if j not in items:
                u_train_items[j] = 0.0
                neg_count += 1
        return u, u_train_items

    # 训练模型
    def train_model(self):
        t1 = time.time()
        print('Start training...')
        for step in range(self.iter_nums):
            t2 = time.time()
            # 构造训练样本
            # user_train_items = self.sample_instances()
            pool = mp.Pool(self.process_nums)
            res = pool.map(self.sample_instances_u, self.user_items_train.keys(), len(self.user_items_train)//self.process_nums)
            pool.close()
            pool.join()
            user_train_items = dict()
            for u, u_train_items in res:
                user_train_items[u] = u_train_items
            print('  cost time(sample): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t2)))
            for u in self.user_items_train:
                for i, rui in user_train_items[u].items():
                    # 计算预测误差
                    res = rui - self.P[u].dot(self.Q[i])
                    # 更新参数
                    # for k in range(self.F): # 使用defaultdict(dict)表示P和Q时更新速度很慢
                    #     self.P[u][k] += self.alpha * (res * self.Q[i][k] - self.lambda_ * self.P[u][k])
                    #     self.Q[i][k] += self.alpha * (res * self.P[u][k] - self.lambda_ * self.Q[i][k])
                    self.P[u] += self.alpha * (res * self.Q[i] - self.lambda_ * self.P[u])
                    self.Q[i] += self.alpha * (res * self.P[u] - self.lambda_ * self.Q[i])
            self.alpha *= 0.9 # 每轮迭代后降低学习率
            print('  step %2d done, cost time: %s' % (step+1, time.strftime('%H:%M:%S', time.gmtime(time.time() - t2))))
        print('Training done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))

    # 测试单个用户
    def test_one_user(self, u):
        real_items = self.user_items_test[u]
        rec_popularity  = 0
        # Recommend N items to u
        scores = defaultdict(float)
        # tmp_items = self.item_set - set(self.user_items_train[u])
        tmp_items = set(self.item_users_len_train.keys()) - set(self.user_items_train[u])
        for i in tmp_items:
            scores[i] = self.P[u].dot(self.Q[i])
        # 查找topN items
        rec_items = []
        for j, score in sorted(scores.items(), key=lambda item:item[1], reverse=True)[:self.N]:
            rec_items.append(j)
            if j in self.item_users_len_train:
                rec_popularity += np.log(1 + self.item_users_len_train[j])
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    # 测试模型
    def test_model(self):
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
            pool = mp.Pool(self.process_nums)
            res = pool.map(self.test_one_user, self.user_items_test.keys(), len(self.user_items_test)//self.process_nums)
            for r1, r2, r3, r4 in res:
                hit_nums += r1
                real_nums += r2
                rec_popularity += r3
                for i in r4:
                    all_rec_items.add(i)
        rec_nums = self.N * len(self.user_items_test)
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
        str1 = '(file_name=%s, N=%d, F=%d, neg_ratio=%d) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.file_name, self.N, self.F, self.neg_ratio, \
            hit_nums/rec_nums*100, hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')

    def run_model(self):
        self.read_and_split()
        self.init_model_params()
        self.train_model()
        self.test_model()



if __name__ == '__main__':
    t1 = time.time()
    lfm = LFM(F=100, neg_ratio=10, iter_nums=10, process_nums=16)
    lfm.run_model()
    print('Cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))