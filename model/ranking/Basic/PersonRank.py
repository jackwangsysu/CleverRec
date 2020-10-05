# coding: utf-8

"PersonRank: Simple Graph-based Recommendation Model."

__author__ = "Xiaodong Wang"
__email__ = "jackwangsysu@gmail.com"

import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import multiprocessing as mp
import os, time

DATA_DIR = '../../dataset/movieLens/ml-100k/'
RESULT_FILE = './PersonRank.txt'


class PersonRank(object):
    def __init__(self, file_name='u.data', test_size=1/8, alpha=0.5, iter_nums=10, N=10, process_nums=1):
        self.file_name =file_name
        self.test_size = test_size
        self.alpha = alpha # 继续游走的概率(阻尼系数)
        self.iter_nums, self.N, self.process_nums = iter_nums, N, process_nums

    # 读取数据并划分
    def read_and_split(self):
        R = []
        user_map, item_map = dict(), dict()
        u_id, i_id = 0, 0
        with open(os.path.join(DATA_DIR, self.file_name), 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split('\t')[:2]
                if lst[0] not in user_map:
                    user_map[lst[0]] = u_id
                    u_id += 1
                if lst[1] not in item_map:
                    item_map[lst[1]] = i_id
                    i_id += 1
                R.append((user_map[lst[0]], item_map[lst[1]]))
        self.user_nums, self.item_nums = u_id + 1, i_id + 1
        self.shape_ = self.user_nums + self.item_nums
        
        # Train/test split
        train, test = train_test_split(R, test_size=self.test_size)
        self.G, self.user_items_train, self.item_user_len_tr, self.user_items_test = defaultdict(dict), defaultdict(list), defaultdict(int), defaultdict(list)
        for u, i in train:
            self.G[u][i+self.user_nums] = 1.0
            self.G[i+self.user_nums][u] = 1.0
            self.user_items_train[u].append(i)
            self.item_user_len_tr[i] += 1
        for u, i in test:
            self.user_items_test[u].append(i)
        print('Preprocess done.')

    def evaluate_u(self, u, rank):
        args_u = np.argsort(-rank) # 按访问概率排序
        # 查找topN items
        u_items_train = [] if u not in self.user_items_train else self.user_items_train[u]
        real_items = self.user_items_test[u]
        rec_items = []
        k_count, rec_popularity = 0, 0
        for arg in args_u:
            if k_count >= self.N:
                break
            i = arg - self.user_nums
            if  i >= 0 and i not in u_items_train:
                k_count += 1
                rec_items.append(i)
                if i in self.item_user_len_tr:
                    rec_popularity += np.log(1 + self.item_user_len_tr[i])
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    # 针对目标用户root，迭代计算每个顶点的访问概率
    def PersonRank_u(self, root):
        rank = np.zeros(self.shape_) # 每个顶点的访问概率
        rank[root] = 1.0
        # 在二分图上进行迭代直到每个顶点的访问概率收敛
        for step in range(self.iter_nums):
            tmp_rank = np.zeros(self.shape_)
            for i, j_wij in self.G.items():
                for j, wij in j_wij.items():
                    tmp_rank[j] += self.alpha * rank[i] / len(j_wij)
                    if j == root:
                        tmp_rank[j] += 1.0 - self.alpha
            rank = tmp_rank
        return self.evaluate_u(root, rank)

    # 计算(1-alpha)*(I-alpha*M.T)^(-1)，其中M为转移概率矩阵
    def cal_transitive_matrix(self):
        t1 = time.time()
        print('Start calculating transitive matrix...')
        data, row, col = [], [], []
        for i in self.G:
            for j in self.G[i]:
                data.append(1.0 / len(self.G[i]))
                row.append(i)
                col.append(j)
        M = sp.csc_matrix((data, (row, col)), shape=(self.shape_, self.shape_))
        self.M_inv = (1 - self.alpha) * sp.linalg.inv(sp.eye(self.shape_) - self.alpha * M.T)
        print('Calculating done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))

    # 使用矩阵形式加速(无需迭代) (参考: https://blog.csdn.net/weixin_42057852/article/details/102967894)
    def PersonRank_u_mat(self, u):
        rank = np.zeros(self.shape_)
        rank[u] = 1.0
        rank = self.M_inv.dot(rank)
        return self.evaluate_u(u, rank)

    # 训练并测试
    def run_model(self):
        self.read_and_split()
        self.cal_transitive_matrix()
        # 预测并评估
        t1 = time.time()
        print('Start testing...')
        # 多线程
        pool = mp.Pool(self.process_nums)
        # res = pool.map(self.PersonRank_u, self.user_items_test.keys(), len(self.user_items_test)//self.process_nums)
        res = pool.map(self.PersonRank_u_mat, self.user_items_test.keys(), len(self.user_items_test)//self.process_nums)
        pool.close()
        pool.join()
        hit_nums, real_nums, rec_popularity = 0, 0, 0
        all_rec_items = set()
        for r1, r2, r3, r4 in res:
            hit_nums += r1
            real_nums += r2
            rec_popularity += r3
            for i in r4:
                all_rec_items.add(i)
        rec_nums = self.N * len(self.user_items_test)
        str1 = '准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f (alpha=%.2f)' % (hit_nums/rec_nums*100, hit_nums/real_nums*100, \
            len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums, self.alpha)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))



if __name__ == '__main__':
    t1 = time.time()
    pr = PersonRank(alpha=0.8, iter_nums=10, process_nums=32)
    pr.run_model()
    print('Cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))