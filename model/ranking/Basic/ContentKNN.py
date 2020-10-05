# coding: utf-8

"Simple Content Filtering Recommender."

__author__ = 'Xiaodong Wang'
__email__ = 'jackwangsysu@gmail.com'

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing as mp
import os, sys, time, math

DATA_DIR = '../../dataset/movieLens/ml-10m/'
RESULT_FILE = 'ContentKNN.txt'

class ContentKNN(object):
    def __init__(self, test_size=1.0/8, N=10, K=10):
        self.test_size = test_size
        self.N, self.K = N, K

    # 读取数据并划分
    def read_and_split(self):
        # 读取交互数据
        R = []
        with open(os.path.join(DATA_DIR, 'ratings.dat'), 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split('::')[:2]
                R.append((lst[0], lst[1]))
        self.user_nums, self.item_nums = len(set(r[0] for r in R)), len(set(r[1] for r in R))
        print('user_nums: %d, item_nums: %d' % (self.user_nums, self.item_nums))
        # 读取内容数据
        self.word_items = defaultdict(dict) # 关键词-项目倒排表
        with open(os.path.join(DATA_DIR, 'movies.dat'), 'r', encoding = 'ISO-8859-1') as fc:
            for line in fc.readlines():
                lst = line.strip().split('::')
                words = lst[-1].split('|')
                for word in words:
                    self.word_items[word][lst[0]] = 1.0
        print('len(word): ', len(self.word_items))

        # Train/test split
        train, test = train_test_split(R, test_size=self.test_size)
        self.user_items_train, self.item_users_len_tr, self.user_items_test = defaultdict(list), defaultdict(int), defaultdict(list)
        for u, i in train:
            self.user_items_train[u].append(i)
            self.item_users_len_tr[i] += 1
        for u, i in test:
            self.user_items_test[u].append(i)

    # 计算项目之间的相似度 (根据关键词, 如电影类型)
    def cal_item_similarities(self):
        t1 = time.time()
        print('Start calculating item similarities...')
        # 计算关键词权重
        word_weights = dict()
        for word, item_sij in self.word_items.items():
            for item, sij in item_sij.items():
                sij /= math.log(1.0 + len(item_sij))
        S, S_d = defaultdict(dict), defaultdict(float)
        for word in self.word_items:
            for i in self.word_items[word]:
                S_d[i] += self.word_items[word][i] ** 2
                for j in self.word_items[word]:
                    if i == j:
                        continue
                    if j not in S[i]:
                        S[i][j] = 0
                    S[i][j] += self.word_items[word][i] * self.word_items[word][j]
        # 计算项目相似度
        for i in S:
            for j in S[i]:
                S[i][j] /= math.sqrt(S_d[i] * S_d[j])
        print('Calculating similarities done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
        print('项目相似度矩阵大小: %.2fK' % (sys.getsizeof(S)/1024))
        self.sorted_S = {k: sorted(v.items(), key=lambda x:x[1], reverse=True) for k, v in S.items()} # 先对S进行排序, 测试非常省时

    # 测试单个用户
    def test_one_user(self, u):
        real_items = self.user_items_test[u]
        seen_items = set(self.user_items_train[u])
        # Recommend N items to u
        scores = defaultdict(float)
        for i in seen_items:
            if i not in self.sorted_S:
                continue
            k_count = 0
            # for j, sij in sorted(S_[i].items(), key=lambda x:x[1], reverse=True): # 此时对S进行排序导致测试非常耗时
            for j, sij in self.sorted_S[i]:
                if k_count >= self.K:
                    break
                if j not in seen_items:
                    scores[j] += sij
                    k_count += 1
            # for j, sij in self.sorted_S[i][:self.K]:
            #     if j not in seen_items:
            #         scores[j] += sij
        # 查找topN items并计算评价指标
        rec_items = []
        rec_popularity = 0
        for j, score in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:self.N]:
            rec_items.append(j)
            if j in self.item_users_len_tr:
                rec_popularity += math.log(1.0 + self.item_users_len_tr[j])
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    def run_model(self):
        self.read_and_split()
        self.cal_item_similarities() # 计算项目相似度
        # 预测并评估
        t1 = time.time()
        hit_nums, real_nums, rec_popularity = 0, 0, 0
        all_rec_items = set()
        for u in self.user_items_test:
            res = self.test_one_user(u)
            hit_nums += res[0]
            real_nums += res[1]
            rec_popularity += res[2]
            for i in res[3]:
                all_rec_items.add(i)
        rec_nums = self.N * len(self.user_items_test)
        str1 = '(N=%d, K=%d) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.N, self.K, hit_nums/rec_nums*100, \
            hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))



if __name__ == '__main__':
    t1 = time.time()
    contentknn = ContentKNN()
    contentknn.run_model()
    print('cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))