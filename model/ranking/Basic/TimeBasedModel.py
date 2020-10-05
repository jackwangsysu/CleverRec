# coding: utf-8

"Time-based Model (RecentPop, TItemCF, TUserCF, 时间段图模型SGM)."

__author__ = 'Xiaodong Wang'
__email__ = 'jackwangsysu@gmail.com'

import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import os, time, math

DATA_DIR = '../../dataset/movieLens/ml-1m/'
RESULT_FILE = 'TimeBasedModel.txt'

class Preprocess(object):
    def __init__(self, test_size=1.0/8):
        self.test_size = test_size
    
    # 读取数据并划分
    def read_and_split(self):
        user_items = defaultdict(list)
        t0, t1 = 0, float('inf')
        item_set = set()
        with open(os.path.join(DATA_DIR, 'ratings.dat'), 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split('::')
                u, i, t = lst[0], lst[1], float(lst[3])
                user_items[u].append((i, t))
                item_set.add(i)
                t0, t1 = max(t0, t), min(t1, t)
        user_nums, item_nums = len(user_items), len(item_set)
        print('user_nums: %d, item_nums: %d' % (user_nums, item_nums))
        print('最近记录日期: %s, 最远记录日期: %s' % (datetime.fromtimestamp(t0), datetime.fromtimestamp(t1))) # 最近2003-03-01 01:49:50
        
        # Train/test split
        train, train_iu, test, item_users_len_tr = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
        for u, its in user_items.items():
            its = sorted(its, key=lambda x:x[1])
            s_idx = math.ceil((1.0 - self.test_size) * len(its))
            train[u] = its[:s_idx]
            for i, t in its[:s_idx]:
                train_iu[i].append((u, t))
            test[u] = its[s_idx:]
        print('len(train): %d, len(test): %d' % (sum([len(train[u]) for u in train]), sum(len(test[u]) for u in test)))
        for u, its in train.items():
            for i, t in its:
                item_users_len_tr[i] += 1
        return train, train_iu, test, item_users_len_tr, item_nums, t0

# 最近最热门
class RecentPop(object):
    def __init__(self, N=10, alpha=1.0, t0=time.time(), train=None, test=None, item_users_len_tr=None, item_nums=0):
        self.N, self.alpha, self.t0 = N, alpha, t0
        self.train, self.test, self.item_users_len_tr, self.item_nums = train, test, item_users_len_tr, item_nums
        self.model = 'RecentPop'

    # 测试单个用户
    def test_one_user(self, u):
        real_items = [r[0] for r in self.test[u]]
        seen_items = set(r[0] for r in self.train[u])
        rec_items = []
        k_count, rec_popularity = 0, 0
        for i, score in self.sorted_item_pops:
            if k_count >= self.N:
                break
            if i not in seen_items:
                rec_items.append(i)
                rec_popularity += math.log(1.0 + self.item_users_len_tr[i])
                k_count += 1
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    def run_model(self):
        t1 = time.time()
        print('Start RecentPop...')
        item_pops = defaultdict(float)
        for u, its in self.train.items():
            for i, t in its:
                item_pops[i] += 1.0 / (1.0 + self.alpha * (self.t0 - t))
                # item_pops[i] += 1.0 # MostPopular
        self.sorted_item_pops = sorted(item_pops.items(), key=lambda x:x[1], reverse=True)
        # 预测并评估
        hit_nums, real_nums, rec_popularity = 0, 0, 0
        all_rec_items = set()
        for u in self.test:
            res = self.test_one_user(u)
            hit_nums += res[0]
            real_nums += res[1]
            rec_popularity += res[2]
            for i in res[3]:
                all_rec_items.add(i)
        rec_nums = self.N * len(self.test)
        str1 = '(model=%s, alpha=%.2f) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.model, self.alpha, \
            hit_nums/rec_nums*100, hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('RecentPop done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))

# 时间上下文相关的ItemCF算法TItemCF
class TItemCF(object):
    def __init__(self, N=10, K=10, alpha=1.0, beta=1.0, t0=time.time(), sim_type='cosine', train=None, test=None, item_users_len_tr=None, item_nums=0, \
        process_nums=2):
        self.N, self.K, self.alpha, self.beta, self.t0, self.sim_type = N, K, alpha, beta, t0, sim_type
        self.train, self.test, self.item_users_len_tr, self.item_nums = train, test, item_users_len_tr, item_nums
        self.process_nums = process_nums
        self.model = 'TItemCF'

    # 计算项目之间的相似度
    def cal_item_similarities(self):
        t_ = time.time()
        print('Calculating item similarities...')
        S = defaultdict(dict)
        for u, its in self.train.items():
            for i, t1 in its:
                for j, t2 in its:
                    if i == j:
                        continue
                    if j not in S[i]:
                        S[i][j] = 0
                    if self.sim_type == 'cosine':
                        # S[i][j] += 1.0 / (1.0 + self.alpha * abs(t1 - t2))
                        S[i][j] += 1.0
        for i in S:
            for j in S[i]:
                S[i][j] /= math.sqrt(self.item_users_len_tr[i] * self.item_users_len_tr[j])
        # Normalize the rows
        for i in S:
            max_i = max(S[i].values())
            for j in S[i]:
                S[i][j] /= max_i
        sorted_S = {k: sorted(v.items(), key=lambda x:x[1], reverse=True) for k, v in S.items()}
        print('Item similarities done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t_)))
        return sorted_S

    # 测试单个用户
    def test_one_user(self, u):
        real_items = [r[0] for r in self.test[u]]
        seen_items = set(r[0] for r in self.train[u])
        scores = defaultdict(float)
        for i, ti in self.train[u]:
            k_count = 0
            for j, sij in sorted_S_[i]:
                if k_count >= self.K:
                    break
                if j not in seen_items:
                    # scores[j] += sij / (1.0 + self.beta * abs(self.t0 - ti))
                    scores[j] += sij
                    k_count += 1
        rec_items = []
        rec_popularity = 0
        for i, score in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:self.N]:
            rec_items.append(i)
            rec_popularity += math.log(1.0 + self.item_users_len_tr[i])
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    # 初始化进程池
    def child_initialize(self, sorted_S):
        global sorted_S_
        sorted_S_ = sorted_S

    def run_model(self):
        sorted_S = self.cal_item_similarities()
        t1 = time.time()
        print('Start testing TItemCF...')
        # 预测并评估
        hit_nums, real_nums, rec_popularity = 0, 0, 0
        all_rec_items = set()
        pool = mp.Pool(self.process_nums, initializer=self.child_initialize, initargs=(sorted_S, ))
        res = pool.map(self.test_one_user, self.test.keys(), len(self.test)//self.process_nums)
        pool.close()
        pool.join()
        for r1, r2, r3, r4 in res:
            hit_nums += r1
            real_nums += r2
            rec_popularity += r3
            for i in r4:
                all_rec_items.add(i)
        rec_nums = self.N * len(self.test)
        str1 = '(model=%s, K=%d, sim_type=%s) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.model, self.K, self.sim_type, \
            hit_nums/rec_nums*100, hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))

# 时间上下文相关的UserCF算法TUserCF
class TUserCF(object):
    def __init__(self, N=10, K=10, alpha=1.0, beta=1.0, t0=time.time(), sim_type='cosine', train=None, train_iu=None, test=None, item_users_len_tr=None, \
        item_nums=0, process_nums=2):
        self.N, self.K, self.alpha, self.beta, self.t0, self.sim_type = N, K, alpha, beta, t0, sim_type
        self.train, self.train_iu, self.test, self.item_users_len_tr, self.item_nums = train, train_iu, test, item_users_len_tr, item_nums
        self.process_nums = process_nums
        self.model = 'TUserCF'

    # 计算用户之间的相似度
    def cal_user_similarities(self):
        t1 = time.time()
        print('Start calculating user similarities...')
        S = defaultdict(dict)
        for i, uts in self.train_iu.items():
            for u, t1 in uts:
                for v, t2 in uts:
                    if u == v:
                        continue
                    if v not in S[u]:
                        S[u][v] = 0
                    if self.sim_type == 'cosine':
                        S[u][v] += 1.0 / (1.0 + self.alpha * abs(t1 - t2))
        for u in S:
            for v in S[u]:
                S[u][v] /= math.sqrt(len(self.train[u]) * len(self.train[v]))
        # Normalize the rows
        for u in S:
            max_u = max(S[u].values())
            for v in S[u]:
                S[u][v] /= max_u
        sorted_S = {k: sorted(v.items(), key=lambda x:x[1], reverse=True) for k, v in S.items()}
        print('User similarities done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))
        return sorted_S

    def test_one_user(self, u):
        real_items = [r[0] for r in self.test[u]]
        seen_items = set(r[0] for r in self.train[u])
        rec_items = []
        k_count, rec_popularity = 0, 0
        scores = defaultdict(float)
        for v, suv in sorted_Su_[u][:self.K]:
            for i, tvi in self.train[v]:
                if i not in seen_items:
                    scores[i] += suv / (1.0 + self.beta * abs(self.t0 - tvi))
        for i, score in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:self.N]:
            rec_items.append(i)
            rec_popularity += math.log(1.0 + self.item_users_len_tr[i])
        hit_nums = len(set(real_items) & set(rec_items))
        return hit_nums, len(real_items), rec_popularity, rec_items

    def child_initialize(self, sorted_S):
        global sorted_Su_
        sorted_Su_ = sorted_S

    def run_model(self):
        sorted_S = self.cal_user_similarities()
        t1 = time.time()
        print('Start testing TUserCF...')
        # 预测并评估
        hit_nums, real_nums, rec_popularity = 0, 0, 0
        all_rec_items = set()
        pool = mp.Pool(self.process_nums, initializer=self.child_initialize, initargs=(sorted_S, ))
        res = pool.map(self.test_one_user, self.test.keys(), len(self.test)//self.process_nums)
        pool.close()
        pool.join()
        for r1, r2, r3, r4 in res:
            hit_nums += r1
            real_nums += r2
            rec_popularity += r3
            for i in r4:
                all_rec_items.add(i)
        rec_nums = self.N * len(self.test)
        str1 = '(model=%s, K=%d, sim_type=%s) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.model, self.K, self.sim_type, \
            hit_nums/rec_nums*100, hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))

# 时间段图模型
class SGM(object):
    def __init__(self, N=10, alpha=0.5, beta=0.5, train=None, test=None, item_users_len_tr=None, item_nums=0, process_nums=2):
        self.N, self.alpha, self.beta = N, alpha, beta
        self.train, self.test, self.item_users_len_tr, self.item_nums = train, test, item_users_len_tr, item_nums
        self.process_nums = process_nums
        self.model = 'SGM'

    # 构建时间段图
    def construct_graph(self):
        G = defaultdict(dict)
        for u, its in self.train.items():
            for i, t in its:
                G['user_'+u]['item_'+i] = 1.0 # 所有边权均为1
                G['user_'+u+'_'+str(t)]['item_'+i] = 1.0
                G['user_'+u]['item_'+i+'_'+str(t)] = 1.0
        return G

    # 路径融合算法 (为单个用户进行推荐)
    def path_fusion_u(self, u):
        pass

    def run_model(self):
        G = self.construct_graph()


if __name__ == '__main__':
    t1 = time.time()
    preprocess = Preprocess()
    train, train_iu, test, item_users_len_tr, item_nums, t0 = preprocess.read_and_split()
    # RecentPop
    rp = RecentPop(alpha=1.0, t0=t0, train=train, test=test, item_users_len_tr=item_users_len_tr, item_nums=item_nums)
    rp.run_model()
    # TItemCF
    ticf = TItemCF(K=10, alpha=1.0, beta=1.0, t0=t0, sim_type='cosine', train=train, test=test, item_users_len_tr=item_users_len_tr, \
        item_nums=item_nums, process_nums=16)
    ticf.run_model()
    # TUserCF
    tucf = TUserCF(K=80, alpha=1.0, beta=1.0, t0=t0, sim_type='cosine', train=train, train_iu=train_iu, test=test, \
        item_users_len_tr=item_users_len_tr, item_nums=item_nums, process_nums=16)
    tucf.run_model()
