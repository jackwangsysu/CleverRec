# coding: utf-8

"Simple Tag-based Recommender."

__author__ = 'Xiaodong Wang'
__email__ = 'jackwangsysu@gmail.com'

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing as mp
import time, os, sys, math

DATA_DIR = '../../dataset/Delicious/'
RESULT_FILE = './SimpleTagBased.txt'

class TagBasedModel(object):
    def __init__(self, test_size=0.1, N=10, process_nums=2, model='SimpleTagBased'):
        self.test_size = test_size
        self.N, self.process_nums = N, process_nums
        self.model = model # ['SimpleTagBased', 'TagBasedTFIDF', 'TagBasedTFIDF++']

    # 读取数据并划分
    def read_and_split(self):
        R, tmp_R = [], defaultdict(set)
        with open(os.path.join(DATA_DIR, 'user_taggedbookmarks.dat'), 'r') as fr:
            for line in fr.readlines()[1:]:
                lst = line.strip().split('\t')[:3]
                tmp_R[(lst[0], lst[1])].add(lst[2])
        for k, v in tmp_R.items():
            R.append((k[0], k[1], v))
        user_set, item_set = set(r[0] for r in R), set(r[1] for r in R)
        self.user_nums, self.item_nums, self.tag_nums = len(user_set), len(item_set), sum([len(r[2]) for r in R])
        print('user_nums: %d, item_nums: %d, tag_nums: %d' % (self.user_nums, self.item_nums, self.tag_nums))

        # Train/test split
        train, test = train_test_split(R, test_size=self.test_size, random_state=2020) # 分割的键值是user和item, 不包括tag
        # print('len(train): %d, len(test): %d' % (len(train), len(test)))
        user_tags, tag_items, item_tags, tag_users, user_items_train, self.item_users_len_tr, self.user_items_test = defaultdict(dict), defaultdict(dict), \
            defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(int), defaultdict(set)
        for u, i, tags in train:
            for t in tags:
                if t not in user_tags[u]:
                    user_tags[u][t] = 0
                user_tags[u][t] += 1
                tag_users[t].add(u)
                if i not in tag_items[t]:
                    tag_items[t][i] = 0
                tag_items[t][i] += 1
                item_tags[i].add(t)
            user_items_train[u].add(i)
            self.item_users_len_tr[i] += 1
        tag_users_len = {k: len(v) for k, v in tag_users.items()} # 标签被多少个不同用户打过
        for u, i, tags in test:
            self.user_items_test[u].add(i)
        return user_tags, tag_items, item_tags, user_items_train, tag_users_len

    # 计算标签之间的相似度
    def cal_tag_similarities(self, tag_items, item_tags):
        t1_ = time.time()
        print('Start calculating tag similarities...')
        S, S_d = defaultdict(dict), defaultdict(float)
        for item, tags in item_tags.items():
            for t1 in tags:
                S_d[t1] += len(tag_items[t1]) ** 2
                for t2 in tags:
                    if t1 == t2:
                        continue
                    if t2 not in S[t1]:
                        S[t1][t2] = 0
                    S[t1][t2] += len(tag_items[t1]) * len(tag_items[t2])
        for t1, t2_w12 in S.items():
            for t2, w12 in t2_w12.items():
                S[t1][t2] = w12 / math.sqrt(S_d[t1] * S_d[t2])
        # Normalize the rows
        for t1, t2_w12 in S.items():
            max_t1 = max(t2_w12.values())
            for t2, w12 in t2_w12.items():
                S[t1][t2] = w12 / max_t1
        print('Calculating tag similarities done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1_)))
        print('标签相似度大小: %.2fK' % (sys.getsizeof(S)/1024))
        return S

    # 标签扩展 (标签不足20个的用户, 查找其最相似的20个标签进行扩展)
    def extend_tags(self, u):
        expanded_tags = defaultdict(float)
        seen_tags = set(user_tags_[u].keys())
        for tag, wut in user_tags_[u].items():
            for t, w12 in S_[tag].items():
                if t in seen_tags:
                    continue
                if t not in expanded_tags:
                    expanded_tags[t] = 0
                expanded_tags[t] += wut * w12
        expanded_tags.update(user_tags_[u])
        expanded_tags = dict(sorted(expanded_tags.items(), key=lambda x:x[1], reverse=True)[:20])
        return expanded_tags

    # 测试单个用户
    def test_one_user(self, u):
        real_items = self.user_items_test[u]
        seen_items = user_items_train_[u] if u in user_items_train_ else set()
        # 查找topN items
        scores = defaultdict(int)
        u_tags = self.extend_tags(u) if len(user_tags_[u]) < 20 else user_tags_[u]
        for tag, wut in u_tags.items(): # 用户u打过的所有标签
            for item, wti in tag_items_[tag].items(): # 与这些标签相关的项目
                if item in seen_items:
                    continue
                if self.model == 'TagBasedTFIDF': # 惩罚热门标签
                    scores[item] += wut * wti / math.log(1.0 + tag_users_len_[tag])
                elif self.model == 'TagBasedTFIDF++': # 同时惩罚热门标签和热门项目
                    scores[item] += wut * wti / (math.log(1.0 + tag_users_len_[tag]) * math.log(1.0 + self.item_users_len_tr[item]))
                else: # SimpleTagBased / TagExtended
                    scores[item] += wut * wti
        rec_items = set()
        rec_popularity = 0
        for j, score in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:self.N]:
            rec_items.add(j)
            if j in self.item_users_len_tr:
                rec_popularity += math.log(1.0 + self.item_users_len_tr[j])
        hit_nums = len(real_items & rec_items)
        return hit_nums, len(real_items), rec_popularity, rec_items

    # 初始化进程池参数
    def child_initialize(self, user_tags, tag_items, user_items_train, tag_users_len, S):
        global user_tags_, tag_items_, user_items_train_, tag_users_len_, S_
        user_tags_, tag_items_, user_items_train_, tag_users_len_, S_ = user_tags, tag_items, user_items_train, tag_users_len, S

    def run_model(self):
        user_tags, tag_items, item_tags, user_items_train, tag_users_len = self.read_and_split()
        S = self.cal_tag_similarities(tag_items, item_tags)
        print('Start testing...')
        hit_nums, real_nums, rec_popularity = 0, 0, 0
        all_rec_items = set()
        # 多线程
        pool = mp.Pool(self.process_nums, initializer=self.child_initialize, initargs=(user_tags, tag_items, user_items_train, tag_users_len, S))
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
        str1 = '(model=%s, N=%d) 准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (self.model, self.N, hit_nums/rec_nums*100, \
            hit_nums/real_nums*100, len(all_rec_items)/self.item_nums*100, rec_popularity/rec_nums)
        print(str1)
        with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
            fw.write(str1 + '\n')
        print('Testing done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))



if __name__ == '__main__':
    t1 = time.time()
    models = ['SimpleTagBased', 'TagBasedTFIDF', 'TagBasedTFIDF++', 'TagExtended']
    for model in models:
        tbd = TagBasedModel(process_nums=32, model=model)
        tbd.run_model()
    print('cost time(total): %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))