# Coding: utf-8

"Non-personalized Recommender (Random, MostPopular)."

__author__ = "Xiaodong Wang"
__email__ = "jackwangsysu@gmail.com"

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os, time, math

DATA_DIR = '../../dataset/movieLens/ml-1m/'
RESULT_FILE = './NonPersonalizedModel.txt'

# Load source data
R = []
with open(os.path.join(DATA_DIR, 'ratings.dat'), 'r') as fr:
    for line in fr.readlines():
        lst = line.strip().split('::')[:2]
        R.append((lst[0], lst[1]))
user_nums, item_nums = len(set(r[0] for r in R)), len(set(r[1] for r in R))
print('user_nums: %d, item_nums: %d' % (user_nums, item_nums))

# Train/test split
train, test = train_test_split(R, test_size=1.0/8)
user_items_train, item_users_len_tr, user_items_test = defaultdict(list), defaultdict(int), defaultdict(list)
for u, i in train:
    user_items_train[u].append(i)
    item_users_len_tr[i] += 1
for u, i in test:
    user_items_test[u].append(i)

# Random
def run_random(N=10):
    t1 = time.time()
    print('Start Random...')
    hit_nums, real_nums, rec_popularity = 0, 0, 0
    all_rec_items = set()
    train_items = list(item_users_len_tr.keys())
    train_item_nums = len(train_items)
    # 给用户随机推荐其未交互过的所有项目中的N个
    for u, real_items in user_items_test.items():
        u_train_items = set(user_items_train[u])
        real_nums += len(real_items)
        rec_items = []
        random_j = []
        for s in range(N):
            j = np.random.randint(train_item_nums)
            while j in random_j or train_items[j] in u_train_items:
                j = np.random.randint(train_item_nums)
            random_j.append(j)
            i = train_items[j] # 待推荐项目
            rec_items.append(i)
            all_rec_items.add(i)
            rec_popularity += math.log(1 + item_users_len_tr[i])
        hit_nums += len(set(real_items) & set(rec_items))
    rec_nums = N * len(user_items_test)
    str1 = '(model=Random)准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (hit_nums/rec_nums*100, hit_nums/real_nums*100, \
        len(all_rec_items)/item_nums*100, rec_popularity/rec_nums)
    print(str1)
    with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
        fw.write(str1 + '\n')
    print('Random done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))

# Most Popular
def run_most_popular(N=10):
    t1 = time.time()
    print('Start MostPopular...')
    # 给用户推荐其未交互过的项目中最热门的N个项目, 不同用户可能不同
    # 所有项目按流行度排序
    sorted_items = sorted(item_users_len_tr.items(), key=lambda item:item[1], reverse=True)
    hit_nums, real_nums, rec_popularity = 0, 0, 0
    all_rec_items = set()
    for u, real_items in user_items_test.items():
        u_items_train = set(user_items_train[u])
        real_nums += len(real_items)
        # 计算rec_items
        rec_items = []
        rec_count = 0
        for i, users_len in sorted_items:
            if rec_count >= 10:
                break
            if i not in u_items_train:
                rec_items.append(i)
                all_rec_items.add(i)
                rec_popularity += math.log(1 + users_len) # 项目流行度(取对数后流行度的平均值更加平稳)
                rec_count += 1
        hit_nums += len(set(real_items) & set(rec_items))
    rec_nums = N * len(user_items_test)
    str1 = '(Model=MostPopular)准确率: %.2f%%, 召回率: %.2f%%, 覆盖率: %.2f%%, 流行度: %.4f' % (hit_nums/rec_nums*100, hit_nums/real_nums*100, \
        len(all_rec_items)/item_nums*100, rec_popularity/rec_nums)
    print(str1)
    with open(RESULT_FILE, 'a', encoding='utf-8') as fw:
        fw.write(str1 + '\n')
    print('MostPopular done, cost time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)))



if __name__ == '__main__':
    t1 = time.time()
    run_random()
    run_most_popular()
    print('Cost time(total): %s' % (time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))))

