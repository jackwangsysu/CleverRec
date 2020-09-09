# coding: utf-8

" Transform data to libFM format (Each row feature: label,id1:v1,id2:v2 ...). "

import pandas as pd
from datetime import datetime
from dateutil import rrule
from tools import re_index
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import os, math, warnings

warnings.filterwarnings('ignore')

DATA_DIR = '../dataset/'
DATASET = 'ml-1m'
FILE_NAME = 'ratings.dat'

# Read source data
ratings = pd.read_csv(os.path.join(DATA_DIR, DATASET, FILE_NAME), sep='::', header=0, names=['u_id', 'i_id', 'rating', 'time'], usecols=[0, 1, 2, 3])
ratings.time = ratings.time.astype(int)

# Reindex
user_set, item_set = set(ratings.u_id.unique()), set(ratings.i_id.unique())
user_nums, item_nums = len(user_set), len(item_set)
user_map, item_map = re_index(user_set), re_index(item_set)
ratings.u_id = ratings.u_id.map(lambda x : user_map[x]) + 1 # Start from 1
ratings.i_id = ratings.i_id.map(lambda x : item_map[x]) + 1

ratings.sort_values(['u_id', 'time'], inplace=True)
user_items = ratings.groupby('u_id').i_id.apply(list).to_dict()
max_rated_num = max(len(user_items[u]) for u in user_items)
t1, t2 = ratings['time'][0], ratings['time'][ratings.shape[0]-1]
print('最近记录日期: %s, 最远记录日期: %s' % (datetime.fromtimestamp(t1), datetime.fromtimestamp(t2)))

# Construct row features
process_nums = 32
batch_size = math.ceil(ratings.shape[0]/process_nums)
is_real_valued = True # Whether consider real-valued features

def _child_initialize(data, u_items, bsz):
    global ratings_, user_items_, batch_size_
    ratings_, user_items_, batch_size_ = data, u_items, bsz

def _get_row_features(batch_id):
    end_idx = min(ratings_.shape[0], (batch_id+1)*batch_size_)
    res = []
    for id in range(batch_id*batch_size_, end_idx):
        row_feature = []
        u, i, rating, t = ratings_['u_id'][id], ratings_['i_id'][id], ratings_['rating'][id], ratings_['time'][id]
        row_feature.append(rating)

        if is_real_valued:
            row_feature.append(str(u)+':'+str(1))
            row_feature.append(str(user_nums+i)+':'+str(1))
            
            # # Time
            # months = rrule.rrule(rrule.MONTHLY, dtstart=datetime.fromtimestamp(t2), until=datetime.fromtimestamp(t)).count()
            # row_feature.append(str(user_nums+item_nums)+':'+str(months))

            # # Last movie rated
            # u_items = user_items_[u]
            # last_movie_id = u_items.index(i) - 1
            # if last_movie_id > 0:
            #     row_feature.append(str(user_nums+2*item_nums+1+u_items[last_movie_id])+':'+str(1))
            # else:
            #     row_feature.append(str(0)+':'+str(0))

            # # Other movies rated
            # for j in u_items:
            #     row_feature.append(str(user_nums+item_nums+j)+':'+str(round(1.0/len(u_items), 4)))
            # if len(u_items) < max_rated_num:
            #     row_feature.extend([str(0)+':'+str(0)]*(max_rated_num-len(u_items)))
        else:
            row_feature.append(u)
            row_feature.append(user_nums+i)
            # row_feature.append(user_nums+item_nums)
        res.append(row_feature)
    return res

# Multi-process
pool = mp.Pool(process_nums, initializer=_child_initialize, initargs=(ratings, user_items, batch_size))
res = pool.map(_get_row_features, range(process_nums))
pool.close()
pool.join()
row_features = []
for row in res:
    row_features.extend(row)

# Train/test split
train_features, test_features = train_test_split(row_features, test_size=0.2)

# Write to file
train_features, test_features = pd.DataFrame(train_features), pd.DataFrame(test_features)
train_features.to_csv(os.path.join(DATA_DIR, DATASET, DATASET+'.train.libfm'), index=0, header=None)
test_features.to_csv(os.path.join(DATA_DIR, DATASET, DATASET+'.test.libfm'), index=0, header=None)
