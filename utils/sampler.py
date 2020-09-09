# coding: utf-8

" Sample instances for training. "

import numpy as np
import math
from utils.tools import timer

# Get training instances for pointwise learning
def pointwise_ranking_sampler(data, neg_ratio, batch_size, fism_like=False):
    u_features, i_features, labels = [], [], []
    if fism_like:
        u_neighbors_num = []
    for u, items in data.ui_train.items():
        seen_items = set(data.ui_train[u])
        for i in items:
            # Positive instances
            u_features.append(u)
            i_features.append(i)
            labels.append(1.0)
            # Sample negative items
            random_j = set()
            for s in range(neg_ratio):
                j = np.random.randint(data.item_nums)
                while j in random_j or j in seen_items:
                    j = np.random.randint(data.item_nums)
                random_j.add(j)
                # Negative instances
                u_features.append(u)
                i_features.append(j)
                labels.append(0.0)
                if fism_like:
                    u_neighbors_num.append(len(seen_items))
    train_nums = len(u_features)
    train_batches = math.ceil(train_nums/batch_size)
    # Shuffle the features
    s_idx = np.random.permutation(train_nums)
    u_features, i_features, labels = np.array(u_features)[s_idx], np.array(i_features)[s_idx], np.array(labels)[s_idx]
    if fism_like:
        u_neighbors_num = np.array(u_neighbors_num)[s_idx]
        return train_batches, u_features, i_features, labels, u_neighbors_num
    else:
        return train_batches, u_features, i_features, labels

# Get training instances for pairwise learning
def pairwise_ranking_sampler(data, neg_ratio, batch_size, fism_like=False):
    u_features, i_features, j_features = [], [], []
    if fism_like:
        u_neighbors_num = []
    for u, items in data.ui_train.items():
        seen_items = set(data.ui_train[u])
        for i in items:
            random_j = set()
            for s in range(neg_ratio):
                u_features.append(u)
                i_features.append(i)
                # Sample negative items
                j = np.random.randint(data.item_nums)
                while j in random_j or j in seen_items:
                    j = np.random.randint(data.item_nums)
                random_j.add(j)
                j_features.append(j)
                if fism_like:
                    u_neighbors_num.append(len(seen_items))
    train_nums = len(u_features)
    train_batches = math.ceil(train_nums/batch_size)
    # Shuffle the features
    s_idx = np.random.permutation(train_nums)
    u_features, i_features, j_features = np.array(u_features)[s_idx], np.array(i_features)[s_idx], np.array(j_features)[s_idx]
    if fism_like:
        u_neighbors_num = np.array(u_neighbors_num)[s_idx]
        return train_batches, u_features, i_features, j_features, u_neighbors_num
    else:
        return train_batches, u_features, i_features, j_features

# For CML
def ranking_sampler_cml(data, neg_ratio, batch_size):
    u_features, i_features, neg_items = [], [], []
    for u, items in data.ui_train.items():
        seen_items = set(data.ui_train[u])
        for i in items:
            u_neg_items = []
            u_features.append(u)
            i_features.append(i)
            # Sample negative items
            random_j = set()
            for s in range(neg_ratio):
                j = np.random.randint(data.item_nums)
                while j in random_j or j in seen_items:
                    j = np.random.randint(data.item_nums)
                random_j.add(j)
                u_neg_items.append(j)
            neg_items.append(u_neg_items)
    train_nums = len(u_features)
    train_batches = math.ceil(train_nums/batch_size)
    # Shuffle the features
    s_idx = np.random.permutation(train_nums)
    u_features, i_features, neg_items = np.array(u_features)[s_idx], np.array(i_features)[s_idx], np.array(neg_items)[s_idx]
    return train_batches, u_features, i_features, neg_items

# For SBPR
def ranking_sampler_sbpr(data, SPu, neg_ratio, batch_size):
    u_features, i_features, i_s_features, i_neg_features, suk_features = [], [], [], [], []
    for u, items in data.ui_train.items():
        if u not in SPu:
            continue
        tru, spu = set(items), set(SPu[u])
        for i in items:
            # Sample from SPu and Nu
            for j in range(neg_ratio):
                u_features.append(u)
                i_features.append(i)
                # Sample social item
                s = np.random.randint(len(spu))
                i_s_features.append(SPu[u][s])
                # Sample negative item
                neg = np.random.randint(data.item_nums)
                while neg in tru or neg in spu:
                    neg = np.random.randint(data.item_nums)
                i_neg_features.append(neg)

                # Calculate suk (Social coefficient)
                suk = 0 # The number of u's friend who consumed sample s while u didn't
                for friend in data.user_friends[u]:
                    if friend not in data.ui_train:
                        continue
                    if SPu[u][s] in data.ui_train[friend]:
                        suk += 1
                suk_features.append(suk)

    train_nums = len(u_features)
    train_batches = math.ceil(train_nums/batch_size)
    # Shuffle the features
    shuffle_indices = np.random.permutation(np.arange(train_nums))
    u_features, i_features, i_s_features, i_neg_features, suk_features = np.array(u_features)[shuffle_indices], np.array(i_features)[shuffle_indices], \
        np.array(i_s_features)[shuffle_indices], np.array(i_neg_features)[shuffle_indices], np.array(suk_features)[shuffle_indices]
    return train_batches, u_features, i_features, i_s_features, i_neg_features, suk_features

# For RML-DGATs
def ranking_sampler_sohrml(data, neg_ratio):
    # Item domain
    u_features_i, i_features, j_features = [], [], []
    for u, items in data.ui_train.items():
        seen_items = set(items)
        for i in items:
            random_j = set()
            for s in range(neg_ratio):
                u_features_i.append(u)
                i_features.append(i)
                j = np.random.randint(data.item_nums)
                while j in random_j or j in seen_items:
                    j = np.random.randint(data.item_nums)
                random_j.add(j)
                j_features.append(j)
    # Social domain
    u_features_s, v_features, w_features = [], [], []
    for u, friends in data.user_friends.items():
        seen_friends = set(friends)
        for v in friends:
            random_w = set()
            for s in range(neg_ratio):
                u_features_s.append(u)
                v_features.append(v)
                w = np.random.randint(data.user_nums)
                while w in random_w or w in seen_friends:
                    w = np.random.randint(data.user_nums)
                random_w.add(w)
                w_features.append(w)
    # Shuffle the features
    train_nums_i, train_nums_s = len(u_features_i), len(u_features_s)
    s_idx_i, s_idx_s = np.random.permutation(train_nums_i), np.random.permutation(train_nums_s)
    u_features_i, i_features, j_features = np.array(u_features_i)[s_idx_i], np.array(i_features)[s_idx_i], np.array(j_features)[s_idx_i]
    u_features_s, v_features, w_features = np.array(u_features_s)[s_idx_s], np.array(v_features)[s_idx_s], np.array(w_features)[s_idx_s]
    return u_features_i, i_features, j_features, u_features_s, v_features, w_features, train_nums_i, train_nums_s