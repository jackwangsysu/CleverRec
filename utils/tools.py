# coding: utf-8

import numpy as np, scipy.sparse as sp, tensorflow as tf
import gensim.models.word2vec as word2vec
import os, sys, functools, time, logging
from collections import defaultdict

# Reindex ids
def re_index(data_set):
    data_map = {}
    id = 0
    for d in data_set:
        data_map[d] = id
        id += 1
    return data_map

# Time cost decorator
def timer(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t1 = time.time()
            print('Start %s...' % text)
            res = func(*args, **kwargs)
            print('%s done, time: %s' % (text, time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))))
            return res
        return wrapper
    return decorator

# Logger
def get_logger(log_dir, model):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Log into file
    fh = logging.FileHandler(os.path.join(log_dir, model+'.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # Log on console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # Add to handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Initializer
def get_initializer(init_method, stddev=None):
    initializer = None
    if init_method == 'normal':
        initializer = tf.random_normal_initializer(stddev=stddev)
    elif init_method == 'tnormal':
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    elif init_method == 'uniform':
        initializer = tf.random_uniform_initializer(-stddev, stddev)
    elif init_method == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
    elif init_method == 'xavier_normal':
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    return initializer

# Loss function
def get_loss(loss_func, y, logits=None, margin=None):
    loss = 0
    if loss_func == 'cross_entropy':
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)) # reduce_sum/reduce_mean
    elif loss_func == 'bpr':
        loss = tf.reduce_sum(-tf.log_sigmoid(y))
    elif loss_func == 'hinge':
        loss = tf.reduce_sum(tf.maximum(y+margin, 0))
    elif loss_func == 'square':
        loss = tf.reduce_sum(tf.squared_difference(y, logits))
    return loss

# Optimizer
def get_optimizer(optimizer, lr):
    optim = None
    if optimizer == 'SGD':
        optim = tf.train.GradientDescentOptimizer(lr)
    elif optimizer == 'Adam':
        optim = tf.train.AdamOptimizer(lr)
    elif optimizer == 'Adagrad':
        optim = tf.train.AdagradOptimizer(lr)
    return optim

# Generate user-item sparse matrix (For FISM)
def get_ui_sp_mat(data):
    ui_indices_list, ui_values_list = [], []
    for u, items in data.ui_train.items():
        for i in items:
            ui_indices_list.append([u, i])
            ui_values_list.append(1.0/len(items))
    ui_sp_mat = tf.SparseTensor(ui_indices_list, ui_values_list, [data.user_nums+1, data.item_nums+1])
    return ui_sp_mat

# Generate user-item sparse matrix (For TransCF)
def get_sp_mat(data):
    ui_indices_list, ui_values_list, iu_indices_list, iu_values_list = [], [], [], []
    iu_nums = defaultdict(int)
    for u, items in data.ui_train.items():
        for i in items:
            ui_indices_list.append([u, i])
            ui_values_list.append(1.0/len(items))
            iu_indices_list.append([i, u])
            iu_nums[i] +=1
    for indice in iu_indices_list:
        iu_values_list.append(1.0/iu_nums[indice[0]])
    ui_sp_mat = tf.SparseTensor(ui_indices_list, ui_values_list, [data.user_nums, data.item_nums])
    iu_sp_mat = tf.SparseTensor(iu_indices_list, iu_values_list, [data.item_nums, data.user_nums])
    return ui_sp_mat, iu_sp_mat

# Get SPu for SBPR
def get_SPu(data):
    SPu = {}
    for u in data.ui_train:
        u_s_items = set()
        if u in data.user_friends:
            for friend in data.user_friends[u]:
                if friend not in data.ui_train:
                    continue
                u_s_items = u_s_items.union(set(data.ui_train[friend])).difference(set(data.ui_train[u]))
            if u_s_items: # not empty
                SPu[u] = list(u_s_items)
    return SPu

# Get the topK latent friends and SPu
def get_topK_friends_and_SPu(data, walk_count, walk_length, walk_dim, window_size, topK_f):
    t1 = time.time()
    # Generate CUNet
    CUNet = defaultdict(list) # Collaborative User Net
    u_neighbors = defaultdict(dict)
    for u1 in data.ui_train:
        for u2 in data.ui_train:
            if u1 != u2:
                weight = len(set(data.ui_train[u1]).intersection(set(data.ui_train[u2])))
                if weight > 0:
                    CUNet[u1].extend([u2]*weight)
                    u_neighbors[u1][u2] = weight
    print('CUNet done!')
    print('Cost time(CUNet): ', time.strftime('%H: %M: %S', time.gmtime(time.time()-t1)))

    t2 = time.time()
    # Generate random deep walks
    deep_walks = []
    visited = defaultdict(dict)
    for u in CUNet:
        for t in range(walk_count):
            walk_path = [str(u)]
            last_node = u
            for i in range(1, walk_length):
                unvisited_neighbors = list((set(u_neighbors[last_node].keys())).difference(set(visited[last_node])))
                if len(unvisited_neighbors) == 0: # All neighbors have been visited
                    next_node = np.random.choice(CUNet[last_node]) # Randomly choose one
                else:
                    # Select the next node
                    all_weights = [u_neighbors[last_node][neighbor] for neighbor in unvisited_neighbors]
                    max_id = all_weights.index(max(all_weights))
                    next_node = unvisited_neighbors[max_id]
                walk_path.append(str(next_node))
                visited[u][next_node] = 1 # Mark as visited
                last_node = next_node
            deep_walks.append(walk_path)
    # Shuffle deep walks
    np.random.shuffle(deep_walks)
    print('Deep walks done!')
    print('Cost time(DeepWalk): ', time.strftime('%H: %M: %S', time.gmtime(time.time()-t2)))

    t3 = time.time()
    # Generate user embeddings by word2vec
    model = word2vec.Word2Vec(deep_walks, size=walk_dim, window=window_size, min_count=0, iter=3)
    # May get error：TypeError: ufunc 'add' did not contain a loop with signature matching types; <-- Solution：walk_path = [str(u)]
    print('User embeddings done!')
    print('Cost time(word2vec): ', time.strftime('%H: %M: %S', time.gmtime(time.time()-t3)))

    # Calculate cosine similarity
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return (a.dot(b))/(np.linalg.norm(a)*np.linalg.norm(b))

    t4 = time.time()
    # Construct the user similartiy matrix
    topK_friends = {}
    for u1 in CUNet:
        sims = []
        for u2 in CUNet:
            if u1 != u2:
                sims.append([u2, cosine_sim(model.wv[str(u1)], model.wv[str(u2)])])
        topK_friends[u1] = list(np.array(sorted(sims, key=lambda s: s[1], reverse=True)[:topK_f])[:,0])
    print('TopK semantic friends done!')
    print('Cost time(topK): ', time.strftime('%H: %M: %S', time.gmtime(time.time()-t4)))

    # Get SPu
    t5 = time.time()
    SPu = {}
    for u in data.ui_train:
        u_s_items = set()
        if u in topK_friends:
            for friend in topK_friends[u]:
                if friend not in data.ui_train:
                    continue
                u_s_items = u_s_items.union(set(data.ui_train[friend])).difference(set(data.ui_train[u]))
            if u_s_items: # not empty
                SPu[u] = list(u_s_items)
    print('SPu done!')
    print('Cost time(SPu): ', time.strftime('%H: %M: %S', time.gmtime(time.time()-t5)))
    return topK_friends, SPu

# Generate the user/item neighbors (For RML-DGATs)
def get_neighbors_rml_dgats(data, max_i, max_s):
    u_max_i, u_max_s = max(len(data.ui_train[u]) for u in data.ui_train), max(len(data.user_friends[u]) for u in data.user_friends)
    u_nums_i, u_nums_s = max_i if 0 < max_i < u_max_i else u_max_i, max_s if 0 < max_s < u_max_s else u_max_s
    user_nbrs_i, user_nbrs_s = np.zeros((data.user_nums, u_nums_i), np.int32), np.zeros((data.user_nums, u_nums_s), np.int32)
    
    # Item domain
    iu_train = defaultdict(list)
    # For user
    for u, items in data.ui_train.items():
        for i in items:
            iu_train[i].append(u)
        if 0 < u_nums_i < len(items):
            tmp_items = np.random.choice(items, size=u_nums_i, replace=False).tolist()
        else:
            tmp_items = items + [data.item_nums]*(u_nums_i-len(items))
        user_nbrs_i[u] = tmp_items
    # For items
    i_max = max(len(iu_train[i]) for i in iu_train)
    i_nums_u = max_i if 0 < max_i < i_max else i_max
    item_nbrs = np.zeros((data.item_nums, i_nums_u), np.int32)
    for i, users in iu_train.items():
        if 0 < i_nums_u < len(users):
            tmp_users = np.random.choice(users, size=i_nums_u, replace=False).tolist()
        else:
            tmp_users = users + [data.user_nums]*(i_nums_u-len(users))
        item_nbrs[i] = tmp_users

    # Social domain
    for u, friends in data.user_friends.items():
        if 0 < u_nums_s < len(friends):
            tmp_friends = np.random.choice(friends, size=u_nums_s, replace=False).tolist()
        else:
            tmp_friends = friends + [data.user_nums]*(u_nums_s-len(friends))
        user_nbrs_s[u] = tmp_friends
    return tf.constant(user_nbrs_i, tf.int32), tf.constant(item_nbrs, tf.int32), tf.constant(user_nbrs_s, tf.int32), tf.constant(list(iu_train.keys()))

# Generate the item adjacency matrix (For SoHRML)
def get_adj_mat_i(data, max_i):
    R_u, R_i = sp.dok_matrix((data.user_nums, data.item_nums), dtype=np.float32), sp.dok_matrix((data.item_nums, data.user_nums), dtype=np.float32)
    total_nums = data.user_nums + data.item_nums
    adj_mat_i = sp.dok_matrix((total_nums, total_nums), dtype=np.float32)
    iu_train = defaultdict(list)
    for u, items in data.ui_train.items():
        for i in items:
            iu_train[i].append(u)
        # Sample neighboring items
        if 0 < max_i < len(items):
            tmp_items = np.random.choice(items, size=max_i, replace=False).tolist()
        else:
            tmp_items = items
        for i in tmp_items:
            R_u[u, i] = 1.0
    for i, users in iu_train.items():
        # Sample neighboring users
        if 0 < max_i < len(users):
            tmp_users = np.random.choice(users, size=max_i, replace=False).tolist()
        else:
            tmp_users = users
        for u in tmp_users:
            R_i[i, u] = 1.0
    R_u, R_i, adj_mat_i = R_u.tolil(), R_i.tolil(), adj_mat_i.tolil()

    # Create the adj_mat_i
    adj_mat_i[:data.user_nums, data.user_nums:] = R_u # R
    adj_mat_i[data.user_nums:, :data.user_nums] = R_i # R.T
    adj_mat_i = (adj_mat_i + sp.eye(adj_mat_i.shape[0])).tocoo() # A + I

    # Row/column indices
    all_r_list_i, all_c_list_i = list(adj_mat_i.row), list(adj_mat_i.col)
    return adj_mat_i.tocsr(), all_r_list_i, all_c_list_i
    
# Generate the social adjacency matrix (For SoHRML)
def get_adj_mat_s(data, max_s):
    adj_mat_s = sp.dok_matrix((data.user_nums, data.user_nums), dtype=np.float32)
    for u, friends in data.user_friends.items():
        # Sample neighboring friends
        if 0 < max_s < len(friends):
            tmp_friends = np.random.choice(friends, size=max_s, replace=False).tolist()
        else:
            tmp_friends = friends
        for v in tmp_friends:
            adj_mat_s[u, v] = 1.0
    adj_mat_s = (adj_mat_s + sp.eye(adj_mat_s.shape[0])).tocoo() # T + I

    # Row/column indices
    all_r_list_s, all_c_list_s = list(adj_mat_s.row), list(adj_mat_s.col)
    return adj_mat_s.tocsr(), all_r_list_s, all_c_list_s