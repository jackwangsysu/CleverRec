# coding: utf-8

" SoHRML: Relational Metric Learning with High-Order Neighborhood Interactions for Social Recommendation. "

import numpy as np, tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os, math

class SoHRML(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(SoHRML, self).__init__(sess, data, configs, logger)
        self.embed_size, self.atten_size, self.gamma, self.reg1, self.reg2, self.margin = int(configs['embed_size']), int(configs['atten_size']), \
            float(configs['gamma']), float(configs['reg1']), float(configs['reg2']), float(configs['margin'])
        self.att_type, self.mlp_type, self.gat_layer_nums = int(configs['att_type']), int(configs['mlp_type']), int(configs['gat_layer_nums'])
        self.node_dropout, self.message_dropout = float(configs['node_dropout']), float(configs['message_dropout'])
        self.adj_folds, self.train_batches = int(configs['adj_folds']), int(configs['train_batches'])
        logger.info(' model_params: embed_size=%d, atten_size=%d, att_type=%d, mlp_type=%d, gat_layer_nums=%d, gamma=%s, reg1=%s, reg2=%s, margin=%s, node_dropout=%s, message_dropout=%s' % \
            (self.embed_size, self.atten_size, self.att_type, self.mlp_type, self.gat_layer_nums, self.gamma, self.reg1, self.reg2, self.margin, self.node_dropout, self.message_dropout) + \
                ', ' + self.model_params)
        # Specify training model
        self.train_model = self.train_model_sohrml

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('sohrml_inputs'):
            # Item domain
            self.u_idx = __create_p(tf.int32, [None], 'u_idx')
            self.i_idx =  __create_p(tf.int32, [None], 'i_idx')
            self.j_idx =  __create_p(tf.int32, [None], 'j_idx') # Negative item
            # Social domain
            self.u_idx_s = __create_p(tf.int32, [None], 'u_idx_s')
            self.v_idx = __create_p(tf.int32, [None], 'v_idx')
            self.w_idx = __create_p(tf.int32, [None], 'w_idx') # Negative social connection
            # For the sparse attentive matrix
            self.r_batch_i, self.c_batch_i, self.r_batch_s, self.c_batch_s = __create_p(tf.int32, [None], 'r_batch_i'), __create_p(tf.int32, [None], 'c_batch_i'), \
                __create_p(tf.int32, [None], 'r_batch_s'), __create_p(tf.int32, [None], 'c_batch_s')
            self.att_scores_i, self.att_scores_s = __create_p(tf.float32, [None], 'att_scores_i'), __create_p(tf.float32, [None], 'att_scores_s')
            self.is_train = __create_p(tf.int16, [], 'is_train')

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_))
        with tf.name_scope('sohrml_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')

            # Attention network params
            self.W = __create_w([2*self.embed_size, self.atten_size], 'W')
            self.h = __create_b([self.atten_size], 'h')
            self.b = __create_b([self.atten_size], 'b')

            # GAT params
            self.gat_params = {}
            for id in range(self.gat_layer_nums):
                self.gat_params['W_gat_'+str(id)] = __create_w([self.embed_size, self.embed_size], 'W_gat_'+str(id))
                self.gat_params['b_gat_'+str(id)] = __create_b([self.embed_size], 'b_gat_'+str(id))

            # MLP params
            self.mlp_params = {}
            for id in range(self.mlp_type):
                self.mlp_params['W_mlp_'+str(id)] = __create_w([2*self.embed_size, min(self.mlp_type-id, 2)*self.embed_size], 'W_mlp_'+str(id))
                self.mlp_params['b_mlp_'+str(id)] = __create_b([min(self.mlp_type-id, 2)*self.embed_size], 'b_mlp_'+str(id))

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            # Item domain
            self.u_embed_i = tf.gather(self.P, self.u_idx)
            self.i_embed = tf.gather(self.Q, self.i_idx)
            self.j_embed = tf.gather(self.Q, self.j_idx)
            # Social domain
            self.u_embed_s = tf.gather(self.P, self.u_idx_s)
            self.v_embed = tf.gather(self.P, self.v_idx)
            self.w_embed = tf.gather(self.P, self.w_idx)

    # Calculate the attention scores
    def _get_att_scores(self):
        with tf.name_scope('att_scores'):
            ego_embed_i = tf.concat([self.P, self.Q], axis=0)
            r_embed_i = tf.nn.embedding_lookup(ego_embed_i, self.r_batch_i)
            c_embed_i = tf.nn.embedding_lookup(ego_embed_i, self.c_batch_i)
            r_embed_s = tf.nn.embedding_lookup(self.P, self.r_batch_s)
            c_embed_s = tf.nn.embedding_lookup(self.P, self.c_batch_s)
            if self.att_type == 0:
                self.att_scores_i_batch = tf.einsum('ab,ab->a', r_embed_i, c_embed_i)
                self.att_scores_s_batch = tf.einsum('ab,ab->a', r_embed_s, c_embed_s)
            elif self.att_type == 1:
                self.att_scores_i_batch = tf.nn.relu(tf.nn.dropout(tf.einsum('ab,ab->a', r_embed_i, c_embed_i), 0.7))
                self.att_scores_s_batch = tf.nn.relu(tf.nn.dropout(tf.einsum('ab,ab->a', r_embed_s, c_embed_s), 0.7))
            elif self.att_type == 2:
                self.att_scores_i_batch = tf.einsum('ac,c->a', tf.nn.relu( \
                    tf.nn.dropout(tf.einsum('ab,bc->ac', tf.concat([r_embed_i, c_embed_i], 1), self.W) + self.b, 0.7)), \
                        self.h)
                self.att_scores_s_batch = tf.einsum('ac,c->a', tf.nn.relu( \
                    tf.nn.dropout(tf.einsum('ab,bc->ac', tf.concat([r_embed_s, c_embed_s], 1), self.W) + self.b, 0.7)), \
                        self.h)

    # Update the sparse attentive adjacency matrix
    def _get_att_weights(self):
        with tf.name_scope('att_weights'):
            indices_i = np.mat([self.all_r_list_i, self.all_c_list_i]).transpose()
            indices_s = np.mat([self.all_r_list_s, self.all_c_list_s]).transpose()
            self.adj_att_i_out = tf.sparse.softmax(tf.SparseTensor(indices_i, self.att_scores_i, self.adj_i_shape))
            self.adj_att_s_out = tf.sparse.softmax(tf.SparseTensor(indices_s, self.att_scores_s, self.adj_s_shape))

    # Generate relation vectors
    def _build_mlp(self, nbr_embed_1, nbr_embed_2):
        with tf.name_scope('build_mlp'):
            if self.mlp_type == 0: # Element-wise product
                r_vec = tf.multiply(nbr_embed_1, nbr_embed_2)
            else: # 1 to 3 layers MLP
                r_vec = tf.concat([nbr_embed_1, nbr_embed_2], 1)
                for id in range(self.mlp_type):
                    r_vec = tf.nn.relu(tf.matmul(r_vec, self.mlp_params['W_mlp_'+str(id)]) + self.mlp_params['b_mlp_'+str(id)])
            return r_vec

    # Split the adjacency matrix
    def split_adj_mat(self, adj_mat):
        adj_mat_fold = []
        fold_len = math.ceil(adj_mat.shape[0]/self.adj_folds)
        for id in range(self.adj_folds):
            start_idx = id*fold_len
            end_idx = adj_mat.shape[0] if id == self.adj_folds-1 else (id+1)*fold_len
            adj_i = adj_mat[start_idx:end_idx].tocoo()
            # Sparse representations
            indices = np.mat([adj_i.row, adj_i.col]).transpose()
            adj_i_sp = tf.cast(tf.SparseTensor(indices, adj_i.data, adj_i.shape), tf.float32)

            # Node dropout (Randomly block a particular node and discard all its outgoing messages)
            adj_i_sp = tf.cond(self.is_train > 0, lambda: self.node_dropout_(adj_i, adj_i_sp), lambda: adj_i_sp)
            adj_mat_fold.append(adj_i_sp)
        return adj_mat_fold

    # Node dropout
    def node_dropout_(self, adj_i, adj_i_sp):
        nnz_i = adj_i.nnz
        random_tensor = (1.0-self.node_dropout) + tf.random_uniform((nnz_i, ))
        mask_ = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        adj_i_sp = tf.sparse_retain(adj_i_sp, mask_) * tf.div(1.0, 1.0-self.node_dropout)
        return adj_i_sp

    def _build_gat(self):
        with tf.name_scope('build_gat'):
            # Initial embeddings
            ego_embed_i = tf.concat([self.P, self.Q], axis=0)
            ego_embed_s = self.P
            # Split the adjacency matrix
            adj_mat_folds_i, adj_mat_folds_s = self.split_adj_mat(self.adj_att_i), self.split_adj_mat(self.adj_att_s)

            # Multiple layers of GAT
            for k in range(self.gat_layer_nums):
                # A_att*E
                att_embed_i, att_embed_s = [], []
                for id in range(self.adj_folds):
                    att_embed_i.append(tf.sparse_tensor_dense_matmul(adj_mat_folds_i[id], ego_embed_i))
                    att_embed_s.append(tf.sparse_tensor_dense_matmul(adj_mat_folds_s[id], ego_embed_s))
                att_embed_i = tf.concat(att_embed_i, axis=0)
                att_embed_s = tf.concat(att_embed_s, axis=0)

                # With GAT layer
                ego_embed_i = tf.nn.leaky_relu(tf.matmul(att_embed_i, self.gat_params['W_gat_'+str(k)]) + self.gat_params['b_gat_'+str(k)])
                ego_embed_s = tf.nn.leaky_relu(tf.matmul(att_embed_s, self.gat_params['W_gat_'+str(k)]) + self.gat_params['b_gat_'+str(k)])
                # # Without GAT layer
                # ego_embed_i = att_embed_i
                # ego_embed_s = att_embed_s

                ego_embed_i = tf.cond(self.is_train > 0, lambda: tf.nn.dropout(ego_embed_i, keep_prob=1.0-self.message_dropout), lambda: ego_embed_i)
                ego_embed_s = tf.cond(self.is_train > 0, lambda: tf.nn.dropout(ego_embed_s, keep_prob=1.0-self.message_dropout), lambda: ego_embed_s)

            # Get refined embeddings
            u_g_embed_i, i_g_embed_i = tf.split(ego_embed_i, [self.data.user_nums, self.data.item_nums], axis=0)
            u_g_embed_s = ego_embed_s

            # Neighborhood-based representations of users/items
            # Item domain
            self.u_nbr_embed_i = tf.gather(u_g_embed_i, self.u_idx)
            self.i_nbr_embed, self.j_nbr_embed = tf.gather(i_g_embed_i, self.i_idx), tf.gather(i_g_embed_i, self.j_idx)
            # Social domain
            self.u_nbr_embed_s = tf.gather(u_g_embed_s, self.u_idx_s)
            self.v_nbr_embed, self.w_nbr_embed = tf.gather(u_g_embed_s, self.v_idx), tf.gather(u_g_embed_s, self.w_idx)

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Get neighborhood-based representations
            self._build_gat()

            # Get relation vectors
            self.ui_vec = self._build_mlp(self.u_nbr_embed_i, self.i_nbr_embed)
            self.uj_vec = self._build_mlp(self.u_nbr_embed_i, self.j_nbr_embed)
            self.uv_vec = self._build_mlp(self.u_nbr_embed_s, self.v_nbr_embed)
            self.uw_vec = self._build_mlp(self.u_nbr_embed_s, self.w_nbr_embed)

            # Get distance scores
            self.ui_dist = tf.reduce_sum(tf.square(self.u_embed_i + self.ui_vec - self.i_embed), 1)
            self.uj_dist = tf.reduce_sum(tf.square(self.u_embed_i + self.uj_vec - self.j_embed), 1)
            self.uv_dist = tf.reduce_sum(tf.square(self.u_embed_s + self.uv_vec - self.v_embed), 1)
            self.uw_dist = tf.reduce_sum(tf.square(self.u_embed_s + self.uw_vec - self.w_embed), 1)

            # Loss
            loss_i = get_loss(self.loss_func, self.ui_dist - self.uj_dist, margin=self.margin) # Loss in item domain
            loss_s = get_loss(self.loss_func, self.uv_dist - self.uw_dist, margin=self.margin) # Loss in social domain
            self.loss = loss_i + self.gamma * loss_s
            self.loss += self._get_regularizations()

            # Optimize
            self.train = self.optimizer.minimize(self.loss)

    def _get_regularizations(self):
        with tf.name_scope('regularizations'):
            # Neighborhood regularizations
            reg_nbr_i = tf.reduce_sum(tf.square(self.u_embed_i - self.u_nbr_embed_i)) + tf.reduce_sum(tf.square(self.i_embed - self.i_nbr_embed))
            reg_nbr_s = tf.reduce_sum(tf.square(self.u_embed_s - self.u_nbr_embed_s)) + tf.reduce_sum(tf.square(self.v_embed - self.v_nbr_embed))

            # Distance regularizations
            reg_dist_i = tf.reduce_sum(tf.square(self.ui_dist + self.margin - self.uj_dist))
            reg_dist_s = tf.reduce_sum(tf.square(self.uv_dist + self.margin - self.uw_dist))
            return self.reg1*(reg_nbr_i + reg_nbr_s) + self.reg2*(reg_dist_i + reg_dist_s)

    def _unit_clipping(self):
        with tf.name_scope('unit_clipping'):
            self.u_embed_i = tf.clip_by_norm(self.u_embed_i, 1.0, axes=1)
            self.i_embed = tf.clip_by_norm(self.i_embed, 1.0, axes=1)
            self.j_embed = tf.clip_by_norm(self.j_embed, 1.0, axes=1)
            self.u_embed_s = tf.clip_by_norm(self.u_embed_s, 1.0, axes=1)
            self.v_embed = tf.clip_by_norm(self.v_embed, 1.0, axes=1)
            self.w_embed = tf.clip_by_norm(self.w_embed, 1.0, axes=1)

    def _predict(self):
        if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
            self.pre_scores = self.ui_dist
        else:
            pass

    def _save_model(self):
        pass

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._get_att_scores()
        self._get_att_weights()
        self._create_inference()
        self._predict()
        # self._save_model()