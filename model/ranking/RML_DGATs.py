# coding: utf-8

" RML-DGATs: Relational Metric Learning with Dual Graph Attention Networks for Social Recommendation. "

import numpy as np, tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss, get_neighbors_rml_dgats
import os, math

class RML_DGATs(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(RML_DGATs, self).__init__(sess, data, configs, logger)
        self.embed_size, self.atten_size, self.gamma, self.reg1, self.reg2, self.margin = int(configs['embed_size']), int(configs['atten_size']), \
            float(configs['gamma']), float(configs['reg1']), float(configs['reg2']), float(configs['margin'])
        self.att_type, self.mlp_type = int(configs['att_type']), int(configs['mlp_type'])
        self.max_i, self.max_s = int(configs['max_i']), int(configs['max_s'])
        self.train_batches = int(configs['train_batches'])
        logger.info(' model_params: embed_size=%d, atten_size=%d, att_type=%d, mlp_type=%d, gamma=%s, reg1=%s, reg2=%s, margin=%s' % (self.embed_size, \
            self.atten_size, self.att_type, self.mlp_type, self.gamma, self.reg1, self.reg2, self.margin) + ', ' + self.model_params)
        # Get the neighbors
        self.user_nbrs_i, self.item_nbrs, self.user_nbrs_s, self.item_set = get_neighbors_rml_dgats(data, self.max_i, self.max_s)
        # Specify the training model
        self.train_model = self.train_model_sohrml

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('rml_dgats_inputs'):
            # Item domain
            self.u_idx = __create_p(tf.int32, [None], 'u_idx')
            self.i_idx =  __create_p(tf.int32, [None], 'i_idx')
            self.j_idx =  __create_p(tf.int32, [None], 'j_idx') # Negative item
            # Social domain
            self.u_idx_s = __create_p(tf.int32, [None], 'u_idx_s')
            self.v_idx = __create_p(tf.int32, [None], 'v_idx')
            self.w_idx = __create_p(tf.int32, [None], 'w_idx') # Negative social connection
            self.is_train = __create_p(tf.int16, [], 'is_train')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_')

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_))
        with tf.variable_scope('rml_dgats_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums+1, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums+1, self.embed_size], 'Q')

            # Attention network params
            self.W = __create_w([2*self.embed_size, self.atten_size], 'W')
            self.h = __create_b([self.atten_size], 'h')
            self.b = __create_b([self.atten_size], 'b')

            # GAT params
            self.W_gat = __create_w([self.embed_size, self.embed_size], 'W_gat')

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

    def _build_gat(self, t_nbrs, t_idx, t_embed, t_nums, t_embed_mat, t_max_nbrs):
        with tf.name_scope('build_gat'):
            t_nbrs_idx = tf.gather(t_nbrs, t_idx)
            t_nbrs_existed = tf.cast(tf.not_equal(t_nbrs_idx, t_nums), tf.float32)
            # Neighbors' embeddings
            t_nbrs_embed = tf.einsum('ab,abc->abc', t_nbrs_existed, tf.gather(t_embed_mat, t_nbrs_idx))
            # Add u/i to its neibhors
            t_nbrs_embed = tf.concat([t_nbrs_embed, tf.expand_dims(t_embed, 1)], 1) # [batch_size, max_i+1/max_s+1, embed_size]

            # Calculate the attention weights
            if self.att_type == 0: # <pu, qi>
                t_nbrs_atten = tf.einsum('ac,abc->ab', t_embed, t_nbrs_embed)
            elif self.att_type == 1: # ReLU(<pu, qi>)
                t_nbrs_atten = tf.nn.relu(tf.einsum('ac,abc->ab', t_embed, t_nbrs_embed))
            elif self.att_type == 2: # h.T*ReLU(W'[pu,qi]+b)
                t_nbrs_atten = tf.einsum('abk,k->ab', tf.nn.relu( \
                    tf.nn.dropout(tf.einsum('abc,cd->abd', tf.concat([tf.tile(tf.expand_dims(t_embed, 1), [1, t_max_nbrs+1, 1]), \
                        t_nbrs_embed], 2), self.W) + self.b, 0.7)), self.h)
            # Softmax
            t_nbrs_atten = tf.nn.softmax(t_nbrs_atten)
            # Aggreate neighbors' features
            t_nbr_embed = tf.einsum('ab,abc->ac', t_nbrs_atten, t_nbrs_embed) # u/i's neighborhood-based embedding

            # GAT layer
            t_nbr_embed = tf.nn.leaky_relu(tf.matmul(t_nbr_embed, self.W_gat))
            # t_nbr_embed = tf.cond(self.is_train > 0, lambda: tf.nn.dropout(t_nbr_embed, 0.7), lambda: t_nbr_embed)

            return t_nbr_embed

    # Generate relation vectors
    def _build_mlp(self, nbr_embed_1, nbr_embed_2, is_test=False):
        with tf.name_scope('build_mlp'):
            if self.mlp_type == 0:
                if is_test:
                    r_vec = tf.einsum('ac,bc->abc', nbr_embed_1, nbr_embed_2) # [batch_size_t_, item_nums, embed_size]
                else:
                    r_vec = tf.multiply(nbr_embed_1, nbr_embed_2)
            else:
                if is_test:
                    r_vec = tf.concat([tf.tile(tf.expand_dims(nbr_embed_1, 1), [1, self.data.item_nums, 1]), tf.tile(tf.expand_dims(nbr_embed_2, 0), \
                        [self.batch_size_t_, 1, 1])], 2)
                else:
                    r_vec = tf.concat([nbr_embed_1, nbr_embed_2], 1)
                for id in range(self.mlp_type):
                    r_vec = tf.nn.relu(tf.matmul(r_vec, self.mlp_params['W_mlp_'+str(id)]) + self.mlp_params['b_mlp_'+str(id)])
            return r_vec

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Get neighborhood-based representations
            # Item-level GAT
            self.u_nbr_embed_i = self._build_gat(self.user_nbrs_i, self.u_idx, self.u_embed_i, self.data.item_nums, self.Q, self.max_i)
            # User-level GAT
            self.i_nbr_embed = self._build_gat(self.item_nbrs, self.i_idx, self.i_embed, self.data.user_nums, self.P, self.max_i)
            self.j_nbr_embed = self._build_gat(self.item_nbrs, self.j_idx, self.j_embed, self.data.user_nums, self.P, self.max_i)

            # Friend-level GATs
            self.u_nbr_embed_s = self._build_gat(self.user_nbrs_s, self.u_idx_s, self.u_embed_s, self.data.user_nums, self.P, self.max_s)
            self.v_nbr_embed = self._build_gat(self.user_nbrs_s, self.v_idx, self.v_embed, self.data.user_nums, self.P, self.max_s)
            self.w_nbr_embed = self._build_gat(self.user_nbrs_s, self.w_idx, self.w_embed, self.data.user_nums, self.P, self.max_s)

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
            i_embed_t = tf.gather(self.Q, self.item_set)
            i_nbr_embed_t = self._build_gat(self.item_nbrs, self.item_set, i_embed_t, self.data.user_nums, self.P, self.max_i) # [item_nums, embed_size]
            ui_vec_t = self._build_mlp(self.u_nbr_embed_i, i_nbr_embed_t, is_test=True) # [batch_size_t_, item_nums, embed_size]
            self.pre_scores = tf.reduce_sum(tf.square(tf.expand_dims(self.u_embed_i, 1) + ui_vec_t - tf.expand_dims(i_embed_t, 0)), 2)

    def _save_model(self):
        pass

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        self._predict()
        # self._save_model()