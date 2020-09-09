# coding: utf-8

" TransCF: Translational Collaborative Filtering (2018). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss, get_sp_mat
import os

class TransCF(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(TransCF, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg1, self.reg2, self.margin = int(configs['embed_size']), float(configs['reg1']), float(configs['reg2']), float(configs['margin'])
        logger.info(' model_params: embed_size=%d, reg1=%s, reg2=%s, margin=%s' % (self.embed_size, self.reg1, self.reg2, self.margin) + ', ' + self.model_params)
        # Generate ui_sp_mat and iu_sp_mat
        self.ui_sp_mat, self.iu_sp_mat = get_sp_mat(data)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('bpr_inputs'):
            self.u_idx, self.i_idx, self.j_idx = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx'), __create_p(tf.int32, [None], 'j_idx')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.name_scope('transcf_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed, self.i_embed, self.j_embed = tf.nn.embedding_lookup(self.P, self.u_idx), tf.nn.embedding_lookup(self.Q, self.i_idx), \
                tf.nn.embedding_lookup(self.Q, self.j_idx)

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Neighborhood aggregation
            all_u_nbr_embed = tf.sparse_tensor_dense_matmul(self.ui_sp_mat, self.Q) # [user_nums, embed_size]
            self.all_i_nbr_embed = tf.sparse_tensor_dense_matmul(self.iu_sp_mat, self.P)
            self.u_nbr_embed = tf.gather(all_u_nbr_embed, self.u_idx) # u's neighborhood-based representation
            self.i_nbr_embed = tf.gather(self.all_i_nbr_embed, self.i_idx)
            self.j_nbr_embed = tf.gather(self.all_i_nbr_embed, self.j_idx)

            # Generate relation vectors
            self.ui_r = tf.einsum('ab,ab->ab', self.u_nbr_embed, self.i_nbr_embed)
            self.uj_r = tf.einsum('ab,ab->ab', self.u_nbr_embed, self.j_nbr_embed)

            # Calculate distance scores
            self.ui_dist = tf.reduce_sum(tf.square(self.u_embed + self.ui_r - self.i_embed), 1)
            self.uj_dist = tf.reduce_sum(tf.square(self.u_embed + self.uj_r - self.j_embed), 1)

            # Optimize
            self.loss = get_loss(self.loss_func, self.ui_dist - self.uj_dist, margin=self.margin)
            self.loss += self._get_regularizations()
            # Optimize
            self.train = self.optimizer.minimize(self.loss)

            # Unit clipping
            self._unit_clipping()

    # Regularizations
    def _get_regularizations(self):
        with tf.name_scope('regularizations'):
            # Neighborhood regularization
            reg_nbr = tf.reduce_sum(tf.square(self.u_embed - self.u_nbr_embed)) + tf.reduce_sum(tf.square(self.i_embed - self.i_nbr_embed))
            # Distance regularization
            reg_dist = tf.reduce_sum(tf.square( self.ui_dist + self.margin - self.uj_dist))
            return self.reg1 * reg_nbr + self.reg2 * reg_dist

    def _unit_clipping(self):
        with tf.name_scope('unit_clipping'):
            self.u_embed = tf.clip_by_norm(self.u_embed, 1.0, axes=1)
            self.i_embed = tf.clip_by_norm(self.i_embed, 1.0, axes=1)
            self.j_embed = tf.clip_by_norm(self.j_embed, 1.0, axes=1)

    def _predict(self):
        if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                self.pre_scores = self.ui_dist
        else:
            self.pre_scores = tf.reduce_sum(tf.square(tf.expand_dims(self.u_embed, 1) + \
                tf.einsum('ac,bc->abc', self.u_nbr_embed, self.all_i_nbr_embed) - \
                    tf.expand_dims(self.Q, 0)), 2) # self.P + self.ui_r - self.Q

    def _save_model(self):
        var_list = {'transcf_params/P': self.P, 'transcf_params/Q': self.Q}
        self.saver = tf.train.Saver(var_list=var_list)
        tmp_dir = os.path.join(self.saved_model_dir, self.model)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        self._predict()
        # self._save_model()
