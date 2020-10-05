# coding: utf-8

" LRML: Latent Relational Metric Learning (2018). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

class LRML(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(LRML, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg, self.margin = int(configs['embed_size']), float(configs['reg']), float(configs['margin'])
        self.mem_size = int(configs['mem_size']) # Number of memory slots
        logger.info(' model_params: embed_size=%d, mem_size=%d, reg=%s, margin=%s' % (self.embed_size, self.mem_size, self.reg, self.margin) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('bpr_inputs'):
            self.u_idx, self.i_idx, self.j_idx = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx'), __create_p(tf.int32, [None], 'j_idx')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('lrml_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')
            # Memory network
            self.K = __create_w([self.embed_size, self.mem_size], 'K')
            self.M = __create_w([self.mem_size, self.embed_size], 'M')

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed = tf.gather(self.P, self.u_idx)
            self.i_embed = tf.gather(self.Q, self.i_idx)
            self.j_embed = tf.gather(self.Q, self.j_idx)

    # LRAM Module: calculate relation vectors (Train/Test)
    def _lram(self, user_embed, item_embed, is_test=False):
        if is_test:
            joint_embed = tf.einsum('ac,bc->abc', user_embed, item_embed) # [batch_size, item_nums, embed_size]
        else:
            joint_embed = tf.multiply(user_embed, item_embed)
        key_attention = tf.matmul(joint_embed, self.K)
        atten_weights = tf.nn.softmax(key_attention)
        r_vec = tf.matmul(atten_weights, self.M)
        return r_vec

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Calculate relation vectors
            self.ui_vec = self._lram(self.u_embed, self.i_embed)
            self.uj_vec = self._lram(self.u_embed, self.j_embed)
            # Calculate distance scores
            self.ui_dist = tf.reduce_sum(tf.square(self.u_embed + self.ui_vec - self.i_embed), 1)
            self.uj_dist = tf.reduce_sum(tf.square(self.u_embed + self.uj_vec - self.j_embed), 1)
            # Optimize
            self.loss = get_loss(self.loss_func, self.ui_dist - self.uj_dist, margin=self.margin) + \
                self.reg * (tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.i_embed) + tf.nn.l2_loss(self.j_embed))
            self.train = self.optimizer.minimize(self.loss)

    def _unit_clipping(self):
        with tf.name_scope('unit_clipping'):
            self.P = tf.clip_by_norm(self.P, 1.0, axes=[1])
            self.Q = tf.clip_by_norm(self.Q, 1.0, axes=[1])

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                self.pre_scores = self.ui_dist
            else:
                r_vec_t = self._lram(self.u_embed, self.Q, is_test=True)
                self.pre_scores = tf.reduce_sum(tf.square(tf.expand_dims(self.u_embed, 1) + r_vec_t - tf.expand_dims(self.Q, 0)), 2)

    def _save_model(self):
        var_list = {'lrml_params/P': self.P, 'lrml_params/Q': self.Q, 'lrml_params/K': self.K, 'lrml_params/M': self.M}
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