# coding: utf-8

" CML: Collaborative Metric Learning (2017). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

class CML(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(CML, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg, self.margin = int(configs['embed_size']), float(configs['reg']), float(configs['margin'])
        logger.info(' model_params: embed_size=%d, reg=%s, margin=%s' % (self.embed_size, self.reg, self.margin) + ', ' + self.model_params)
        # Specify training/testing model
        self.train_model = self.train_model_cml

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('cml_inputs'):
            self.u_idx, self.i_idx = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx')
            self.neg_items = __create_p(tf.int32, [None, self.neg_ratio], 'neg_items')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('cml_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed, self.i_embed = tf.gather(self.P, self.u_idx), tf.gather(self.Q, self.i_idx)
            self.neg_embed = tf.transpose(tf.gather(self.Q, self.neg_items), (0, 2, 1)) # [batch_size, embed_size, neg_ratio]

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Calculate the distances
            self.u_i_dist = tf.reduce_sum(tf.squared_difference(self.u_embed, self.i_embed), 1)
            self.u_neg_dist = tf.reduce_sum(tf.squared_difference(tf.expand_dims(self.u_embed, -1), self.neg_embed), 1) # [batch_size, neg_ratio]
            # Get the minimize distance of all negative items
            u_neg_dist_min = tf.reduce_min(self.u_neg_dist, 1)
            loss_per_pair = tf.maximum(self.u_i_dist + self.margin - u_neg_dist_min, 0)

            ## WARP Loss
            # Imposters indicator
            imposters = (tf.expand_dims(self.u_i_dist, -1) + self.margin - self.u_neg_dist) > 0
            # Approximate the rank of positive item
            rank = tf.reduce_mean(tf.cast(imposters, dtype=tf.float32), 1)*self.data.item_nums/self.neg_ratio
            loss_per_pair *= tf.log(rank+1)

            # Optimize
            cov_loss = self._get_covariance_loss()
            self.loss = tf.reduce_sum(loss_per_pair) + cov_loss
            self.train = self.optimizer.minimize(self.loss)

            # Unit clipping
            self._unit_clipping()

    def _get_covariance_loss(self):
        with tf.name_scope('covariance_loss'):
            X = tf.concat((self.Q, self.P), 0)
            n_rows = tf.cast(tf.shape(X)[0], tf.float32)
            X = X - (tf.reduce_mean(X, axis=0))
            cov = tf.matmul(X, X, transpose_a=True)/n_rows
            cov_loss = self.reg * tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_size, tf.float32)))
            return cov_loss

    def _unit_clipping(self):
        with tf.name_scope('unit_clipping'):
            # self.P = tf.clip_by_norm(self.P, 1.0, axes=[1])
            # self.Q = tf.clip_by_norm(self.Q, 1.0, axes=[1])
            self.u_embed = tf.clip_by_norm(self.u_embed, 1.0, axes=[1])
            self.i_embed = tf.clip_by_norm(self.i_embed, 1.0, axes=[1])
            self.neg_embed = tf.clip_by_norm(self.neg_embed, 1.0, axes=[1])

    def _predict(self):
        if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
            self.pre_scores = self.u_i_dist
        else:
            self.pre_scores = tf.reduce_sum(tf.squared_difference(tf.expand_dims(self.u_embed, 1), self.Q), 2)

    def _save_model(self):
        var_list = {'cml_params/P': self.P, 'cml_params/Q': self.Q}
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
