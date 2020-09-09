# coding: utf-8

" FM: Factorized Machine (2010). "

import tensorflow as tf
from model.RatingRecommender import RatingRecommender
from utils.tools import get_loss
import os

class FM(RatingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(FM, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg = int(configs['embed_size']), float(configs['reg'])
        logger.info(' model_params: embed_size=%d, reg=%s' % (self.embed_size, self.reg) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('fm_inputs'):
            self.x_idx = __create_p(tf.int32, [None, None], 'x_idx') # Input idx
            if self.is_real_valued:
                self.x_value = __create_p(tf.float32, [None, None], 'x_value') # Input value
            self.y = __create_p(tf.float32, [None], 'y') # Labels

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('fm_params'):
            self.w0 = tf.Variable(tf.constant(0.0), name='w0') # bias
            self.wi = __create_w([self.data.feature_nums+1], 'wi') # Linear part (0 as the invalid id)
            self.vif = __create_w([self.data.feature_nums+1, self.embed_size], 'vif') # Feature embeddings (2nd-order part)

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.wi_embed = tf.gather(self.wi, self.x_idx) # [batch_size, feature_nums]
            self.vif_embed = tf.gather(self.vif, self.x_idx) # [batch_size, feature_nums, embed_size]

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Calculate prediction scores
            if self.is_real_valued:
                # Consider real values
                squared_sum_embed = tf.square(tf.reduce_sum(tf.einsum('ab,abc->abc', self.x_value, self.vif_embed), 1))
                sum_squared_embed = tf.reduce_sum(tf.einsum('ab,abc->abc', tf.square(self.x_value), tf.square(self.vif_embed)), 1)
            else:
                squared_sum_embed = tf.square(tf.reduce_sum(self.vif_embed, 1))
                sum_squared_embed = tf.reduce_sum(tf.square(self.vif_embed), 1)
            y_2nd = tf.reduce_sum(squared_sum_embed - sum_squared_embed, 1)
            self.y_pre = self.w0 + tf.reduce_sum(self.wi_embed, 1) + 0.5 * y_2nd

            # Optimize
            self.loss = get_loss(self.loss_func, self.y, logits=self.y_pre) + self.reg * (tf.nn.l2_loss(self.wi) + tf.nn.l2_loss(self.vif))
            self.train = self.optimizer.minimize(self.loss)
        
    # def _predict(self):
    #     with tf.name_scope('prediction'):
    #         self.pre_scores = self.y_pre

    def _save_model(self):
        pass

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        # self._predict()
        # self._save_model()